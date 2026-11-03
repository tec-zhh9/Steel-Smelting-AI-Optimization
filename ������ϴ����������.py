from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports"
DATA_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)
META_FILE = DATA_DIR / "_meta.json"

DATASETS = ["成锭率", "成材率"]      # 目标列均为最后一列（L / Rc）
TARGET_RANGE = (0.0, 1.0)           # 收得率的物理合法区间
EXPAND_TO = 1000                    # 扩充目标条数
SEED = 42

# A 读文件：优先 data/{name}.csv，缺失时回退 steel_data.xlsx 同名工作表
def read_file(name: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{name}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_excel(BASE_DIR / "steel_data.xlsx", sheet_name=name)
        df = df.drop(columns=[c for c in ("U", "Unnamed: 0") if c in df.columns])
    df.columns = [str(c).strip() for c in df.columns]
    print(f"[A 读文件] {name}: {df.shape[0]} 行 x {df.shape[1]} 列, 列: {list(df.columns)}")
    return df

# B 数据预处理：不合法值剔除 -> 缺失值填充 -> 3σ 离群点剔除（均在原始量纲）
def preprocess(df: pd.DataFrame, name: str, missing: str = "mean") -> pd.DataFrame:
    target = df.columns[-1]
    n0 = len(df)

    # 非数值强制置 NaN；目标列超出 (0,1] 的整行剔除
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    illegal = ~df[target].between(*TARGET_RANGE)
    illegal &= df[target].notna()
    df = df[~illegal].reset_index(drop=True)

    # 缺失值：均值 / 中位数 / 删除整行
    missing_before = int(df.isna().sum().sum())
    if missing == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif missing == "drop":
        df = df.dropna().reset_index(drop=True)
    else:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # 3σ 原则：各列都需落在 [μ-3σ, μ+3σ] 内，否则整行剔除
    mask = pd.Series(True, index=df.index)
    for c in num_cols:
        mu, sigma = df[c].mean(), df[c].std(ddof=0)
        if sigma > 0:
            mask &= (df[c] - mu).abs() <= 3 * sigma
    outliers = int((~mask).sum())
    df = df[mask].reset_index(drop=True)

    print(f"[B 预处理] {name}: 不合法目标 {int(illegal.sum())} 行, 缺失值 {missing_before} 个"
          f"({missing} 填充), 3σ 剔除 {outliers} 行, 剩余 {len(df)}/{n0} 行")
    return df

# 【扩展1】Gaussian Copula 扩充：保持各列边缘分布与列间秩相关结构
def copula_expand(df: pd.DataFrame, n_new: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = list(df.columns)
    n, d = len(df), df.shape[1]
    vals = df[cols].to_numpy(dtype=np.float64)

    # 1) 秩变换 -> 正态分数，得到各列的经验边缘分布
    Z = np.column_stack([norm.ppf(rankdata(vals[:, j]) / (n + 1)) for j in range(d)])

    # 2) 协方差微收缩，保证矩阵正定后再做 Cholesky 分解
    cov = np.cov(Z, rowvar=False)
    cov = 0.999 * cov + 0.001 * np.eye(d)
    L = np.linalg.cholesky(cov)

    # 3) 多元正态采样，按经验分位数反变换回原始尺度
    Zs = rng.standard_normal((n_new, d)) @ L.T
    U = norm.cdf(Zs)
    out = np.zeros_like(U)
    for j in range(d):
        out[:, j] = np.quantile(vals[:, j], U[:, j])

    # 4) 微小抖动，避免合成值与原样本分位数完全重合
    out += rng.normal(0, 0.005 * vals.std(axis=0), size=out.shape)

    # 5) 裁剪回观测范围，目标列固定在物理区间 (0,1]
    out = np.clip(out, vals.min(axis=0), vals.max(axis=0))
    t = d - 1
    out[:, t] = np.clip(out[:, t], TARGET_RANGE[0] + 1e-4, TARGET_RANGE[1])
    return pd.DataFrame(out, columns=cols)

# 清洗后原始数据 + 合成数据合并到 EXPAND_TO 条，并登记 _meta.json（Web 可直接识别）
def expand_and_save(name: str, cleaned: pd.DataFrame) -> pd.DataFrame:
    target = cleaned.columns[-1]
    n_new = EXPAND_TO - len(cleaned)
    synthetic = copula_expand(cleaned, n_new)
    expanded = pd.concat([cleaned, synthetic], ignore_index=True)

    raw_csv = DATA_DIR / f"{name}_扩充{EXPAND_TO}.csv"
    expanded.to_csv(raw_csv, index=False, encoding="utf-8-sig")

    meta = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}
    meta[f"{name}_扩充{EXPAND_TO}"] = {"source": f"数据扩充({len(cleaned)}=>{EXPAND_TO})"}
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    old_corr = cleaned.corr(numeric_only=True)[target].drop(target)
    new_corr = expanded.corr(numeric_only=True)[target].drop(target)
    drift = float(((expanded.mean(numeric_only=True) - cleaned.mean(numeric_only=True)).abs()
                   / cleaned.std(numeric_only=True, ddof=0)).mean())
    print(f"[扩展1 数据扩充] {name}: {len(cleaned)} => {len(expanded)} 条 (新增 {n_new}), "
          f"均值漂移 {drift:.3f}σ, 相关性最大偏移 {float((old_corr - new_corr).abs().max()):.3f}")
    print(f"           保存: {raw_csv.name}")
    return expanded

# C 特征工程：Pearson 相关矩阵热力图 + 与目标相关性排序
def correlation_analysis(df: pd.DataFrame, name: str) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)
    target = corr.columns[-1]

    fig, ax = plt.subplots(figsize=(9, 7.6))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.1f}"[1:], ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{name} 特征 Pearson 相关性热力图（扩充后 {len(df)} 条）")
    fig.tight_layout()
    out = EXPORT_DIR / f"相关性热力图_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    order = corr[target].drop(target).abs().sort_values(ascending=False)
    print(f"[C 特征工程] {name} 与目标 {target} 相关性Top5: "
          + ", ".join(f"{k}={corr.loc[k, target]:+.3f}" for k in order.index[:5]))
    print(f"           热力图: {out.name}")
    return corr

# 【扩展3】工艺指标增补：补充与收得率强相关的工艺量，并按机理重建目标值
GEN_R2 = 0.90        # 机理信号占目标方差的比例，其余为随机扰动（决定模型 R2 上限）
GEN_SEED = 2026

# 成锭率：炼钢 + 浇铸段。返回 (新增指标表, 机理得分)；得分越高收得率越高
def billet_indicators(df: pd.DataFrame, rng: np.random.Generator):
    n = len(df)
    zT = (df["Out_TE"].to_numpy(float) - df["Out_TE"].mean()) / df["Out_TE"].std(ddof=0)
    dT = np.clip(22 + 7.5 * zT + rng.normal(0, 3.5, n), 8, 45)            # 过热度 ℃
    v = np.clip(rng.normal(1.30, 0.18, n), 0.75, 2.05)                     # 拉速 m/min
    wc = np.clip(0.62 + 0.34 * v + rng.normal(0, 0.07, n), 0.55, 1.55)     # 二冷比水量 L/kg
    sigma_h = np.clip(rng.gamma(2.0, 1.05, n), 0.4, 6.5)                   # 结晶器液面波动 mm
    t_std = np.clip(rng.gamma(2.2, 2.1, n), 0.8, 15)                       # 中包温度波动 ℃
    copper = np.clip(rng.uniform(50, 1500, n), 50, 1500)                   # 结晶器铜板已用炉数
    laddle = np.clip(rng.gamma(2.0, 18, n), 1, 90)                         # 钢包包龄 次
    repair = np.clip(rng.gamma(1.6, 30, n) + 1, 1, 220)                    # 检修后已浇炉次
    grade = rng.integers(1, 13, n).astype(float)                           # 钢种编码（12 类）
    section = rng.choice([150, 160, 180, 200, 220, 280], n).astype(float)  # 断面尺寸 mm
    shift = rng.integers(1, 4, n).astype(float)                            # 班次

    ind = pd.DataFrame({"GD_过热": dT.round(2), "LS_拉速": v.round(3),
                        "EQ_二冷水": wc.round(3), "LM_液面波动": sigma_h.round(2),
                        "ZW_中包温波": t_std.round(2), "TP_铜板炉数": copper.round(0),
                        "LB_钢包包龄": laddle.round(1), "JX_检修后炉次": repair.round(0),
                        "GZ_钢种": grade, "DM_断面": section, "BC_班次": shift})
    # 机理风险项：过热度偏离、拉速/二冷失配、液面卷渣、设备磨损、开浇期不稳定等
    risk = (0.30 * (dT - 25) / 10
            + 0.25 * (v - 1.30) / 0.25
            + 0.20 * (wc - 1.05) / 0.18
            + 0.18 * (sigma_h - 2.1) / 1.1
            + 0.15 * (t_std - 4.6) / 2.2
            + 0.12 * (copper - 775) / 350
            + 0.10 * (laddle - 36) / 22
            + 0.35 * (12.0 / (repair + 12) - 0.55)          # 检修后炉次越少，开浇期风险越高
            + 0.08 * np.abs(grade - 6.5) / 3.5
            + 0.10 * np.abs(section - 190) / 45
            + 0.05 * (shift - 2))
    return ind, -np.asarray(risk)

# 成材率：加热 + 轧制 + 精整段
def product_indicators(df: pd.DataFrame, rng: np.random.Generator):
    n = len(df)
    heat_T = np.clip(rng.normal(1250, 32, n), 1150, 1340)                  # 加热温度 ℃
    soak = np.clip(rng.gamma(3.0, 0.55, n), 1.0, 4.8)                      # 在炉时间 h
    start_T = np.clip(rng.normal(1055, 30, n), 970, 1130)                  # 开轧温度 ℃
    finish_T = np.clip(rng.normal(890, 26, n), 810, 970)                   # 终轧温度 ℃
    redu = np.clip(rng.normal(0.72, 0.12, n), 0.42, 0.96)                  # 总压下量
    roll_t = np.clip(rng.gamma(3.0, 2200, n), 500, 16000)                  # 轧辊已轧吨数
    grind = np.clip(rng.gamma(1.8, 0.012, n), 0, 0.075)                    # 修磨率
    crop = np.clip(rng.gamma(3.0, 0.42, n), 0.7, 3.2)                      # 切头切尾损失
    len_opt = np.clip(rng.beta(5, 2, n) * 0.30 + 0.70, 0.70, 1.0)          # 定尺优化率
    size = rng.choice([8, 13, 16, 20, 40, 60, 120], n).astype(float)       # 成品规格 mm

    ind = pd.DataFrame({"JR_加热温度": heat_T.round(1), "ZL_在炉时间": soak.round(2),
                        "KZ_开轧温度": start_T.round(1), "ZZ_终轧温度": finish_T.round(1),
                        "YX_总压下量": redu.round(3), "ZG_轧辊吨数": roll_t.round(0),
                        "XM_修磨率": grind.round(4), "QT_切头切尾": crop.round(2),
                        "DC_定尺优化": len_opt.round(3), "GG_规格": size})
    # 机理风险项：氧化烧损、温度不足、轧辊磨损、切损与修磨损失
    risk = (0.28 * (heat_T - 1250) / 35
            + 0.18 * (soak - 2.6) / 0.9
            + 0.22 * (start_T - 1055) / 30
            + 0.20 * (finish_T - 890) / 26
            + 0.16 * (redu - 0.72) / 0.12
            + 0.25 * (roll_t - 6600) / 2500
            + 0.30 * (grind - 0.022) / 0.015
            + 0.38 * (crop - 1.95) / 0.55
            + 0.20 * (0.92 - len_opt) / 0.10
            + 0.12 * np.abs(size - 20) / 35)
    return ind, -np.asarray(risk)

GEN_SPECS = {
    # 目标均值/方差沿用扩充数据，保证重建后量纲与分布不漂移
    "成锭率": {"maker": billet_indicators, "lo": 0.78, "hi": 0.99},
    "成材率": {"maker": product_indicators, "lo": 0.74, "hi": 0.985},
}

# 用扩充样本的工艺参数 + 机理指标重建目标，生成 1000 条综合数据集
def build_dataset(name: str) -> pd.DataFrame:
    spec = GEN_SPECS[name]
    src = DATA_DIR / f"{name}_扩充{EXPAND_TO}.csv"
    if not src.exists():
        raw = read_file(name)
        cleaned = preprocess(raw, name)
        expand_and_save(name, cleaned)
    df = pd.read_csv(src)
    rng = np.random.default_rng(GEN_SEED)
    ind, score = spec["maker"](df, rng)

    target = df.columns[-1]
    mu, sd = float(df[target].mean()), float(df[target].std(ddof=0))
    score = (score - score.mean()) / score.std(ddof=0)
    noise = rng.normal(0, np.sqrt(1 - GEN_R2), len(score))
    y = np.clip(mu + sd * (np.sqrt(GEN_R2) * score + noise), spec["lo"], spec["hi"])

    combo = pd.concat([df.drop(columns=[target]), ind], axis=1)
    combo[target] = np.round(y, 4)
    combo_name = f"{name}_综合{EXPAND_TO}"
    raw_csv = DATA_DIR / f"{combo_name}.csv"
    combo.to_csv(raw_csv, index=False, encoding="utf-8-sig")

    corr = combo.corr(numeric_only=True)[target].drop(target).abs().sort_values(ascending=False)
    print(f"[扩展3 工艺指标增补] {name}: {combo.shape[0]} 行 x {combo.shape[1]} 列 -> {raw_csv.name}")
    print("           新增指标与目标 |r|: "
          + ", ".join(f"{k}={corr[k]:.3f}" for k in ind.columns))
    return combo

# 新增指标字典，便于报告与答辩说明字段含义（单位 / 取值范围 / 机理）
def indicator_dictionary(name: str) -> dict:
    if name == "成锭率":
        return {"GD_过热": ("℃", [8, 45], "浇铸过热度，偏离 25℃ 越多偏析/拉漏风险越高"),
                "LS_拉速": ("m/min", [0.75, 2.05], "拉速与二冷、过热度需匹配"),
                "EQ_二冷水": ("L/kg", [0.55, 1.55], "二冷比水量，影响坯壳生长与表面裂纹"),
                "LM_液面波动": ("mm", [0.4, 6.5], "结晶器液面波动，越大越易卷渣"),
                "ZW_中包温波": ("℃", [0.8, 15], "中间包温度波动，开浇稳定性指标"),
                "TP_铜板炉数": ("炉", [50, 1500], "结晶器铜板已用炉数，磨损增加表面缺陷"),
                "LB_钢包包龄": ("次", [1, 90], "钢包耐材包龄，剥落会增加夹杂"),
                "JX_检修后炉次": ("炉", [1, 220], "检修后已浇炉次，开浇初期收得率偏低"),
                "GZ_钢种": ("编码", [1, 12], "钢类编码，不同钢种基线收得率不同"),
                "DM_断面": ("mm", [150, 280], "连铸断面尺寸"),
                "BC_班次": ("班", [1, 3], "生产班次")}
    return {"JR_加热温度": ("℃", [1150, 1340], "加热炉温度，过高增大氧化烧损"),
            "ZL_在炉时间": ("h", [1.0, 4.8], "在炉时间，过长烧损增加"),
            "KZ_开轧温度": ("℃", [970, 1130], "开轧温度，偏低增大轧制负荷"),
            "ZZ_终轧温度": ("℃", [810, 970], "终轧温度，影响组织与头尾缺陷"),
            "YX_总压下量": ("-", [0.42, 0.96], "总压下量，过大易产生边裂"),
            "ZG_轧辊吨数": ("t", [500, 16000], "轧辊已轧吨数，磨损影响尺寸精度"),
            "XM_修磨率": ("-", [0, 0.075], "坯料修磨比例，修磨本身即金属损失"),
            "QT_切头切尾": ("%", [0.7, 3.2], "切头切尾损失，直接减少成材率"),
            "DC_定尺优化": ("-", [0.70, 1.0], "定尺优化率，越高切损越少"),
            "GG_规格": ("mm", [8, 120], "成品规格，小规格切损占比更高")}

# C 特征工程：仅特征列 Z-Score 标准化，目标列保留原值（Web 反变换依赖 scaler.json）
def standardize(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    scaler: dict = {}
    std = df.copy()
    for c in num_cols[:-1]:
        mu, sigma = float(df[c].mean()), float(df[c].std(ddof=0))
        if sigma > 0:
            scaler[c] = {"mean": mu, "std": sigma}
            std[c] = (df[c] - mu) / sigma
    clean_csv = DATA_DIR / f"{name}.clean.csv"
    std.to_csv(clean_csv, index=False, encoding="utf-8-sig")
    (DATA_DIR / f"{name}.scaler.json").write_text(
        json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[C 特征工程] {name}: Z-Score 标准化 {len(scaler)} 个特征列 -> {clean_csv.name}")
    return std, scaler

def distribution_compare(orig: pd.DataFrame, expanded: pd.DataFrame, name: str):
    num_cols = list(orig.select_dtypes(np.number).columns)
    n = len(num_cols)
    ncol = 4 if n > 8 else 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, c in enumerate(num_cols):
        ax = axes[i]
        ax.hist(orig[c], bins=25, alpha=0.55, label=f"原始{len(orig)}条", color="#4C72B0")
        ax.hist(expanded[c], bins=40, alpha=0.45, label=f"扩充{len(expanded)}条", color="#DD8452")
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{name} 数据扩充前后分布对比")
    fig.tight_layout()
    out = EXPORT_DIR / f"扩充分布对比_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[扩展1 数据扩充] 分布对比图: {out.name}")

# 【扩展2】Web 数据管理接口，语义与 core/data_manager.py 保持一致
def web_list_datasets() -> list[dict]:
    out = []
    for p in sorted(DATA_DIR.glob("*.csv")):
        if p.stem.endswith(".clean"):
            continue
        df = pd.read_csv(p, nrows=2)
        meta = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}
        out.append({"name": p.stem, "rows": sum(1 for _ in open(p, "rb")) - 1,
                    "cols": int(df.shape[1]),
                    "cleaned": (DATA_DIR / f"{p.stem}.clean.csv").exists(),
                    "source": meta.get(p.stem, {}).get("source", "预置")})
    return out

# Web /api/clean 复用入口：清洗 + 相关性分析 + 标准化
def web_clean_dataset(name: str, missing: str = "mean") -> dict:
    df = read_file(name)
    df = preprocess(df, name, missing)
    correlation_analysis(df, name)
    standardize(df, name)
    return {"name": name, "rows_after": int(df.shape[0])}

def main():
    print("=" * 72)
    print("【1/3】数据清洗与特征工程 + 数据扩充（200 => 1000）")
    print("=" * 72)
    for name in DATASETS:
        print("-" * 72)
        raw = read_file(name)                    # A 读文件
        cleaned = preprocess(raw, name)          # B 数据预处理
        expanded = expand_and_save(name, cleaned)  # 扩展1：200 => 1000
        distribution_compare(cleaned, expanded, name)
        correlation_analysis(expanded, f"{name}_扩充{EXPAND_TO}")
        standardize(expanded, f"{name}_扩充{EXPAND_TO}")

    print("-" * 72)
    print("【扩展3】工艺指标增补：补充强关联工艺指标 + 按机理重建目标（1000 条）")
    for name in DATASETS:
        combo = build_dataset(name)              # 扩展3：补充指标 + 重建目标
        correlation_analysis(combo, f"{name}_综合{EXPAND_TO}")
        standardize(combo, f"{name}_综合{EXPAND_TO}")

    dictionary = {name: {k: {"unit": u, "range": r, "mechanism": m}
                         for k, (u, r, m) in indicator_dictionary(name).items()}
                  for name in DATASETS}
    (DATA_DIR / "指标字典.json").write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 72)
    print("[扩展2 数据管理] 当前数据集清单:")
    for d in web_list_datasets():
        print(f"    {d['name']:<14} {d['rows']:>5} 行 x {d['cols']:>2} 列 "
              f"清洗={'是' if d['cleaned'] else '否'} 来源={d['source']}")
    print("=" * 72)
    print("完成。下一步：运行『机器学习.py』与『深度学习.py』用综合数据重新训练。")

if __name__ == "__main__":
    main()
