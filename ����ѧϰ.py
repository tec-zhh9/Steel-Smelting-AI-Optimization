from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EXPORT_DIR = BASE_DIR / "exports"
MODELS_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

SEED = 42
ML_ALGORITHMS = {"random_forest": "随机森林", "xgboost": "XGBoost", "svr": "支持向量机"}

# 优先使用综合工艺数据集，其次回退扩增数据集，最后回退原始数据集
def discover_datasets() -> list[str]:
    for pattern in ("*_综合1000.clean.csv", "*_扩充1000.clean.csv"):
        out = [p.name.removesuffix(".clean.csv") for p in sorted(DATA_DIR.glob(pattern))]
        if out:
            return out
    return ["成锭率", "成材率"]

def prepare(dataset: str, test_size: float = 0.2):
    df = pd.read_csv(DATA_DIR / f"{dataset}.clean.csv")
    df = df.select_dtypes(include=[np.number]).dropna()
    features, target = list(df.columns[:-1]), df.columns[-1]
    X, y = df[features].to_numpy(float), df[target].to_numpy(float)

    # 目标分位数分箱 + 分层抽样，保证训练/测试目标分布一致（收得率方差极小）
    strat = pd.qcut(y, 4, labels=False, duplicates="drop")
    tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                         random_state=SEED).split(X, strat))
    scaler = StandardScaler().fit(X[tr])
    return (scaler.transform(X[tr]), scaler.transform(X[te]),
            y[tr], y[te], features, target, scaler)

def metrics(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    acc = float(np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8) <= 0.05))  # 相对误差≤5% 视为准确
    return {"mse": round(mse, 6), "mae": round(mae, 6), "r2": round(r2, 4),
            "accuracy": round(acc, 4)}

def build_model(algo: str, params: dict | None = None):
    p = params or {}
    if algo == "random_forest":
        return RandomForestRegressor(random_state=SEED, n_jobs=-1, **p)
    if algo == "xgboost":
        base = dict(learning_rate=0.1, max_depth=5, subsample=0.9,
                    colsample_bytree=0.9, random_state=SEED, n_jobs=-1)
        base.update(p)
        return XGBRegressor(**base)
    if algo == "svr":
        return SVR(**p)
    raise ValueError(f"未知算法: {algo}")

# E 调参：各算法的网格搜索空间
PARAM_GRIDS = {
    "random_forest": {"n_estimators": [200, 400], "max_depth": [None, 12]},
    "xgboost": {"n_estimators": [300, 600], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    "svr": {"C": [1, 10, 100], "epsilon": [0.01, 0.05], "gamma": ["scale", 0.05]},
}

# 同一划分上对比调参前后：默认模型 vs GridSearchCV 最优模型
def evaluate(algo: str, dataset: str):
    X_tr, X_te, y_tr, y_te, features, target, scaler = prepare(dataset)

    base = build_model(algo).fit(X_tr, y_tr)
    m_base = metrics(y_te, base.predict(X_te))

    grid = GridSearchCV(build_model(algo), PARAM_GRIDS[algo], cv=5,
                        scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    m_best = metrics(y_te, best.predict(X_te))
    return m_base, m_best, grid.best_params_, best, features, target, scaler

# 【扩展3】按 core/trainer.py 的格式保存 pkl + meta.json，Web 端可直接加载与优化
def save_web_model(algo: str, dataset: str, model, scaler, features, target, m: dict) -> str:
    model_id = f"{algo}_{uuid.uuid4().hex[:8]}"
    joblib.dump({"model": model, "scaler": scaler}, MODELS_DIR / f"{model_id}.pkl")
    meta = {"id": model_id, "algo": algo, "algo_label": ML_ALGORITHMS[algo], "category": "ml",
            "dataset": dataset, "features": features, "target": target, "metrics": m,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    (MODELS_DIR / f"{model_id}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_id

# 与 core/optimizer._load_model 一致：输入原始工艺值，内部完成标准化后预测
def load_web_model(model_id: str):
    meta = json.loads((MODELS_DIR / f"{model_id}.meta.json").read_text(encoding="utf-8"))
    bundle = joblib.load(MODELS_DIR / f"{model_id}.pkl")
    c_scaler = load_clean_scaler(meta["dataset"])
    # 收得率物理上限：预测不得超过训练样本最大值，避免出现 >100% 的非法结果
    cap = float(pd.read_csv(DATA_DIR / f"{meta['dataset']}.clean.csv")[meta["target"]].max())

    def predict_orig(X_raw: np.ndarray) -> np.ndarray:
        Xs = X_raw
        if c_scaler:
            Xs = np.array([[(x - c_scaler[f]["mean"]) / c_scaler[f]["std"]
                            if f in c_scaler else x for x, f in zip(row, meta["features"])]
                           for row in np.atleast_2d(X_raw)])
        return np.clip(bundle["model"].predict(bundle["scaler"].transform(Xs)), 0.0, cap)

    return meta, predict_orig

def load_clean_scaler(dataset: str) -> dict | None:
    f = DATA_DIR / f"{dataset}.scaler.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

def feature_bounds(dataset: str) -> tuple[list[str], dict[str, dict]]:
    df = pd.read_csv(DATA_DIR / f"{dataset}.clean.csv").select_dtypes(np.number).dropna()
    features = list(df.columns[:-1])
    scaler = load_clean_scaler(dataset)
    bounds = {}
    for c in features:
        lo, hi, mu = float(df[c].min()), float(df[c].max()), float(df[c].mean())
        if scaler and c in scaler:
            m, s = scaler[c]["mean"], scaler[c]["std"]
            lo, hi, mu = lo * s + m, hi * s + m, mu * s + m
        bounds[c] = {"min": lo, "max": hi, "mean": mu}
    return features, bounds

# 【扩展2】实数编码遗传算法：锦标赛选择 + 算术交叉 + 高斯变异 + 精英保留
def genetic_algorithm(predict, lo: np.ndarray, hi: np.ndarray,
                      pop: int = 60, gens: int = 60, seed: int = SEED) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    d = len(lo)
    popx = rng.uniform(lo, hi, size=(pop, d))
    fit = predict(popx)
    span = hi - lo
    for _ in range(gens):
        order = np.argsort(fit)[::-1]
        new = [popx[i].copy() for i in order[:2]]          # 精英保留
        pool = popx[order[:pop // 2]]                      # 锦标赛候选池（前 50%）
        while len(new) < pop:
            i, j = rng.integers(0, len(pool), 2)
            alpha = rng.uniform(0.3, 0.7)                  # 算术交叉
            child = alpha * pool[i] + (1 - alpha) * pool[j]
            child = child + rng.normal(0, 0.08 * span)     # 高斯变异
            new.append(np.clip(child, lo, hi))
        popx = np.array(new)
        fit = predict(popx)
    best = int(np.argmax(fit))
    return popx[best], float(fit[best])

# 【扩展2】模拟退火：按 Metropolis 准则以一定概率接受更差解，跳出局部最优
def simulated_annealing(predict, lo: np.ndarray, hi: np.ndarray,
                        iters: int = 3000, seed: int = SEED) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi)
    fx = float(predict(x.reshape(1, -1))[0])
    best_x, best_f, T = x.copy(), fx, 1.0
    span = (hi - lo) * 0.06
    for _ in range(iters):
        cand = np.clip(x + rng.normal(0, span), lo, hi)
        fc = float(predict(cand.reshape(1, -1))[0])
        if fc > fx or rng.random() < np.exp((fc - fx) / max(T, 1e-9)):  # 温度越低越难接受差解
            x, fx = cand, fc
        if fx > best_f:
            best_x, best_f = x.copy(), fx
        T *= 0.998
    return best_x, best_f

# 【扩展2】连续域蚁群：每维离散为 bins 个区间，信息素引导选桶 + 桶内随机扰动
def ant_colony(predict, lo: np.ndarray, hi: np.ndarray,
               n_ants: int = 25, iters: int = 45, bins: int = 20,
               seed: int = SEED) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    d = len(lo)
    tau = np.ones((d, bins))
    edges = [np.linspace(lo[j], hi[j], bins + 1) for j in range(d)]
    best_x, best_f = None, -np.inf
    for _ in range(iters):
        p = tau / tau.sum(axis=1, keepdims=True)
        X = np.zeros((n_ants, d))
        for j in range(d):
            idx = rng.choice(bins, size=n_ants, p=p[j])
            X[:, j] = [rng.uniform(edges[j][k], edges[j][k + 1]) for k in idx]
        f = predict(X)
        k = int(np.argmax(f))
        if f[k] > best_f:
            best_x, best_f = X[k].copy(), float(f[k])
        tau *= 0.7                                        # 信息素挥发
        for j in range(d):
            b = np.searchsorted(edges[j], best_x[j]) - 1
            tau[j, np.clip(b, 0, bins - 1)] += 1.0         # 最优蚁路径强化
        tau += 1e-3
    return best_x, best_f

# 三种智能算法在同一搜索范围内寻优，取预测值最高的方案作为推荐
def optimize_by_metaheuristics(model_id: str) -> dict:
    meta, predict = load_web_model(model_id)
    features, bounds = feature_bounds(meta["dataset"])
    lo = np.array([bounds[f]["min"] for f in features])
    hi = np.array([bounds[f]["max"] for f in features])
    current = np.array([bounds[f]["mean"] for f in features])
    cur_pred = float(predict(current.reshape(1, -1))[0])

    results = {}
    for name, fn in [("遗传算法", genetic_algorithm), ("模拟退火", simulated_annealing),
                     ("蚁群算法", ant_colony)]:
        x, f = fn(predict, lo, hi)
        results[name] = (x, f)
        print(f"    {name:<6} 最优预测 {meta['target']} = {f:.4f} "
              f"(较均值参数 {(f - cur_pred) / max(abs(cur_pred), 1e-8) * 100:+.2f}%)")

    best_name = max(results, key=lambda k: results[k][1])
    x_orig, opt_pred = results[best_name]

    rows, before, after = [], [], []
    for j, f in enumerate(features):
        c, r = float(current[j]), float(x_orig[j])
        tol = max(abs(c), 1e-6) * 1e-3
        direction = "上调" if r > c + tol else ("下调" if r < c - tol else "保持")
        rows.append({"param": f, "current": round(c, 4), "recommended": round(r, 4),
                     "direction": direction})
        before.append(round((c - lo[j]) / max(hi[j] - lo[j], 1e-8), 4))
        after.append(round((r - lo[j]) / max(hi[j] - lo[j], 1e-8), 4))

    result = {"model_id": model_id, "model_label": meta["algo_label"],
              "dataset": meta["dataset"], "target": meta["target"],
              "current_pred": round(cur_pred, 4), "optimized_pred": round(opt_pred, 4),
              "improvement": round((opt_pred - cur_pred) / max(abs(cur_pred), 1e-8) * 100, 2),
              "params": rows, "algorithm": best_name,
              "chart": {"names": features, "before": before, "after": after},
              "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    with open(EXPORT_DIR / "optimizations.jsonl", "a", encoding="utf-8") as fp:
        fp.write(json.dumps(result, ensure_ascii=False) + "\n")
    (EXPORT_DIR / "last_optimization.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame([{"算法": k, "最优预测": round(v[1], 4),
                   "较均值提升%": round((v[1] - cur_pred) / abs(cur_pred) * 100, 2)}
                  for k, v in results.items()]).to_csv(
        EXPORT_DIR / f"智能寻优对比_{meta['dataset']}.csv", index=False, encoding="utf-8-sig")
    return result

# 【扩展1】指标归一化加权：R2 0.35 + 准确率 0.30 + MSE 0.20 + MAE 0.15
def comprehensive_evaluation(rows: list[dict], dataset: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c, w, higher in [("r2", 0.35, True), ("accuracy", 0.30, True),
                         ("mse", 0.20, False), ("mae", 0.15, False)]:
        v = df[c].astype(float)
        rngv = v.max() - v.min()
        norm = (v - v.min()) / rngv if rngv > 0 else v * 0 + 0.5
        df[f"norm_{c}"] = norm if higher else 1 - norm
    df["综合得分"] = (df.norm_r2 * 0.35 + df.norm_accuracy * 0.30
                     + df.norm_mse * 0.20 + df.norm_mae * 0.15).round(4)
    df = df.sort_values("综合得分", ascending=False).reset_index(drop=True)
    df.insert(0, "排名", df.index + 1)

    out_csv = EXPORT_DIR / f"多算法综合评价_{dataset}.csv"
    df[["排名", "算法", "mse", "mae", "r2", "accuracy", "综合得分"]].to_csv(
        out_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["算法"], df["综合得分"], color=["#C44E52", "#55A868", "#4C72B0"][:len(df)])
    for i, s in enumerate(df["综合得分"]):
        ax.text(i, s, f"{s:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("综合得分")
    ax.set_title(f"{dataset} 机器学习多算法综合评价（R2 0.35 + 准确率 0.30 + MSE 0.20 + MAE 0.15）")
    fig.tight_layout()
    fig.savefig(EXPORT_DIR / f"多算法综合评价_{dataset}.png", dpi=150)
    plt.close(fig)
    return df

def main():
    print("=" * 72)
    print("【2/3】机器学习建模（随机森林 / XGBoost / 支持向量机）+ 评估调参 + 扩展")
    print("=" * 72)
    for dataset in discover_datasets():
        print("-" * 72)
        print(f"数据集: {dataset}")
        summary = []
        best_id, best_pack = None, None
        # 三种算法分别训练、网格调参，并保存为 Web 兼容模型
        for algo in ML_ALGORITHMS:
            m_base, m_best, params, model, features, target, scaler = evaluate(algo, dataset)
            model_id = save_web_model(algo, dataset, model, scaler, features, target, m_best)
            print(f"[D 建模] {ML_ALGORITHMS[algo]:<5} 调参前 R2={m_base['r2']:.4f} "
                  f"MSE={m_base['mse']:.6f} | 调参后 R2={m_best['r2']:.4f} "
                  f"MSE={m_best['mse']:.6f} 准确率={m_best['accuracy']:.2%}")
            print(f"        最优参数 {params} -> 模型 {model_id}")
            summary.append({"算法": ML_ALGORITHMS[algo], **m_best})
            if best_id is None or m_best["r2"] > best_pack["r2"]:
                best_id, best_pack = model_id, m_best

        print("[扩展1 多算法综合评价]")
        rank = comprehensive_evaluation(summary, dataset)
        print(rank[["排名", "算法", "r2", "accuracy", "综合得分"]].to_string(index=False))

        print(f"[扩展2 遗传/退火/蚁群寻优] 目标模型: {best_id}")
        result = optimize_by_metaheuristics(best_id)
        print(f"    推荐 {result['target']}: {result['current_pred']} -> "
              f"{result['optimized_pred']} ({result['improvement']:+.2f}%, {result['algorithm']})")
    print("=" * 72)
    print("完成。模型已保存至 models/，Web 系统参数优化页可直接使用。")

if __name__ == "__main__":
    main()
