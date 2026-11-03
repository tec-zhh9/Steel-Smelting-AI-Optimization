from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data_manager

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)
OPT_LOG = EXPORT_DIR / "optimizations.jsonl"

def count_optimizations() -> int:
    if not OPT_LOG.exists():
        return 0
    return sum(1 for _ in open(OPT_LOG, "rb"))

# 按 meta 区分机器学习(pkl)与深度学习(pt)两种加载方式，统一返回“原始量纲 -> 预测值”函数
def _load_model(model_id: str):
    meta = json.loads((MODELS_DIR / f"{model_id}.meta.json").read_text(encoding="utf-8"))
    if meta["category"] == "ml":
        bundle = joblib.load(MODELS_DIR / f"{model_id}.pkl")

        def predict(X_raw: np.ndarray) -> np.ndarray:
            return bundle["model"].predict(bundle["scaler"].transform(X_raw))
    else:
        import torch
        from .trainer import _build_net

        net = _build_net(meta["algo"], len(meta["features"]))
        net.load_state_dict(torch.load(MODELS_DIR / f"{model_id}.pt",
                                       map_location="cpu", weights_only=True))
        net.eval()
        mean = np.array(meta["scaler_mean"])
        scale = np.array(meta["scaler_scale"])
        y_mean = float(meta.get("y_mean", 0.0))
        y_std = float(meta.get("y_std", 1.0))

        def predict(X_raw: np.ndarray) -> np.ndarray:
            Xs = (X_raw - mean) / scale
            with torch.no_grad():
                out = net(torch.tensor(Xs, dtype=torch.float32)).numpy()
            return out * y_std + y_mean

    return meta, predict

def _load_scaler(dataset: str) -> dict | None:
    f = data_manager.DATA_DIR / f"{dataset}.scaler.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return None

# 取清洗后数据各工艺参数的最小/最大值作为寻优边界，并按 scaler.json 反变换回原始量纲
def dataset_feature_bounds(dataset: str) -> dict:
    df = data_manager.load_dataset(dataset, cleaned=True)
    df = df.select_dtypes(include=[np.number]).dropna()
    features = list(df.columns[:-1])
    scaler = _load_scaler(dataset)
    bounds = {}
    for c in features:
        lo, hi, mu = float(df[c].min()), float(df[c].max()), float(df[c].mean())
        if scaler and c in scaler:
            m, s = scaler[c]["mean"], scaler[c]["std"]
            lo, hi, mu = lo * s + m, hi * s + m, mu * s + m
        bounds[c] = {"min": round(lo, 4), "max": round(hi, 4), "mean": round(mu, 4)}
    return {"features": features, "target": df.columns[-1], "bounds": bounds}

# 多起点随机搜索 + 坐标细化：全局撒点找大致最优，再在当前参数附近局部撒点并做 3 轮坐标搜索
def optimize(model_id: str, current: dict[str, float],
             n_random: int = 4000, seed: int = 42) -> dict:
    meta, predict = _load_model(model_id)
    features = meta["features"]
    dataset = meta["dataset"]
    info = dataset_feature_bounds(dataset)
    bounds = info["bounds"]
    scaler = _load_scaler(dataset)

    def to_model_space(x_orig: np.ndarray) -> np.ndarray:
        if not scaler:
            return x_orig
        return np.array([(x_orig[j] - scaler[f]["mean"]) / scaler[f]["std"]
                         if f in scaler else x_orig[j]
                         for j, f in enumerate(features)])

    def to_orig_space(x_std: np.ndarray) -> np.ndarray:
        if not scaler:
            return x_std
        return np.array([x_std[j] * scaler[f]["std"] + scaler[f]["mean"]
                         if f in scaler else x_std[j]
                         for j, f in enumerate(features)])

    def predict_orig(X_orig: np.ndarray) -> np.ndarray:
        return predict(np.apply_along_axis(to_model_space, 1, X_orig))

    cur_orig = np.array([float(current[f]) for f in features], dtype=np.float64)
    cur_std = to_model_space(cur_orig)

    df = data_manager.load_dataset(dataset, cleaned=True)
    df = df.select_dtypes(include=[np.number]).dropna()
    lo = np.array([float(df[f].min()) for f in features])
    hi = np.array([float(df[f].max()) for f in features])
    # 收得率物理上限：预测不得超过训练样本中的最大值，避免推荐出 >100% 的非法工艺
    cap = float(df[meta["target"]].max()) if meta["target"] in df.columns else 1.0

    lo_orig = to_orig_space(lo)
    hi_orig = to_orig_space(hi)

    cur_pred = float(np.clip(predict(cur_std.reshape(1, -1))[0], 0.0, cap))

    rng = np.random.default_rng(seed)
    # 全范围随机采样 + 当前参数 ±20% 范围的局部采样
    global_samples = rng.uniform(lo, hi, size=(n_random, len(features)))
    span = (hi - lo) * 0.2
    local_lo = np.maximum(lo, cur_std - span)
    local_hi = np.minimum(hi, cur_std + span)
    local_samples = rng.uniform(local_lo, local_hi, size=(n_random // 2, len(features)))
    samples = np.vstack([global_samples, local_samples])
    preds = np.clip(predict(samples), 0.0, cap)
    best = samples[int(np.argmax(preds))]

    # 坐标细化：每次只动一个参数，步长逐轮减半
    step = (hi - lo) * 0.05
    x = best.copy()
    for _ in range(3):
        for j in range(len(features)):
            candidates = []
            for d in (-1, 0, 1):
                xc = x.copy()
                xc[j] = np.clip(xc[j] + d * step[j], lo[j], hi[j])
                candidates.append(xc)
            scores = np.clip(predict(np.array(candidates)), 0.0, cap)
            x = candidates[int(np.argmax(scores))]
        step *= 0.5

    opt_pred = float(np.clip(predict(x.reshape(1, -1))[0], 0.0, cap))
    x_orig = to_orig_space(x)

    rows = []
    for f, c, r in zip(features, cur_orig, x_orig):
        tol = max(abs(c), 1e-6) * 1e-3
        if r > c + tol:
            direction = "上调"
        elif r < c - tol:
            direction = "下调"
        else:
            direction = "保持"
        rows.append({"param": f, "current": round(float(c), 4),
                     "recommended": round(float(r), 4), "direction": direction})

    result = {
        "model_id": model_id,
        "model_label": meta["algo_label"],
        "dataset": dataset,
        "target": meta["target"],
        "current_pred": round(cur_pred, 4),
        "optimized_pred": round(opt_pred, 4),
        "improvement": round((opt_pred - cur_pred) / max(abs(cur_pred), 1e-8) * 100, 2),
        "params": rows,

        "chart": {
            "names": features,
            "before": [round(float((c - l) / max(h - l, 1e-8)), 4)
                       for c, l, h in zip(cur_orig, lo_orig, hi_orig)],
            "after": [round(float((r - l) / max(h - l, 1e-8)), 4)
                      for r, l, h in zip(x_orig, lo_orig, hi_orig)],
        },
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OPT_LOG, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(result, ensure_ascii=False) + "\n")

    (EXPORT_DIR / "last_optimization.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

# 将最近一次优化方案导出为 CSV（UTF-8 BOM，Excel 可直接打开）
def export_last() -> Path:
    last_file = EXPORT_DIR / "last_optimization.json"
    if not last_file.exists():
        raise FileNotFoundError("尚无优化方案可导出")
    result = json.loads(last_file.read_text(encoding="utf-8"))
    df = pd.DataFrame(result["params"])
    df.columns = ["参数名称", "当前值", "推荐值", "调整方向"]
    df.insert(0, "数据集", result["dataset"])
    df.insert(1, "模型", result["model_label"])
    summary = pd.DataFrame([
        {"参数名称": f"优化前预测{result['target']}", "当前值": result["current_pred"],
         "推荐值": result["optimized_pred"], "调整方向": f"提升 {result['improvement']}%"},
    ])
    out = pd.concat([df, summary], ignore_index=True)
    dest = EXPORT_DIR / f"优化方案_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out.to_csv(dest, index=False, encoding="utf-8-sig")
    return dest
