from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
META_FILE = DATA_DIR / "_meta.json"
DATA_DIR.mkdir(exist_ok=True)

# 数据集来源等元信息持久化在 data/_meta.json
def _load_meta() -> dict:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    return {}

def _save_meta(meta: dict) -> None:
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _clean_path(name: str) -> Path:
    return DATA_DIR / f"{name}.clean.csv"

def list_datasets() -> list[dict]:
    meta = _load_meta()
    out = []
    # 跳过 .clean 派生文件，只列出原始数据集
    for p in sorted(DATA_DIR.glob("*.csv")):
        if p.stem.endswith(".clean"):
            continue
        try:
            df = pd.read_csv(p, nrows=5)
            full_rows = sum(1 for _ in open(p, "rb")) - 1
        except Exception:
            continue
        out.append({
            "name": p.stem,
            "rows": full_rows,
            "cols": int(df.shape[1]),
            "cleaned": _clean_path(p.stem).exists(),
            "source": meta.get(p.stem, {}).get("source", "预置"),
        })
    return out

def load_dataset(name: str, cleaned: bool = True) -> pd.DataFrame:
    if cleaned and _clean_path(name).exists():
        return pd.read_csv(_clean_path(name))
    return pd.read_csv(DATA_DIR / f"{name}.csv")

def save_upload(filename: str, content: bytes) -> dict:
    name = Path(filename).stem
    if not filename.lower().endswith(".csv"):
        raise ValueError("仅支持 CSV 格式文件")
    dest = DATA_DIR / f"{name}.csv"
    dest.write_bytes(content)
    df = pd.read_csv(dest)
    meta = _load_meta()
    meta[name] = {"source": "上传"}
    _save_meta(meta)
    return {"name": name, "rows": int(df.shape[0]), "cols": int(df.shape[1])}

def delete_dataset(name: str) -> None:
    for p in [DATA_DIR / f"{name}.csv", _clean_path(name)]:
        if p.exists():
            p.unlink()
    meta = _load_meta()
    meta.pop(name, None)
    _save_meta(meta)

def preview(name: str, page: int = 1, page_size: int = 10) -> dict:
    df = load_dataset(name, cleaned=False)
    total = int(df.shape[0])
    start = (page - 1) * page_size
    chunk = df.iloc[start:start + page_size]
    return {
        "columns": list(df.columns),
        "rows": json.loads(chunk.to_json(orient="values", force_ascii=False)),
        "total": total,
        "page": page,
        "pages": max(1, int(np.ceil(total / page_size))),
    }

def distribution(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include=[np.number])
    return {"columns": list(num.columns),
            "mean": [round(float(v), 4) for v in num.mean()],
            "std": [round(float(v), 4) for v in num.std(ddof=0)]}

def clean_dataset(name: str, missing_strategy: str,
                  remove_outliers: bool, standardize: bool) -> dict:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    before = distribution(df)

    # 1) 缺失值处理：均值 / 中位数 / 删除整行
    missing_before = int(df.isna().sum().sum())
    num_cols = list(df.select_dtypes(include=[np.number]).columns)

    if missing_strategy == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif missing_strategy == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif missing_strategy == "drop":
        df = df.dropna().reset_index(drop=True)
    else:
        raise ValueError(f"未知缺失值策略: {missing_strategy}")

    # 2) 异常值处理：3σ 原则，逐列剔除越界样本
    outliers_removed = 0
    if remove_outliers:
        mask = pd.Series(True, index=df.index)
        for c in num_cols:
            mu, sigma = df[c].mean(), df[c].std(ddof=0)
            if sigma > 0:
                mask &= (df[c] - mu).abs() <= 3 * sigma
        outliers_removed = int((~mask).sum())
        df = df[mask].reset_index(drop=True)

    # 3) 仅特征列做 Z-Score 标准化，目标列保留原值便于结果解读
    scaler = None
    if standardize:
        target = num_cols[-1]
        scaler = {}
        for c in num_cols[:-1]:
            mu, sigma = df[c].mean(), df[c].std(ddof=0)
            if sigma > 0:
                scaler[c] = {"mean": float(mu), "std": float(sigma)}
                df[c] = (df[c] - mu) / sigma

    df.to_csv(_clean_path(name), index=False, encoding="utf-8-sig")

    scaler_file = DATA_DIR / f"{name}.scaler.json"
    if scaler:
        scaler_file.write_text(json.dumps(scaler, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    elif scaler_file.exists():
        scaler_file.unlink()
    return {
        "missing_before": missing_before,
        "outliers_removed": outliers_removed,
        "rows_after": int(df.shape[0]),
        "before": before,
        "after": distribution(df),
    }

# 首页四个统计卡片：数据集数、样本数、模型数、优化方案数
def stats_summary() -> dict:
    datasets = list_datasets()
    total_rows = sum(d["rows"] for d in datasets)
    models_dir = BASE_DIR / "models"
    model_count = len(list(models_dir.glob("*.meta.json"))) if models_dir.exists() else 0
    from . import optimizer
    return {
        "datasets": len(datasets),
        "samples": total_rows,
        "models": model_count,
        "optimizations": optimizer.count_optimizations(),
    }
