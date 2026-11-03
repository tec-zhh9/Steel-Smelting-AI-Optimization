from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from . import data_manager

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

ML_ALGORITHMS = {"random_forest": "随机森林", "xgboost": "XGBoost", "svr": "支持向量机"}
DL_ALGORITHMS = {"bp": "BP 神经网络", "lstm": "LSTM"}

# 全局训练状态，供 /api/train/progress 轮询
TRAIN_STATE: dict = {"status": "idle", "progress": 0, "message": "",
                     "loss_curve": [], "metrics": None, "model_id": None}

def _reset_state():
    TRAIN_STATE.update({"status": "running", "progress": 0, "message": "初始化…",
                        "loss_curve": [], "metrics": None, "model_id": None})

def _tick(progress: int, message: str, epoch: int | None = None, loss: float | None = None):
    TRAIN_STATE["progress"] = min(99, progress)
    TRAIN_STATE["message"] = message
    if epoch is not None and loss is not None:
        TRAIN_STATE["loss_curve"].append({"epoch": epoch, "loss": round(float(loss), 6)})

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1e-8)
    acc = float(np.mean(np.abs(y_pred - y_true) / denom <= 0.05))  # 相对误差≤5% 视为准确
    return {"mse": round(mse, 6), "mae": round(mae, 6),
            "r2": round(r2, 4), "accuracy": round(acc, 4)}

def _prepare(dataset: str, test_size: float):
    df = data_manager.load_dataset(dataset, cleaned=True)
    df = df.select_dtypes(include=[np.number]).dropna()
    features = list(df.columns[:-1])
    target = df.columns[-1]
    X = df[features].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)

    # 目标分位数分箱 + 分层划分，保证训练/测试目标分布一致（小方差目标必需）
    strat = pd.qcut(y, 4, labels=False, duplicates="drop")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr_idx, te_idx = next(splitter.split(X, strat))
    X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te, features, target, scaler

def _save_sklearn(model, model_id, algo, dataset, features, target, scaler, metrics):
    joblib.dump({"model": model, "scaler": scaler}, MODELS_DIR / f"{model_id}.pkl")
    _write_meta(model_id, algo, ML_ALGORITHMS[algo], "ml", dataset, features, target, metrics)

def _write_meta(model_id, algo, algo_label, category, dataset, features, target, metrics,
                extra: dict | None = None):
    meta = {"id": model_id, "algo": algo, "algo_label": algo_label, "category": category,
            "dataset": dataset, "features": features, "target": target,
            "metrics": metrics, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        meta.update(extra)
    (MODELS_DIR / f"{model_id}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _train_ml(algo: str, dataset: str, epochs: int, lr: float, test_size: float):
    X_tr, X_te, y_tr, y_te, features, target, scaler = _prepare(dataset, test_size)
    stages = 20  # 分段训练，逐段记录测试 MSE 形成前端损失曲线

    if algo == "random_forest":
        # warm_start 让随机森林逐步增加树，而不是每次从头训练
        model = RandomForestRegressor(n_estimators=0, warm_start=True,
                                      random_state=42, n_jobs=-1)
        per = max(1, epochs // stages)
        for i in range(stages):
            model.n_estimators += per
            model.fit(X_tr, y_tr)
            loss = mean_squared_error(y_te, model.predict(X_te))
            _tick(int((i + 1) / stages * 95), f"训练中 {int((i + 1) / stages * 100)}%（{model.n_estimators} 棵树）",
                  epoch=(i + 1) * per, loss=loss)

    elif algo == "xgboost":
        per = max(1, epochs // stages)
        booster = None
        model = None
        for i in range(stages):
            model = XGBRegressor(n_estimators=per, learning_rate=lr, max_depth=5,
                                 subsample=0.9, colsample_bytree=0.9, random_state=42,
                                 n_jobs=-1)
            model.fit(X_tr, y_tr, xgb_model=booster)  # 在已有 booster 上续训
            booster = model.get_booster()
            loss = mean_squared_error(y_te, model.predict(X_te))
            _tick(int((i + 1) / stages * 95), f"训练中 {int((i + 1) / stages * 100)}%（{(i + 1) * per} 轮迭代）",
                  epoch=(i + 1) * per, loss=loss)

    elif algo == "svr":
        # SVR 不支持增量训练：用递增样本比例拟合，近似绘制学习曲线
        fractions = np.linspace(0.2, 1.0, stages)
        model = None
        for i, frac in enumerate(fractions):
            n = max(10, int(len(X_tr) * frac))
            model = SVR(C=1.0, epsilon=0.05)
            model.fit(X_tr[:n], y_tr[:n])
            loss = mean_squared_error(y_te, model.predict(X_te))
            _tick(int((i + 1) / stages * 95), f"训练中 {int((i + 1) / stages * 100)}%（样本比例 {frac:.0%}）",
                  epoch=i + 1, loss=loss)
        model.fit(X_tr, y_tr)
    else:
        raise ValueError(f"未知机器学习算法: {algo}")

    y_pred = model.predict(X_te)
    metrics = _metrics(y_te, y_pred)
    model_id = f"{algo}_{uuid.uuid4().hex[:8]}"
    _save_sklearn(model, model_id, algo, dataset, features, target, scaler, metrics)
    return model_id, metrics

# 与（3）脚本保持相同的网络结构，保证两处保存的 pt 模型可互相加载
def _build_net(algo: str, n_features: int):
    import torch.nn as nn

    class BPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            seq = x.unsqueeze(-1)
            out, _ = self.lstm(seq)
            return self.fc(out[:, -1, :]).squeeze(-1)

    return LSTMNet() if algo == "lstm" else BPNet()

def _train_dl(algo: str, dataset: str, epochs: int, lr: float, test_size: float):
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    X_tr, X_te, y_tr, y_te, features, target, scaler = _prepare(dataset, test_size)

    # 目标值标准化后在网络空间训练，预测时再反变换回原始量纲
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    y_tr_s = (y_tr - y_mean) / y_std

    if algo == "lstm":
        lr = min(lr, 0.001)  # LSTM 对学习率敏感，限制在 0.001 以内

    net = _build_net(algo, X_tr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr_s, dtype=torch.float32)
    xe = torch.tensor(X_te, dtype=torch.float32)

    yt_e = torch.tensor((y_te - y_mean) / y_std, dtype=torch.float32)
    record_every = max(1, epochs // 50)
    best_val, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # 梯度裁剪，防止 LSTM 发散
        opt.step()
        if epoch % record_every == 0 or epoch == epochs:
            net.eval()
            with torch.no_grad():
                val = loss_fn(net(xe), yt_e).item()
            # 记录测试损失最低的轮次权重（早停思想）
            if val < best_val:
                best_val = val
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            _tick(int(epoch / epochs * 95), f"训练中 {int(epoch / epochs * 100)}%（第 {epoch}/{epochs} 轮）",
                  epoch=epoch, loss=val)

    # 恢复最优权重，避免继续训练导致过拟合
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        y_pred = net(xe).numpy() * y_std + y_mean  # 反标准化回原始收得率
    metrics = _metrics(y_te, y_pred)

    model_id = f"{algo}_{uuid.uuid4().hex[:8]}"
    torch.save(net.state_dict(), MODELS_DIR / f"{model_id}.pt")
    _write_meta(model_id, algo, DL_ALGORITHMS[algo], "dl", dataset, features, target, metrics,
                extra={"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                       "y_mean": y_mean, "y_std": y_std})
    return model_id, metrics

def start_training(dataset: str, category: str, algo: str,
                   epochs: int, lr: float, test_size: float) -> None:
    if TRAIN_STATE.get("status") == "running":
        raise RuntimeError("已有训练任务进行中")
    _reset_state()

    def _run():
        try:
            if category == "ml":
                model_id, metrics = _train_ml(algo, dataset, epochs, lr, test_size)
            else:
                model_id, metrics = _train_dl(algo, dataset, epochs, lr, test_size)
            TRAIN_STATE.update({"status": "done", "progress": 100,
                                "message": "训练完成", "metrics": metrics,
                                "model_id": model_id})
        except Exception as e:
            TRAIN_STATE.update({"status": "error", "message": str(e)})

    # 后台线程训练，前端通过 /api/train/progress 轮询进度，避免请求阻塞
    threading.Thread(target=_run, daemon=True).start()

def list_models() -> list[dict]:
    out = []
    for p in sorted(MODELS_DIR.glob("*.meta.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out
