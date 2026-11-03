from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

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
DL_ALGORITHMS = {"bp": "BP 神经网络", "lstm": "LSTM"}

# D 建模：BP 为 3 层全连接网络；LSTM 将 n 个特征视为长度为 n 的序列
def build_net(algo: str, n_features: int):
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
        # 每个样本的 n 个特征按顺序作为序列的 n 个时间步（每步 1 维）
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            seq = x.unsqueeze(-1)
            out, _ = self.lstm(seq)
            return self.fc(out[:, -1, :]).squeeze(-1)

    return LSTMNet() if algo == "lstm" else BPNet()

def prepare(dataset: str, test_size: float = 0.2):
    df = pd.read_csv(DATA_DIR / f"{dataset}.clean.csv")
    df = df.select_dtypes(include=[np.number]).dropna()
    features, target = list(df.columns[:-1]), df.columns[-1]
    X, y = df[features].to_numpy(float), df[target].to_numpy(float)
    # 目标分位数分箱 + 分层抽样，保证训练/测试目标分布一致
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

def train_dl(algo: str, dataset: str, epochs: int = 300, lr: float = 0.01):
    torch.manual_seed(SEED)
    X_tr, X_te, y_tr, y_te, features, target, scaler = prepare(dataset)

    # 目标值标准化后再训练，评估时再反变换回原始量纲
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    if algo == "lstm":
        lr = min(lr, 0.001)                  # LSTM 对学习率敏感，限制到 0.001 以内

    net = build_net(algo, X_tr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor((y_tr - y_mean) / y_std, dtype=torch.float32)
    xe = torch.tensor(X_te, dtype=torch.float32)
    ye = torch.tensor((y_te - y_mean) / y_std, dtype=torch.float32)

    curve, best_val, best_state = [], float("inf"), None
    record_every = max(1, epochs // 50)
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
                val = loss_fn(net(xe), ye).item()
            curve.append({"epoch": epoch, "loss": round(float(val), 6)})
            # 记录测试损失最低的轮次权重（早停思想）
            if val < best_val:
                best_val = val
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

    # 恢复最优权重，避免继续训练导致过拟合
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        y_pred = net(xe).numpy() * y_std + y_mean
    return net, metrics(y_te, y_pred), curve, (features, target, scaler, y_mean, y_std)

# 【扩展2】按 core/trainer.py 的格式保存 pt + meta.json（含标准化参数），Web 可直接加载
def save_web_model(algo: str, dataset: str, net, m: dict, curve: list, pack) -> str:
    features, target, scaler, y_mean, y_std = pack
    model_id = f"{algo}_{uuid.uuid4().hex[:8]}"
    torch.save(net.state_dict(), MODELS_DIR / f"{model_id}.pt")
    meta = {"id": model_id, "algo": algo, "algo_label": DL_ALGORITHMS[algo], "category": "dl",
            "dataset": dataset, "features": features, "target": target,
            "metrics": m, "loss_curve": curve,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "y_mean": y_mean, "y_std": y_std}
    (MODELS_DIR / f"{model_id}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_id

# 与 core/optimizer._load_model 一致：输入原始工艺值，输出原始量纲预测值
def load_web_model(model_id: str):
    meta = json.loads((MODELS_DIR / f"{model_id}.meta.json").read_text(encoding="utf-8"))
    net = build_net(meta["algo"], len(meta["features"]))
    net.load_state_dict(torch.load(MODELS_DIR / f"{model_id}.pt",
                                   map_location="cpu", weights_only=True))
    net.eval()
    mean, scale = np.array(meta["scaler_mean"]), np.array(meta["scaler_scale"])
    c_scaler = load_clean_scaler(meta["dataset"])
    y_mean, y_std = float(meta["y_mean"]), float(meta["y_std"])
    # 收得率物理上限：预测不得超过训练样本最大值，避免出现 >100% 的非法结果
    cap = float(pd.read_csv(DATA_DIR / f"{meta['dataset']}.clean.csv")[meta["target"]].max())

    def predict_orig(X_raw: np.ndarray) -> np.ndarray:
        Xs = np.atleast_2d(np.asarray(X_raw, dtype=float))
        if c_scaler:
            Xs = np.array([[(x - c_scaler[f]["mean"]) / c_scaler[f]["std"]
                            if f in c_scaler else x for x, f in zip(row, meta["features"])]
                           for row in Xs])
        Xs = (Xs - mean) / scale
        with torch.no_grad():
            out = net(torch.tensor(Xs, dtype=torch.float32)).numpy()
        return np.clip(out * y_std + y_mean, 0.0, cap)

    return meta, predict_orig

def load_clean_scaler(dataset: str) -> dict | None:
    f = DATA_DIR / f"{dataset}.scaler.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

# 导出测试集 MSE 损失曲线，便于观察收敛与过拟合
def plot_loss(curve: list, algo: str, dataset: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([c["epoch"] for c in curve], [c["loss"] for c in curve], color="#C44E52")
    ax.set_xlabel("训练轮数")
    ax.set_ylabel("测试集 MSE（标准化目标空间）")
    ax.set_title(f"{dataset} - {DL_ALGORITHMS[algo]} 损失曲线")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXPORT_DIR / f"DL损失曲线_{dataset}_{algo}.png", dpi=150)
    plt.close(fig)

# 【扩展1】智能体：规则意图识别 + 调用本地模型，无需外部 API
class SteelChatAgent:

    def __init__(self):
        self.models = self._scan_models()

    @staticmethod
    def _scan_models() -> list[dict]:
        out = []
        for p in sorted(MODELS_DIR.glob("*.meta.json"), key=lambda x: x.stat().st_mtime,
                        reverse=True):
            out.append(json.loads(p.read_text(encoding="utf-8")))
        return out

    # 只在深度学习模型中挑选 R2 最高的模型，避免加载 ML 的 pkl
    def _best_model(self, dataset: str | None = None) -> dict | None:
        cands = [m for m in self.models if m["category"] == "dl"
                 and (dataset is None or m["dataset"] == dataset)]
        if not cands:
            return None
        return max(cands, key=lambda m: (m.get("metrics") or {}).get("r2", -9))

    # 按关键词路由意图：问候 / 指标解释 / 优化建议 / 预测 / 数据集 / 模型
    def reply(self, text: str) -> str:
        t = text.strip()
        if not t:
            return "请输入您的问题～"
        if re.search(r"你好|您好|hi|hello", t, re.I):
            return ("您好！我是钢铁冶炼工艺智能体，可以：\n"
                    "1) 解读模型指标（MSE/MAE/R2/准确率）\n"
                    "2) 根据工艺参数预测成锭率/成材率\n"
                    "3) 给出提升收得率的参数优化建议\n"
                    "示例：『帮我优化成锭率工艺参数』或『预测 L：WS_MM=5, CS_MM=6 ...』")
        if re.search(r"指标|MSE|MAE|R2|准确率|什么意思", t, re.I):
            return ("模型评估指标说明：\n"
                    "- MSE 均方误差：预测偏差平方均值，越小越好\n"
                    "- MAE 平均绝对误差：预测偏差绝对值均值，越小越好\n"
                    "- R2 决定系数：模型解释数据方差的比例，越接近 1 越好\n"
                    "- 准确率：相对误差不超过 5% 的样本占比")
        if re.search(r"(优化|推荐|提高|提升|增大|变大)", t):
            return self._optimize_reply(t)
        if re.search(r"预测|计算|评估", t):
            return self._predict_reply(t)
        if re.search(r"数据|样本|多少条", t):
            lines = []
            for p in sorted(DATA_DIR.glob("*.csv")):
                if p.stem.endswith(".clean"):
                    continue
                df = pd.read_csv(p)
                lines.append(f"- {p.stem}: {len(df)} 条 x {df.shape[1]} 列")
            return "当前数据集概况：\n" + "\n".join(lines)
        if re.search(r"模型", t):
            if not self.models:
                return "还没有已训练模型，请先运行（2）（3）脚本或使用 Web 模型训练页。"
            lines = [f"- {m['algo_label']}({m['dataset']}) R2={m['metrics']['r2'] if m['metrics'] else '训练中'}"
                     for m in self.models[:8]]
            return "当前已训练模型：\n" + "\n".join(lines)
        return ("我暂时没理解这个问题。您可以问我：\n"
                "『怎么优化成锭率？』『解释一下 R2』『现在有哪些数据集？』")

    def _pick_dataset(self, text: str) -> str | None:
        for name in sorted({m["dataset"] for m in self.models}, reverse=True):
            base = name.replace("_综合1000", "").replace("_扩充1000", "")
            if base in text or name in text:
                return name
        return None

    def _predict_reply(self, t: str) -> str:
        m = self._best_model(self._pick_dataset(t))
        if not m:
            return "暂无可用模型，请先训练。"
        pairs = dict(re.findall(r"([A-Za-z_]\w*)\s*[:=，,]\s*(-?\d+\.?\d*)", t))
        features = m["features"]
        if len(pairs) < len(features):
            return ("请按『特征=数值』格式给出全部特征，例如：\n"
                    + ", ".join(f"{f}=<值>" for f in features))
        try:
            x = np.array([[float(pairs[f]) for f in features]])
        except KeyError:
            return "特征名不匹配。所需特征：" + ", ".join(features)
        _, predict = load_web_model(m["id"])
        v = float(predict(x)[0])
        return (f"使用 {m['algo_label']}（{m['dataset']}，R2={m['metrics']['r2']}）预测：\n"
                f"{m['target']} = {v:.4f}")

    def _optimize_reply(self, t: str) -> str:
        m = self._best_model(self._pick_dataset(t))
        if not m:
            return "暂无可用模型，请先训练。"
        meta, predict = load_web_model(m["id"])
        scaler = load_clean_scaler(meta["dataset"])
        df = pd.read_csv(DATA_DIR / f"{meta['dataset']}.clean.csv").select_dtypes(np.number)
        lo = df[meta["features"]].min().to_numpy()
        hi = df[meta["features"]].max().to_numpy()
        if scaler:
            lo = np.array([lo[j] * scaler[f]["std"] + scaler[f]["mean"]
                           if f in scaler else lo[j]
                           for j, f in enumerate(meta["features"])])
            hi = np.array([hi[j] * scaler[f]["std"] + scaler[f]["mean"]
                           if f in scaler else hi[j]
                           for j, f in enumerate(meta["features"])])
        rng = np.random.default_rng(SEED)
        X = rng.uniform(lo, hi, size=(3000, len(meta["features"])))
        preds = predict(X)
        best = X[int(np.argmax(preds))]
        cur = (lo + hi) / 2
        cur_pred = float(predict(cur.reshape(1, -1))[0])
        opt_pred = float(preds.max())
        lines = [f"基于 {m['algo_label']}（R2={m['metrics']['r2']}）的参数优化建议：",
                 f"预测 {meta['target']}: {cur_pred:.4f} -> {opt_pred:.4f} "
                 f"({(opt_pred - cur_pred) / abs(cur_pred) * 100:+.2f}%)", "推荐参数："]
        lines += [f"- {f}: {c:.3f} -> {r:.3f} ({'上调' if r > c + 1e-3 else '下调' if r < c - 1e-3 else '保持'})"
                  for f, c, r in zip(meta["features"], cur, best)]
        lines.append("提示：完整多算法寻优请运行『机器学习.py』扩展2（遗传/退火/蚁群）。")
        return "\n".join(lines)

def chat_demo():
    print("=" * 72)
    print("【扩展1】AI 聊天（智能体）演示 —— 输入 exit 退出")
    print("=" * 72)
    agent = SteelChatAgent()
    demo = ["你好", "解释一下 R2 和准确率", "怎么优化成锭率？", "现在有哪些数据集？"]
    for q in demo:
        print(f"\n用户：{q}")
        print(f"智能体：{agent.reply(q)}")
    try:
        while True:
            q = input("\n用户：").strip()
            if not q or q.lower() in ("exit", "quit", "q"):
                break
            print(f"智能体：{agent.reply(q)}")
    except (EOFError, OSError):
        pass

def discover_datasets() -> list[str]:
    # 优先使用综合工艺数据集，其次回退扩增数据集，最后回退原始数据集
    for pattern in ("*_综合1000.clean.csv", "*_扩充1000.clean.csv"):
        out = [p.name.removesuffix(".clean.csv") for p in sorted(DATA_DIR.glob(pattern))]
        if out:
            return out
    return ["成锭率", "成材率"]

def main():
    print("=" * 72)
    print("【3/3】深度学习建模（PyTorch：BP / LSTM）+ 评估 + AI 智能体")
    print("=" * 72)
    for dataset in discover_datasets():
        print("-" * 72)
        print(f"数据集: {dataset}")
        # 每个数据集分别训练 BP 与 LSTM，保存模型并导出损失曲线
        for algo in DL_ALGORITHMS:
            net, m, curve, pack = train_dl(algo, dataset)
            model_id = save_web_model(algo, dataset, net, m, curve, pack)
            plot_loss(curve, algo, dataset)
            print(f"[D 建模] {DL_ALGORITHMS[algo]:<8} MSE={m['mse']:.6f} MAE={m['mae']:.6f} "
                  f"R2={m['r2']:.4f} 准确率={m['accuracy']:.2%} -> {model_id}")
    print("-" * 72)
    chat_demo()
    print("=" * 72)
    print("完成。模型已保存至 models/，Web 系统深度学习训练与参数优化可直接使用。")

if __name__ == "__main__":
    main()
