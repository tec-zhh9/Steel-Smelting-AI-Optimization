from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from core import data_manager, optimizer, trainer

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="基于人工智能的钢铁冶炼工艺参数优化系统")
# 挂载前端静态资源与 Jinja2 模板目录
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

# 首页统计与数据集管理接口
@app.get("/api/stats")
def api_stats():
    return data_manager.stats_summary()

@app.get("/api/datasets")
def api_datasets():
    return {"datasets": data_manager.list_datasets()}

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    try:
        info = data_manager.save_upload(file.filename, await file.read())
        return {"ok": True, **info}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.delete("/api/dataset/{name}")
def api_delete_dataset(name: str):
    data_manager.delete_dataset(name)
    return {"ok": True}

@app.get("/api/preview")
def api_preview(name: str, page: int = 1):
    try:
        return data_manager.preview(name, page)
    except FileNotFoundError:
        raise HTTPException(404, f"数据集不存在: {name}")

@app.post("/api/clean")
def api_clean(dataset: str = Form(...), missing: str = Form("mean"),
              outliers: bool = Form(False), standardize: bool = Form(False)):
    try:
        return {"ok": True, **data_manager.clean_dataset(dataset, missing, outliers, standardize)}
    except Exception as e:
        raise HTTPException(400, str(e))

# 模型训练：启动后台线程，前端轮询 /api/train/progress 获取进度
@app.post("/api/train")
def api_train(dataset: str = Form(...), category: str = Form(...),
              algorithm: str = Form(...), epochs: int = Form(100),
              lr: float = Form(0.01), test_size: float = Form(0.2)):
    algos = trainer.ML_ALGORITHMS if category == "ml" else trainer.DL_ALGORITHMS
    if algorithm not in algos:
        raise HTTPException(400, f"算法 {algorithm} 不属于 {category}")
    try:
        trainer.start_training(dataset, category, algorithm, epochs, lr, test_size)
        return {"ok": True}
    except RuntimeError as e:
        raise HTTPException(409, str(e))

@app.get("/api/train/progress")
def api_train_progress():
    return trainer.TRAIN_STATE

@app.get("/api/models")
def api_models():
    return {"models": trainer.list_models()}

# 某数据集各工艺参数的取值范围，前端据此渲染优化输入框
@app.get("/api/params")
def api_params(dataset: str):
    try:
        return optimizer.dataset_feature_bounds(dataset)
    except FileNotFoundError:
        raise HTTPException(404, f"数据集不存在: {dataset}")

# 参数优化：输入当前工艺参数，返回推荐参数与预测提升幅度
@app.post("/api/optimize")
def api_optimize(model_id: str = Form(...), params: str = Form(...)):
    import json as _json
    try:
        current = {k: float(v) for k, v in _json.loads(params).items()}
        return {"ok": True, **optimizer.optimize(model_id, current)}
    except FileNotFoundError:
        raise HTTPException(404, "模型不存在，请先训练")
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/export")
def api_export():
    try:
        path = optimizer.export_last()
        return FileResponse(path, filename=path.name, media_type="text/csv")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

# 本地启动：python app.py（浏览器访问 http://127.0.0.1:8000）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
