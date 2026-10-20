"""
FastAPI application entry point for Ban Quality ML.
Exposes REST endpoints and serves the Web Dashboard.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from ml.config import FEATURES, LABEL_MAPPING, MODEL_VERSION
from app.config import ARTIFACTS_PATH
from app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    NeuronActivationsResponse,
)
from app.services.artifact_loader import artifact_manager
from app.services.predictor import predict_single, compute_single_activations, ModelUnavailableError

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ban_quality_ml.api")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "web" / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load artifacts
    logger.info("Initializing application and loading artifacts...")
    success = artifact_manager.load()
    if not success:
        logger.warning("Artifact loading failed at startup. Service will return 503 for predictions.")
    yield
    # Shutdown
    logger.info("Application shutting down...")


app = FastAPI(
    title="Ban Quality ML API",
    description="Industrial Tire Quality Classification ML System",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Prevent stale browser caching for static assets
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Mount static files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/favicon.ico")
def get_favicon():
    """Silence browser favicon requests without 404."""
    from fastapi.responses import Response
    return Response(status_code=204)


# Custom Exception Handlers (NFR-06: Avoid leaking internal stack traces)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(x) for x in err.get("loc", []) if x != "body"])
        errors.append(f"{loc}: {err.get('msg')}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validasi input gagal.", "errors": errors},
    )


@app.exception_handler(ModelUnavailableError)
async def model_unavailable_handler(request: Request, exc: ModelUnavailableError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled server error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Terjadi kesalahan internal pada server."},
    )


# --- REST Endpoints ---

@app.get("/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint checking service and artifact status."""
    return HealthResponse(
        status="ok" if artifact_manager.is_loaded else "error",
        model_loaded=artifact_manager.is_loaded,
        model_version=artifact_manager.model_version,
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """Return model metadata, feature list, decision threshold, and evaluation metrics."""
    if not artifact_manager.is_loaded or not artifact_manager.evaluation:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Informasi model tidak tersedia. Pastikan training telah selesai.",
        )

    metrics = artifact_manager.evaluation.get("metrics", {})
    return ModelInfoResponse(
        model_version=artifact_manager.model_version or MODEL_VERSION,
        features=list(FEATURES),
        threshold=artifact_manager.threshold,
        label_mapping={str(k): v for k, v in LABEL_MAPPING.items()},
        metrics=metrics,
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict tire vulcanization quality for a single set of process parameters."""
    return predict_single(req)


@app.post("/api/predict/batch", response_model=BatchPredictResponse)
def predict_batch(batch_req: BatchPredictRequest):
    """Batch prediction endpoint for multiple process parameters."""
    results = [predict_single(item) for item in batch_req.items]
    return BatchPredictResponse(results=results)


@app.get("/api/evaluation")
def get_evaluation():
    """Return complete evaluation report and baseline comparison."""
    if not artifact_manager.is_loaded or not artifact_manager.evaluation:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluasi belum tersedia.",
        )
    return artifact_manager.evaluation


@app.get("/api/training-history")
def get_training_history():
    """Return epoch-by-epoch training and validation loss/accuracy history."""
    if not artifact_manager.is_loaded or not artifact_manager.training_history:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="History training belum tersedia.",
        )
    return artifact_manager.training_history


@app.get("/api/scaler")
def get_scaler():
    """Return StandardScaler fitted parameters JSON."""
    if not artifact_manager.is_loaded or not artifact_manager.scaler_data:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scaler belum tersedia.",
        )
    return artifact_manager.scaler_data


@app.get("/api/model-config")
def get_model_config():
    """Return model architecture and configuration JSON."""
    if not artifact_manager.is_loaded or not artifact_manager.model_config:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Konfigurasi model belum tersedia.",
        )
    return artifact_manager.model_config


@app.get("/api/model/weights")
def get_model_weights():
    """Return neural network layer weights and biases for neuron synaptic visualization."""
    if not artifact_manager.is_loaded or artifact_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum dimuat. Tidak dapat mengambil bobot neuron.",
        )
    return artifact_manager.model.get_weights_summary()


@app.post("/api/model/activations", response_model=NeuronActivationsResponse)
def get_neuron_activations(req: PredictRequest):
    """Compute real-time layer-by-layer neuron activations for interactive visualizers."""
    return compute_single_activations(req)


@app.get("/", response_class=HTMLResponse)
def get_dashboard(request: Request):
    """Serve the single-page application industrial dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_version": artifact_manager.model_version or MODEL_VERSION,
            "model_loaded": artifact_manager.is_loaded,
        },
    )
