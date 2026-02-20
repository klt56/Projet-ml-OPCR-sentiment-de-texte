import os
import re
import uuid
import logging
from typing import Tuple, Optional

# Réduit le bruit TensorFlow dans les logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# ------------------------------------------------------------
# Chargement des variables d'environnement (.env)
# ------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------
# Config générale
# ------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "./models/bilstm_stemming_v1")
MODEL_NAME = os.getenv("MODEL_NAME", "bilstm_stemming")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.5"))

# Azure Application Insights
# Variable demandée : APPLICATIONINSIGHTS_CONNECTION_STRING
AI_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
AZURE_MONITOR_ENABLED = False

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airparadis")

if AI_CONN:
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=AI_CONN)
        AZURE_MONITOR_ENABLED = True
        logger.info("azure_monitor_configured")
    except Exception as e:
        logger.exception("azure_monitor_config_failed: %s", e)
else:
    logger.warning("azure_monitor_not_configured_no_connection_string")

# ------------------------------------------------------------
# NLTK preprocessing (stemming + stopwords)
# ------------------------------------------------------------
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Téléchargement stopwords si nécessaire
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    stop_words = set(stopwords.words("english"))
except Exception:
    # Fallback si problème NLTK
    stop_words = set()

# On garde les négations car importantes en sentiment
NEGATIONS = {"no", "nor", "not", "never"}
stop_words = stop_words - NEGATIONS
stop_words |= {"rt", "amp", "im", "dont", "u", "ur", "ive", "youre", "thats"}

stemmer = PorterStemmer()
token_re = re.compile(r"[a-z]+")


def preprocess_stem(text: str, min_len: int = 2) -> str:
    """Nettoyage texte + suppression stopwords + stemming."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)               # mentions
    text = re.sub(r"#", " ", text)                  # symbole hashtag
    text = re.sub(r"[^a-z\s]", " ", text)           # caractères non alpha
    text = re.sub(r"\s+", " ", text).strip()

    tokens = token_re.findall(text)
    tokens = [t for t in tokens if len(t) >= min_len and t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


# ------------------------------------------------------------
# Chargement modèle (robuste)
# ------------------------------------------------------------
model: Optional[tf.keras.Model] = None
model_load_error: Optional[str] = None


def load_model_once() -> None:
    """Charge le modèle une seule fois (idempotent)."""
    global model, model_load_error
    if model is not None:
        return
    try:
        model = tf.keras.models.load_model(MODEL_DIR, compile=False)
        model_load_error = None
        logger.info("model_loaded_from=%s", MODEL_DIR)
    except Exception as e:
        model = None
        model_load_error = str(e)
        logger.exception("model_load_failed_from=%s", MODEL_DIR)


def predict_sentiment(text: str) -> Tuple[int, float]:
    """Retourne (label, proba_pos)."""
    if model is None:
        raise RuntimeError("Model not loaded")

    cleaned = preprocess_stem(text)
    x = tf.constant([[cleaned]], dtype=tf.string)  # modèle string input
    proba_pos = float(model(x, training=False).numpy().reshape(-1)[0])
    label = 1 if proba_pos >= THRESHOLD else 0
    return label, proba_pos


# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="Air Paradis - Sentiment API", version="1.0")


class PredictIn(BaseModel):
    text: str


class PredictOut(BaseModel):
    prediction_id: str
    label: int
    proba_pos: float
    model_name: str
    model_version: str


class FeedbackIn(BaseModel):
    prediction_id: str
    text: str
    predicted_label: int
    predicted_proba: float
    user_validated: bool


@app.on_event("startup")
def startup_event():
    # Charge le modèle au démarrage API
    load_model_once()


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Air Paradis - Sentiment API",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None,
        "azure_monitor_enabled": AZURE_MONITOR_ENABLED,
    }


@app.get("/health")
def health():
    # Donne un vrai état de santé incluant le modèle
    if model is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "model_load_error": model_load_error,
        }
    return {"status": "healthy", "model_loaded": True}


@app.get("/diag/ai")
def diag_ai():
    """Route simple pour forcer l'envoi d'un log test vers Application Insights."""
    logger.warning("ai_connection_test", extra={"source": "manual_diag"})
    return {"status": "sent", "azure_monitor_enabled": AZURE_MONITOR_ENABLED}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text vide")

    # Sécurité si modèle indisponible
    if model is None:
        load_model_once()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"model unavailable: {model_load_error}",
        )

    pred_id = str(uuid.uuid4())
    label, proba = predict_sentiment(text)

    logger.info(
        "prediction_made",
        extra={
            "prediction_id": pred_id,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predicted_label": int(label),
            "predicted_proba": float(proba),
            "text_len": len(text),
        },
    )

    return PredictOut(
        prediction_id=pred_id,
        label=int(label),
        proba_pos=float(proba),
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )


@app.post("/feedback")
def feedback(payload: FeedbackIn):
    base = {
        "prediction_id": payload.prediction_id,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "predicted_label": int(payload.predicted_label),
        "predicted_proba": float(payload.predicted_proba),
        "user_validated": bool(payload.user_validated),
        "text_len": len(payload.text or ""),
    }

    if payload.user_validated:
        logger.info("prediction_validated", extra=base)
    else:
        logger.warning("prediction_rejected", extra=base)

    return {"status": "ok"}


# Lancement local direct : python main.py
