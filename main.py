import os
import re
import uuid
import time
import shutil
import logging
import threading
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

# Dossier persistant (Azure App Service : /home est persisté)
NLTK_DATA_DIR = os.getenv("NLTK_DATA", "/home/nltk_data").strip() or "/home/nltk_data"
os.environ["NLTK_DATA"] = NLTK_DATA_DIR  # utile pour NLTK
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# Stopwords en global (initialement vide)
stop_words: set = set()
STOPWORDS_READY: bool = False
STOPWORDS_ERROR: Optional[str] = None

# Pour éviter de spammer des tentatives en boucle
_last_stopwords_attempt_ts = 0.0
_STOPWORDS_ATTEMPT_COOLDOWN_SEC = 60.0  # 1 minute entre tentatives par process

stemmer = PorterStemmer()
token_re = re.compile(r"[a-z]+")

NEGATIONS = {"no", "nor", "not", "never"}
EXTRA_STOPWORDS = {"rt", "amp", "im", "dont", "u", "ur", "ive", "youre", "thats"}


def _file_lock(path: str):
    """
    Context manager simple de verrou fichier (Linux). Permet d'éviter
    plusieurs téléchargements NLTK en parallèle avec gunicorn -w 4.
    """
    try:
        import fcntl
    except Exception:
        # Si jamais fcntl indispo, on fait sans (moins safe)
        fcntl = None

    class _LockCtx:
        def __init__(self, lock_path: str):
            self.lock_path = lock_path
            self.f = None

        def __enter__(self):
            os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
            self.f = open(self.lock_path, "a+")
            if fcntl:
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.f:
                if fcntl:
                    try:
                        fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
                try:
                    self.f.close()
                except Exception:
                    pass

    return _LockCtx(path)


def _cleanup_possible_corruption() -> None:
    """
    Nettoie les artefacts fréquents quand NLTK a un zip corrompu.
    """
    corpora_dir = os.path.join(NLTK_DATA_DIR, "corpora")
    stopwords_dir = os.path.join(corpora_dir, "stopwords")
    stopwords_zip = os.path.join(corpora_dir, "stopwords.zip")

    # On supprime seulement si ça existe : ça force un download propre
    try:
        if os.path.isfile(stopwords_zip):
            os.remove(stopwords_zip)
        if os.path.isdir(stopwords_dir):
            shutil.rmtree(stopwords_dir, ignore_errors=True)
    except Exception as e:
        logger.warning("nltk_cleanup_failed: %s", e)


def _load_stopwords_only() -> set:
    """
    Charge stopwords (doit être présent). Lève LookupError si absent.
    """
    sw = set(stopwords.words("english"))
    # garde négations, ajoute stopwords custom
    sw = (sw - NEGATIONS) | EXTRA_STOPWORDS
    return sw


def ensure_stopwords_available(force: bool = False) -> None:
    """
    Assure que stopwords est disponible, sinon tente de le télécharger
    dans /home/nltk_data. Protégé par verrou fichier (multi-workers).
    Ne bloque pas en boucle (cooldown).
    """
    global stop_words, STOPWORDS_READY, STOPWORDS_ERROR, _last_stopwords_attempt_ts

    if STOPWORDS_READY and stop_words:
        return

    now = time.time()
    if not force and (now - _last_stopwords_attempt_ts) < _STOPWORDS_ATTEMPT_COOLDOWN_SEC:
        return

    _last_stopwords_attempt_ts = now

    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    lock_path = os.path.join(NLTK_DATA_DIR, ".nltk_download.lock")

    with _file_lock(lock_path):
        # 1) re-check sous verrou : un autre worker a peut-être déjà téléchargé
        try:
            stop_words = _load_stopwords_only()
            STOPWORDS_READY = True
            STOPWORDS_ERROR = None
            logger.info("nltk_stopwords_ready_cached")
            return
        except LookupError:
            pass
        except Exception as e:
            # si corpus corrompu -> nettoyage puis retry download
            logger.warning("nltk_stopwords_load_error_pre_download: %s", e)

        # 2) attempt download
        try:
            logger.info("nltk_stopwords_missing_downloading_to=%s", NLTK_DATA_DIR)
            _cleanup_possible_corruption()
            # Download dans le dossier persistant
            nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception as e:
            STOPWORDS_READY = False
            STOPWORDS_ERROR = f"download_failed: {e}"
            logger.exception("nltk_stopwords_download_failed: %s", e)
            return

        # 3) verify after download
        try:
            stop_words = _load_stopwords_only()
            STOPWORDS_READY = True
            STOPWORDS_ERROR = None
            logger.info("nltk_stopwords_download_ok")
            return
        except Exception as e:
            STOPWORDS_READY = False
            STOPWORDS_ERROR = f"post_download_load_failed: {e}"
            logger.exception("nltk_stopwords_post_download_failed: %s", e)
            return


def preprocess_stem(text: str, min_len: int = 2) -> str:
    """Nettoyage texte + suppression stopwords + stemming."""
    if not isinstance(text, str):
        return ""

    # IMPORTANT : stopwords requis -> on s'assure qu'ils sont dispo
    if not STOPWORDS_READY:
        ensure_stopwords_available(force=False)

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)               # mentions
    text = re.sub(r"#", " ", text)                  # symbole hashtag
    text = re.sub(r"[^a-z\s]", " ", text)           # caractères non alpha
    text = re.sub(r"\s+", " ", text).strip()

    tokens = token_re.findall(text)

    # Si stopwords pas prêt, on ne tente pas de "faire semblant"
    # (tu as dit que c'était super important pour l'exercice)
    if not STOPWORDS_READY:
        raise RuntimeError(f"NLTK stopwords not ready: {STOPWORDS_ERROR}")

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


def _init_stopwords_background():
    """
    Lance l'init stopwords en thread pour ne PAS bloquer le démarrage.
    Ça évite les 504 si le download est lent.
    """
    try:
        ensure_stopwords_available(force=True)
    except Exception as e:
        logger.exception("stopwords_background_init_failed: %s", e)


@app.on_event("startup")
def startup_event():
    # 1) charge le modèle au démarrage API
    load_model_once()

    # 2) stopwords en arrière-plan (verrou multi-workers)
    t = threading.Thread(target=_init_stopwords_background, daemon=True)
    t.start()


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Air Paradis - Sentiment API",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None,
        "azure_monitor_enabled": AZURE_MONITOR_ENABLED,
        "stopwords_ready": STOPWORDS_READY,
        "nltk_data_dir": NLTK_DATA_DIR,
    }


@app.get("/health")
def health():
    # Donne un vrai état de santé incluant le modèle + stopwords
    status = "healthy"
    details = {
        "model_loaded": model is not None,
        "model_load_error": model_load_error,
        "stopwords_ready": STOPWORDS_READY,
        "stopwords_error": STOPWORDS_ERROR,
        "nltk_data_dir": NLTK_DATA_DIR,
    }

    if model is None or not STOPWORDS_READY:
        status = "degraded"

    return {"status": status, **details}


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
        raise HTTPException(status_code=503, detail=f"model unavailable: {model_load_error}")

    # Stopwords requis (sinon résultats faux) -> 503 clair
    if not STOPWORDS_READY:
        ensure_stopwords_available(force=False)
    if not STOPWORDS_READY:
        raise HTTPException(
            status_code=503,
            detail=f"nltk stopwords not ready: {STOPWORDS_ERROR}",
        )

    pred_id = str(uuid.uuid4())

    try:
        label, proba = predict_sentiment(text)
    except RuntimeError as e:
        # Exemple : stopwords pas prêt au moment du preprocess
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("prediction_failed: %s", e)
        raise HTTPException(status_code=500, detail="prediction failed")

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
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )
