import time
import requests
import streamlit as st

st.set_page_config(
    page_title="Air Paradis - Test API",
    page_icon="✈️",
    layout="centered",
)

# URL Azure par défaut
DEFAULT_API_URL = "https://rg-airparadis-hkc3bkbwg0gsbccd.westeurope-01.azurewebsites.net"

# Permet d'écraser l'URL via Streamlit secrets plus tard si besoin
API_URL_FROM_SECRETS = st.secrets.get("API_URL", DEFAULT_API_URL)

# Timeout un peu large pour éviter les faux timeouts si Azure est lent à répondre
TIMEOUT = 120

st.title("Air Paradis - Interface de test de l'API")
st.write(
    "Saisissez un texte, lancez la prédiction, puis validez ou non le résultat."
)

# État Streamlit
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if "last_root" not in st.session_state:
    st.session_state.last_root = None

if "last_health" not in st.session_state:
    st.session_state.last_health = None


def safe_get(url: str):
    try:
        start = time.perf_counter()
        response = requests.get(url, timeout=TIMEOUT)
        elapsed = time.perf_counter() - start

        try:
            data = response.json()
        except Exception:
            data = response.text

        return {
            "ok": True,
            "status_code": response.status_code,
            "data": data,
            "elapsed": elapsed,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


def safe_post(url: str, payload: dict):
    try:
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=TIMEOUT)
        elapsed = time.perf_counter() - start

        try:
            data = response.json()
        except Exception:
            data = response.text

        return {
            "ok": True,
            "status_code": response.status_code,
            "data": data,
            "elapsed": elapsed,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


api_url = st.text_input("URL de l'API", value=API_URL_FROM_SECRETS)
text = st.text_area("Texte à analyser", value="the trip was bad", height=180)

col_a, col_b = st.columns(2)

with col_a:
    if st.button("Tester /"):
        result = safe_get(f"{api_url}/")
        st.session_state.last_root = result

with col_b:
    if st.button("Tester /health"):
        result = safe_get(f"{api_url}/health")
        st.session_state.last_health = result

st.subheader("État de l'API")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Route /**")
    if st.session_state.last_root:
        result = st.session_state.last_root
        if result["ok"]:
            st.success(f"OK - {result['status_code']} - {result['elapsed']:.3f} s")
            st.json(result["data"])
        else:
            st.error(result["error"])
    else:
        st.info("Aucun test effectué.")

with col2:
    st.markdown("**Route /health**")
    if st.session_state.last_health:
        result = st.session_state.last_health
        if result["ok"]:
            status_code = result["status_code"]
            data = result["data"]

            if isinstance(data, dict) and data.get("status") == "healthy":
                st.success(f"Healthy - {status_code} - {result['elapsed']:.3f} s")
            elif isinstance(data, dict) and data.get("status") == "degraded":
                st.warning(f"Degraded - {status_code} - {result['elapsed']:.3f} s")
            else:
                st.info(f"Réponse reçue - {status_code} - {result['elapsed']:.3f} s")

            st.json(data)
        else:
            st.error(result["error"])
    else:
        st.info("Aucun test effectué.")

st.subheader("Prédiction")

if st.button("Analyser"):
    if not text.strip():
        st.warning("Veuillez saisir un texte.")
    else:
        payload = {"text": text}
        result = safe_post(f"{api_url}/predict", payload)

        if result["ok"]:
            if result["status_code"] == 200 and isinstance(result["data"], dict):
                data = result["data"]

                st.session_state.prediction_data = {
                    "text": text,
                    "prediction_id": data.get("prediction_id"),
                    "label": data.get("label"),
                    "proba_pos": data.get("proba_pos"),
                    "raw": data,
                    "elapsed": result["elapsed"],
                }

                st.success(f"Prédiction reçue en {result['elapsed']:.3f} s")
            else:
                st.error(f"Erreur API {result['status_code']}")
                st.json(result["data"])
        else:
            st.error(f"Erreur de connexion : {result['error']}")

if st.session_state.prediction_data:
    pred = st.session_state.prediction_data

    sentiment = "positif" if pred["label"] == 1 else "négatif"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sentiment", sentiment)
    with c2:
        try:
            proba_value = float(pred["proba_pos"])
            st.metric("Probabilité positive", f"{proba_value:.4f}")
        except Exception:
            st.metric("Probabilité positive", str(pred["proba_pos"]))
    with c3:
        st.metric("Temps de réponse", f"{pred['elapsed']:.3f} s")

    st.markdown("**Réponse brute de /predict**")
    st.json(pred["raw"])

    st.subheader("Validation utilisateur")
    col_ok, col_ko = st.columns(2)

    with col_ok:
        if st.button("✅ Prédiction correcte"):
            payload = {
                "prediction_id": pred["prediction_id"],
                "text": pred["text"],
                "predicted_label": pred["label"],
                "predicted_proba": pred["proba_pos"],
                "user_validated": True,
            }
            result = safe_post(f"{api_url}/feedback", payload)

            if result["ok"]:
                if result["status_code"] == 200:
                    st.success("Validation envoyée.")
                    st.json(result["data"])
                else:
                    st.error(f"Erreur feedback {result['status_code']}")
                    st.json(result["data"])
            else:
                st.error(f"Erreur de connexion : {result['error']}")

    with col_ko:
        if st.button("❌ Prédiction incorrecte"):
            payload = {
                "prediction_id": pred["prediction_id"],
                "text": pred["text"],
                "predicted_label": pred["label"],
                "predicted_proba": pred["proba_pos"],
                "user_validated": False,
            }
            result = safe_post(f"{api_url}/feedback", payload)

            if result["ok"]:
                if result["status_code"] == 200:
                    st.success("Rejet envoyé.")
                    st.json(result["data"])
                else:
                    st.error(f"Erreur feedback {result['status_code']}")
                    st.json(result["data"])
            else:
                st.error(f"Erreur de connexion : {result['error']}")
