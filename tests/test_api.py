import os
import unittest
from fastapi.testclient import TestClient

# Désactive Azure Monitor pendant les tests pour éviter bruit réseau/quota
os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"] = ""

from main import app


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)

        body = r.json()
        self.assertIn("status", body)
        self.assertIn(body["status"], ["healthy", "degraded"])

        # si ton main expose ces champs, on les vérifie aussi
        if "model_loaded" in body:
            self.assertIsInstance(body["model_loaded"], bool)

    def test_predict_bad_request_if_empty_text(self):
        r = self.client.post("/predict", json={"text": ""})
        self.assertEqual(r.status_code, 400)

    def test_feedback_ok(self):
        payload = {
            "prediction_id": "test-id-123",
            "text": "this flight was horrible i hate it",
            "predicted_label": 0,
            "predicted_proba": 0.06,
            "user_validated": False
        }
        r = self.client.post("/feedback", json=payload)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})
