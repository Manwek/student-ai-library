"""Isolated Gemini client for future answer generation from verified evidence only."""

import os

from dotenv import load_dotenv
from google import genai

from src.config import GEMINI_MODEL


class MissingGeminiAPIKeyError(RuntimeError):
    """Raised when Gemini is requested without a locally configured API key."""


def get_gemini_api_key() -> str:
    """Read the key from the environment (or an existing local .env file), never log it."""
    load_dotenv(override=False)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise MissingGeminiAPIKeyError(
            "GEMINI_API_KEY is not configured. Add it to your local .env file or environment."
        )
    return api_key


def is_gemini_api_key_configured() -> bool:
    """Safely report only whether a non-empty key is available, never its value."""
    try:
        get_gemini_api_key()
    except MissingGeminiAPIKeyError:
        return False
    return True


class GeminiClient:
    """Send only a question plus pre-verified evidence to the Gemini SDK."""

    def __init__(self, model: str = GEMINI_MODEL) -> None:
        self.model = model
        self._client = genai.Client(api_key=get_gemini_api_key())

    def generate_from_verified_evidence(self, question: str, verified_evidence_text: str) -> str:
        """Generate from supplied evidence only; retrieval and verification remain elsewhere."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if not verified_evidence_text or not verified_evidence_text.strip():
            raise ValueError("verified_evidence_text must not be empty")

        prompt = (
            "Question:\n"
            f"{question.strip()}\n\n"
            "Verified evidence:\n"
            f"{verified_evidence_text.strip()}"
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text or ""
