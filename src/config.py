"""Central configuration for the local Student AI Library services."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "documents"
CHROMA_DB_DIRECTORY = PROJECT_ROOT / "data" / "chroma_db"

# This model generates embeddings on the local machine, never via an embedding API.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# The approved model files are now cached locally, so embeddings run offline and no
# further model download or update check is attempted.
EMBEDDING_LOCAL_FILES_ONLY = True

CHROMA_COLLECTION_NAME = "student_ai_library_chunks"

# Default number of evidence chunks the Retriever Agent returns per question.
DEFAULT_RETRIEVAL_COUNT = 5

# ChromaDB returns lower distance values for more similar chunks. This initial value
# separates the observed relevant examples (up to ~0.80) from an irrelevant example
# (starting at ~1.66); it should later be calibrated with the evaluation set.
MAX_EVIDENCE_DISTANCE = 1.0

# Stable Gemini model with a free-tier option; the Gemini client is the only module
# that uses this setting.
GEMINI_MODEL = "gemini-flash-lite-latest"
