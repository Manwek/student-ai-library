# Student AI Library

Student AI Library is a small grounded Retrieval-Augmented Generation (RAG) application for MSc and business-management research. It retrieves evidence from a curated set of research papers, verifies that the evidence is sufficient, and generates cited answers only when the evidence supports the question.

## Project Goal

The application answers questions only from the provided research papers. When sufficient evidence is not found, it refuses the question with a clear, deterministic message instead of guessing.

## Key Features

- PDF text extraction with PyMuPDF
- Page-level document metadata
- Local Sentence Transformer embeddings
- ChromaDB vector database
- Retriever Agent
- Deterministic Guardrail/Verifier
- Gemini Answer/Citation Agent
- Document/page citations
- Exact refusal mechanism for unsupported questions
- Streamlit interface
- Small, explicit evaluation set

## Architecture

```text
User Question
    → Retriever
    → ChromaDB
    → Guardrail/Verifier
    → Answer/Citation Agent
    → Gemini
    → Answer + Citations
```

The Retriever finds relevant document chunks locally. The Guardrail/Verifier checks the retrieved evidence before generation. Gemini is not called when verification fails; the pipeline returns the refusal message directly.

## Why the Agents Are Separated

This project intentionally does not use three autonomous LLMs:

- **Retriever** retrieves evidence from the local index.
- **Verifier** deterministically checks evidence quality and citation metadata.
- **Answerer** generates an answer only from verified evidence.

Separating these responsibilities makes the workflow more explainable, easier to test, and easier to evaluate than an opaque, all-in-one agent.

## Grounding and Safety

The Guardrail/Verifier applies several deterministic checks:

- a similarity-distance threshold separates sufficiently relevant evidence from weak matches;
- selected evidence must contain non-empty text;
- selected evidence must contain a document title, positive page number, and chunk identifier;
- Gemini receives only the evidence that passed these checks;
- questions that fail verification return exactly:

  > Insufficient knowledge — this is not covered in the provided documents.

The Gemini API key is loaded from a local `.env` file. `.env` is excluded by `.gitignore` and must never be committed.

## Documents

The current collection contains three papers:

- `ai_strategic_decision_making.pdf` — AI-augmented strategic decision-making, including evidence from entrepreneurs and investors.
- `corporate_management_in_age_of_ai.pdf` — possible impacts of AI on corporate leadership, management structures, governance, and liability.
- `managing_ai_digital_density_framework.pdf` — AI, digital density, business-model value propositions, organizational capabilities, and governance.

Original source URLs are not documented in this repository. Before redistributing any paper, verify the applicable copyright, reuse, and licensing terms with the relevant publisher or rights holder.

## Evaluation

The current evaluation set contains:

- 10 evaluation questions
- 5 answerable questions
- 5 unanswerable questions
- 5/5 answerable questions passed
- 5/5 unanswerable questions passed
- 10/10 evaluation tests passed
- 0 Gemini calls during evaluation

The relevant Retriever/Guardrail/Answerer tests currently report 14 passed tests. These results demonstrate the tested behaviors; they do not prove that the system is perfect or complete.

## Installation

Python 3.12 is recommended. On macOS, Homebrew Python 3.12 may be useful if it is not already installed.

```bash
git clone <repository-url>
cd Student-AI-Library
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Gemini API Key

You need your own Gemini API key. Create a local `.env` file containing only your own key:

```dotenv
GEMINI_API_KEY=your_key_here
```

Never commit the actual `.env` file. The placeholder above is not a real key.

## Running the Application

```bash
source .venv/bin/activate
streamlit run app.py
```

The application normally opens at:

[<http://localhost:8501>](https://student-ai-library.streamlit.app/)

## Running Tests

```bash
pytest
```

## Project Structure

```text
app.py
requirements.txt
src/
  agents/          # Retriever, Verifier, and Answer/Citation agents
  ingestion/       # PDF extraction, chunking, and indexing
  services/        # Gemini client and local vector-store service
  config.py
  models.py
data/
  documents/       # Curated PDF research papers
tests/
  evaluation_questions.json
  test_evaluation.py
  test_answerer.py
  test_verifier.py
  manual_end_to_end.py
  validate_ingestion.py
  validate_retriever.py
```

Local secrets, virtual environments, vector indexes, Python caches, and Streamlit caches are development artifacts and are not part of the commit structure.

## Limitations and Future Improvements

- The document collection is small.
- The similarity threshold needs further calibration.
- The evaluation set is intentionally small.
- PDF extraction quality can vary by document.
- Gemini output still depends on model behavior.
- Future work could expand the evaluation dataset, improve retrieval and reranking, and add more robust citation validation.

## Portfolio / Learning Outcomes

This project demonstrates practical RAG architecture, AI product design, grounding, guardrails, evaluation, API security, and explainable agentic workflows. It also shows how deterministic checks can constrain a generative model to a transparent evidence boundary.
