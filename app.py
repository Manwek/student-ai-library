"""Streamlit interface for the Student AI Library."""

from pathlib import Path

import streamlit as st

from src.agents.answerer import AnswerCitationAgent
from src.agents.retriever import RetrieverAgent
from src.agents.verifier import VerifierAgent


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "documents"

DOCUMENT_TITLES = {
    "ai_strategic_decision_making.pdf": (
        "Artificial Intelligence and Strategic Decision-Making"
    ),
    "corporate_management_in_age_of_ai.pdf": (
        "Corporate Management in the Age of AI"
    ),
    "managing_ai_digital_density_framework.pdf": (
        "Managing Artificial Intelligence within a Digital Density Framework"
    ),
}

SUGGESTED_QUESTIONS = [
    "How can AI support strategic decision-making?",
    "What are the implications of AI for corporate management?",
    "How does digital density relate to business value?",
    "What topics are covered in these papers?",
]


@st.cache_resource
def get_local_pipeline() -> tuple[RetrieverAgent, VerifierAgent]:
    """Create the local retrieval and guardrail agents once per Streamlit session."""
    return RetrieverAgent(), VerifierAgent()


def citation_labels(evidence) -> list[str]:
    """Return concise, de-duplicated citations without exposing retrieved text."""
    return list(
        dict.fromkeys(
            f"{item.document_title}, p. {item.page_number}"
            for item in evidence
        )
    )


def run_question(question: str) -> None:
    """Run the existing pipeline and store only display-safe conversation data."""
    question = question.strip()

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Temporary debugging block.
    # We are deliberately showing the real exception so we can identify
    # which part of the RAG pipeline is failing.
    try:
        retriever, verifier = get_local_pipeline()
        evidence = retriever.retrieve(question)
        verification = verifier.verify(question, evidence)

    except Exception as e:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Error: {type(e).__name__}: {e}",
                "citations": [],
                "refused": True,
            }
        )
        return

    if not verification.allowed:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": verification.message,
                "citations": [],
                "refused": True,
            }
        )
        return

    try:
        answer = AnswerCitationAgent().answer(
            question,
            verification,
        )

    except Exception as e:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Error generating answer: {type(e).__name__}: {e}",
                "citations": [],
                "refused": True,
            }
        )
        return

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citation_labels(
                verification.selected_evidence
            ),
            "refused": False,
        }
    )


st.set_page_config(
    page_title="Student AI Library",
    page_icon="📚",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(
                145deg,
                #f7f9fc 0%,
                #eef3f8 52%,
                #f9fafc 100%
            );
            color: #1f2937;
        }

        .block-container {
            max-width: 900px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2.1rem 2.2rem 1.8rem;
            border: 1px solid #dce5ef;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 10px 30px rgba(31, 55, 80, 0.07);
            margin-bottom: 1.2rem;
        }

        .eyebrow {
            color: #536d8a;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .hero h1 {
            color: #19324d;
            font-size: 2.35rem;
            letter-spacing: -0.03em;
            margin: 0;
        }

        .hero h3 {
            color: #496580;
            font-size: 1.13rem;
            font-weight: 500;
            margin: 0.55rem 0 0.85rem;
        }

        .hero p {
            color: #526170;
            margin-bottom: 0;
        }

        .grounded {
            background: #eaf4f1;
            border: 1px solid #c8e2da;
            border-radius: 12px;
            color: #245b50;
            padding: 0.8rem 1rem;
            margin: 0.9rem 0 1.3rem;
        }

        .grounded strong {
            color: #174d43;
        }

        .suggestion-label {
            color: #536d8a;
            font-size: 0.88rem;
            font-weight: 600;
            margin: 0.15rem 0 0.35rem;
        }

        .chat-user {
            background: #e4edf8;
            border-left: 4px solid #5e83aa;
            border-radius: 10px;
            margin: 1.1rem 0 0.35rem;
            padding: 0.75rem 1rem;
        }

        .chat-answer {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #dce5ef;
            border-radius: 10px;
            box-shadow: 0 4px 14px rgba(31, 55, 80, 0.04);
            padding: 1rem 1.1rem;
        }

        .chat-answer.refused {
            background: #fff8ed;
            border-color: #ecd8b8;
            color: #6c4c20;
        }

        .role-label {
            color: #536d8a;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .footer {
            color: #718096;
            font-size: 0.78rem;
            padding-top: 2rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## 📚 Research Library")
    st.caption(
        "Three locally indexed papers for grounded study and exploration."
    )

    for document in sorted(DOCUMENTS_DIRECTORY.glob("*.pdf")):
        title = DOCUMENT_TITLES.get(
            document.name,
            document.stem.replace("_", " ").title(),
        )

        st.markdown(f"**{title}**")
        st.caption(document.name)

    st.divider()

    st.markdown("## 🛡️ Grounding & Safety")
    st.markdown(
        "Local document retrieval finds relevant passages. "
        "Similarity-based verification checks the evidence before "
        "evidence-only generation. If the evidence is insufficient, "
        "the question is refused automatically."
    )


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">
            Research companion · grounded RAG
        </div>

        <h1>Student AI Library</h1>

        <h3>
            Your grounded research assistant for business and management studies.
        </h3>

        <p>
            Ask questions about the research papers in this library.
            Answers are generated only from the provided documents.
        </p>
    </section>

    <div class="grounded">
        <strong>🔒 Grounded answers only</strong><br>
        Questions outside the document collection are refused so your study
        notes stay anchored to the source material.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="suggestion-label">Start with a suggested question</div>',
    unsafe_allow_html=True,
)

suggestion_columns = st.columns(4)

for column, suggestion in zip(
    suggestion_columns,
    SUGGESTED_QUESTIONS,
):
    if column.button(
        suggestion,
        key=f"suggestion_{suggestion}",
        use_container_width=True,
    ):
        run_question(suggestion)


st.markdown("### Ask the library")

question = st.text_input(
    "Your question",
    placeholder=(
        "e.g. What governance principles should managers "
        "consider when adopting AI?"
    ),
    label_visibility="collapsed",
    key="question_input",
)

if st.button(
    "Ask the library",
    type="primary",
    use_container_width=True,
):
    if question.strip():
        run_question(question)
    else:
        st.info(
            "Enter a question or choose one of the suggestions above."
        )


for message in st.session_state.messages:

    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-user">
                <div class="role-label">You</div>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
        continue

    refused_class = (
        " refused"
        if message.get("refused")
        else ""
    )

    st.markdown(
        f"""
        <div class="chat-answer{refused_class}">
            <div class="role-label">
                Student AI Library
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(message["content"])

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    citations = message.get("citations", [])

    if citations:
        with st.expander(
            "📖 Sources",
            expanded=False,
        ):
            for citation in citations:
                st.markdown(f"- {citation}")


st.markdown(
    '<div class="footer">Student AI Library • Grounded RAG Demonstration</div>',
    unsafe_allow_html=True,
)
```
