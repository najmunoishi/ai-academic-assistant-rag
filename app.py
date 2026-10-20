import streamlit as st
from rag_engine import AcademicRAG

st.set_page_config(page_title="Academic RAG Assistant", page_icon="📚")
st.title("AI Academic Assistant")
st.write("Ask a question and inspect the retrieved source documents.")

@st.cache_resource
def load_assistant():
    return AcademicRAG("data/academic_faqs.jsonl")

try:
    assistant = load_assistant()
    question = st.text_input("Your question")
    if question:
        answer, sources = assistant.answer(question)
        st.write(answer)
        st.subheader("Source details")
        for source in sources:
            st.write(
                f"{source['source_id']} — {source['title']} "
                f"(similarity: {source['score']:.3f})"
            )
except Exception as exc:
    st.error(f"Could not load the assistant: {exc}")
