import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Transformer QA System", page_icon="🤖")

st.title("🤖 Transformer Question Answering")
st.markdown("Enter a context passage and a question to find the answer using a DistilBERT model.")

@st.cache_resource
def load_qa_engine():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_engine = load_qa_engine()

col1, col2 = st.columns([2, 1])

with col1:
    context = st.text_area(
        "Context Passage:", 
        height=250,
        placeholder="Paste the text you want the AI to read here..."
    )

with col2:
    question = st.text_input(
        "Your Question:", 
        placeholder="What would you like to know?"
    )
    
    find_button = st.button("🔍 Find Answer")

if find_button:
    if context and question:
        with st.spinner("Analyzing text..."):

            result = qa_engine(question=question, context=context)
            
            st.success(f"**Answer:** {result['answer']}")
            st.metric("Confidence Score", f"{round(result['score'] * 100, 2)}%")
            
            start, end = result['start'], result['end']
            highlighted_text = (
                context[:start] + 
                f":red[{context[start:end]}]" + 
                context[end:]
            )
            st.markdown("---")
            st.markdown("**Highlighted in Context:**")
            st.write(highlighted_text)
    else:
        st.warning("Please provide both a context and a question.")