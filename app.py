import streamlit as st
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from docx import Document

# ======== 1. SETUP: Load secrets securely ========
# Load Google API key from Streamlit secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Create temporary service account credentials file from secrets
service_account_info = dict(st.secrets["google_service_account"])  # Convert AttrDict to dict
with open("temp_google_creds.json", "w") as f:
    json.dump(service_account_info, f)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_google_creds.json"

# ======== 2. UTILITY FUNCTIONS ========
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def get_document_text(files):
    text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        elif file.name.endswith(".docx"):
            text += extract_text_from_docx(file)
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(language="English"):
    prompt_template = f"""
    You are a legal assistant. Answer the question in **{language}** using only the context below.
    Be accurate and include all relevant legal details. If not found, say "Not available in context."

    Context: {{context}}
    Question: {{question}}

    Answer in {language}:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, language="English"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(language)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Answer:", response["output_text"])

def summarize_document_dual(text, language="English"):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    summary_prompt = f"""
    Summarize this legal contract in **{language}** in two ways:

    1. **Client-friendly Summary** – Plain language for non-lawyers. Highlight key points clearly, avoid jargon.
    2. **Detailed Legal Summary** – Accurate legal summary including obligations, payment terms, termination clauses, confidentiality, and penalties.

    Use clear formatting with headers or bullet points.

    Contract Text:
    {text[:10000]}

    Return both summaries in the format:
    ---
    CLIENT-FRIENDLY SUMMARY:
    [Summary here]

    ---
    DETAILED LEGAL SUMMARY:
    [Summary here]
    """
    response = model.invoke(summary_prompt)
    content = response.content if hasattr(response, "content") else str(response)

    parts = content.split("---")
    plain_summary = ""
    detailed_summary = ""

    for part in parts:
        if "CLIENT-FRIENDLY SUMMARY" in part:
            plain_summary = part.strip().split("CLIENT-FRIENDLY SUMMARY:")[-1].strip()
        elif "DETAILED LEGAL SUMMARY" in part:
            detailed_summary = part.strip().split("DETAILED LEGAL SUMMARY:")[-1].strip()

    return plain_summary, detailed_summary

def extract_risky_clauses(text, language="English"):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    clause_prompt = f"""
    From the following legal contract text, extract and list **risky or sensitive clauses** in **{language}** related to:
    - Termination
    - Indemnity
    - Confidentiality
    - Penalties
    - Limitation of Liability

    Respond **in {language}**

    For each clause, provide:
    1. **Clause Type**
    2. **Exact Clause Text**
    3. **Why it may be risky**

    Format the output clearly with headers and bullets.

    Text:
    {text[:10000]}

    Risky Clauses in {language}:
    """
    response = model.invoke(clause_prompt)
    return response.content if hasattr(response, "content") else str(response)

# ======== 3. MAIN STREAMLIT APP ========
def main():
    st.set_page_config("Legal Document Assistant")
    st.header("📄 कानूनी सहायक")

    languages = [
        "English", "Hindi", "Bengali", "Gujarati", "Kannada", "Malayalam", "Marathi", "Punjabi", "Tamil", "Telugu",
        "Urdu", "Arabic", "Chinese", "French", "German", "Spanish", "Italian", "Portuguese", "Russian", "Japanese",
        "Korean", "Turkish", "Dutch", "Thai", "Vietnamese", "Indonesian", "Polish", "Ukrainian", "Greek", "Hebrew",
        "Romanian", "Hungarian", "Czech", "Swedish", "Finnish", "Norwegian", "Danish"
    ]

    with st.sidebar:
        st.title("Upload & Process Contract")
        st.session_state["language"] = st.selectbox("Output Language for All Sections", languages)
        pdf_docs = st.file_uploader("Upload Legal PDF or DOCX Files", accept_multiple_files=True, type=["pdf", "docx"])
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Reading and processing contract..."):
                raw_text = get_document_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state['text'] = raw_text
                st.session_state['summaries'] = summarize_document_dual(raw_text, st.session_state["language"])
                st.success("✅ Document Processed")

    if "text" in st.session_state and "summaries" in st.session_state:
        tab1, tab2, tab3 = st.tabs(["📄 Summary", "⚠️ Risky Clauses", "🔍 Ask a Question"])

        with tab1:
            st.subheader("📄 Document Summary")
            summary_type = st.radio("Select Summary Type", ["🧑‍🏫 Client-Friendly", "👩‍⚖️ Legal Summary"])
            if summary_type == "🧑‍🏫 Client-Friendly":
                st.markdown(st.session_state["summaries"][0])
            else:
                st.markdown(st.session_state["summaries"][1])

        with tab2:
            st.subheader("⚠️ Risky Clauses Identified")
            clauses = extract_risky_clauses(st.session_state['text'], st.session_state["language"])
            st.markdown(clauses)

        with tab3:
            st.subheader("🔍 Ask Questions about the Contract")
            question = st.text_input("Ask your legal question (e.g., What are the termination conditions?)")
            if question:
                user_input(question, st.session_state["language"])

if __name__ == "__main__":
    main()
