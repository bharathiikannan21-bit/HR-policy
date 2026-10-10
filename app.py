import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from huggingface_hub import InferenceClient


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"

st.set_page_config(
    page_title="HR Bot",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CARTOON UI
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eadcff 0%,
        #c9a7ff 50%,
        #a875e8 100%
    );
}

/* Main title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #4b176d;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #63357c;
    font-size: 18px;
    margin-bottom: 25px;
}

/* Cartoon bot */
.bot {
    text-align: center;
    font-size: 75px;
    margin-bottom: -10px;
}

/* Chat bubbles */
.user-msg {
    background: #ffffff;
    padding: 15px 20px;
    border-radius: 20px 20px 5px 20px;
    margin: 10px 0;
    color: #3d2050;
    box-shadow: 0 4px 10px rgba(0,0,0,0.10);
}

.bot-msg {
    background: #f3e8ff;
    padding: 15px 20px;
    border-radius: 20px 20px 20px 5px;
    margin: 10px 0;
    color: #3d2050;
    box-shadow: 0 4px 10px rgba(0,0,0,0.10);
}

/* Buttons */
.stButton > button {
    border-radius: 15px;
    border: none;
    background: #6d28a9;
    color: white;
    font-weight: 600;
}

.stButton > button:hover {
    background: #50127d;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #d8b9ff;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown('<div class="bot">🤖💼</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="title">HR Bot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your Friendly AI HR Assistant 💜</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )

    max_tokens = st.slider(
        "📝 Max Tokens",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )

    st.divider()

    st.subheader("📚 HR Documents")

    uploaded_files = st.file_uploader(
        "Upload HR documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if st.button("🗑️ New Chat"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# --------------------------------------------------
# DOCUMENT LOADING
# --------------------------------------------------

def load_documents(files):

    documents = []

    for file in files:

        file_path = os.path.join(
            "temp_" + file.name
        )

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)

        else:
            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

        documents.extend(loader.load())

        os.remove(file_path)

    return documents


# --------------------------------------------------
# CREATE RAG DATABASE
# --------------------------------------------------

def create_vector_database(files):

    documents = load_documents(files)

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    # Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS
    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db


# --------------------------------------------------
# PROCESS DOCUMENTS
# --------------------------------------------------

if uploaded_files:

    if st.session_state.vector_db is None:

        with st.spinner("📚 Reading HR documents..."):

            st.session_state.vector_db = (
                create_vector_database(uploaded_files)
            )

        st.success("✅ HR knowledge base created!")


# --------------------------------------------------
# RETRIEVER
# --------------------------------------------------

def retrieve_context(question):

    if st.session_state.vector_db is None:
        return ""

    retriever = st.session_state.vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return context


# --------------------------------------------------
# HUGGING FACE LLM
# --------------------------------------------------

def generate_answer(question, context):

    if not HF_API_KEY:
        return "⚠️ Please add your Hugging Face API key in the .env file."

    client = InferenceClient(
        api_key=HF_API_KEY
    )

    prompt = f"""
You are HR Bot, a helpful AI HR assistant.

Answer the employee's question using ONLY
the HR policy context provided below.

If the answer is not available in the context,
say:

"I couldn't find this information in the HR
documents. Please contact HR for assistance."

Do not invent HR policies.

HR POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{question}

Give a clear and concise answer.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-msg">
            👤 <b>You:</b> {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="bot-msg">
            🤖 <b>HR Bot:</b> {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "💬 Ask HR Bot about company policies..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    context = retrieve_context(question)

    if not context:

        answer = (
            "📚 Please upload your HR policy documents "
            "first so I can answer your question."
        )

    else:

        with st.spinner("🤖 Thinking..."):

            answer = generate_answer(
                question,
                context
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()