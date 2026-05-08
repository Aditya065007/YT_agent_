import os
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["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"] = st.secrets["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"]

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

if "db" not in st.session_state:
    st.session_state.db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = []

def build_db_from_url(video_url: str):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    if st.session_state.db is None:
        st.session_state.db = FAISS.from_documents(docs, embeddings)
    else:
        new_db = FAISS.from_documents(docs, embeddings)
        st.session_state.db.merge_from(new_db)

def get_response(query: str) -> str:
    docs = st.session_state.db.similarity_search(query, k=4)
    docs_page_content = " ".join([d.page_content for d in docs])
    history_text = ""
    for exchange in st.session_state.chat_history:
        history_text += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    prompt = PromptTemplate(
        input_variables=["question", "docs", "history"],
        template="""
        You are a helpful assistant that answers questions about YouTube videos
        based on their transcripts.

        Previous conversation history:
        {history}

        Answer the following question: {question}
        Using the following transcript content: {docs}

        Rules:
        - Only use factual information from the transcript
        - Take into account the conversation history above for context
        - If you don't have enough information, say "I don't know"
        - Format your response in numbered points
        - Keep each point clear and concise
        - Start each point on a new line
        """,
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "question": query,
        "docs": docs_page_content,
        "history": history_text
    })
    return response

st.set_page_config(page_title="YouTube Chat", page_icon="🎬")
st.title("🎬 YouTube Video Chatbot")

with st.sidebar:
    st.header("📺 Add YouTube Video")
    video_url = st.text_input("Paste YouTube URL here")
    load_btn = st.button("Load Video")

    if load_btn and video_url:
        if video_url in st.session_state.processed_urls:
            st.warning("This video is already loaded!")
        else:
            with st.spinner("Loading transcript and building database..."):
                try:
                    build_db_from_url(video_url)
                    st.session_state.processed_urls.append(video_url)
                    st.success("Video loaded successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.processed_urls:
        st.divider()
        st.subheader("✅ Loaded Videos")
        for url in st.session_state.processed_urls:
            st.write(f"• {url}")

    if st.button("🗑️ Clear Everything"):
        st.session_state.db = None
        st.session_state.chat_history = []
        st.session_state.processed_urls = []
        st.rerun()

if st.session_state.db is None:
    st.info("👈 Paste a YouTube URL in the sidebar to get started.")
else:
    for exchange in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(exchange["user"])
        with st.chat_message("assistant"):
            st.write(exchange["assistant"])

    user_question = st.chat_input("Ask anything about the video...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(user_question)
            st.write(response)
        st.session_state.chat_history.append({
            "user": user_question,
            "assistant": response
        })