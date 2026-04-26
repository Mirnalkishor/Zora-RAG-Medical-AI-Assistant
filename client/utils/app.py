import streamlit as st
from components.upload import render_uploader
from components.history_download import render_history_download
from components.chatUI import render_chat

st.set_page_config(page_title="Zora — Medical AI", page_icon="🧬", layout="wide")
st.title("🧬 Zora — RAG Medical AI Assistant")
st.caption("Upload medical PDFs and ask questions powered by AI")


render_uploader()
render_chat()
render_history_download()