import faiss
import pickle
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import sys


@st.cache_resource


def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("knowledge.index")
    with open("docs.pkl", "rb") as f:
        documents = pickle.load(f)
    return model, index, documents


model, index, documents = load_resouce()


def ask_question(question):
    embedding = model.encode([question])
    _, indices = index.search(np.array(embedding), k=1)
    return documents[indices[0][0]]["text"]


st.title("MY rag bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("ask me anything"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

answer = ask_question(prompt)

with st.chat_message("assistant"):
    response = f"result \n \n {answer}"
    st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
