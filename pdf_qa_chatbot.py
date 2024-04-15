import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (GoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def main():

    st.title("Chat With Your PDF")
    st.write("Welcome to the PDF Chatbot! Ask me anything about the uploaded document.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a PDF file", type="pdf", key='pdf')

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        # Load the PDF file
        loader = PyMuPDFLoader(tmp_file_path)
        docs = loader.load()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_documents(docs)

        # Create the vector database
        vector_db = FAISS.from_documents(
            documents=documents, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

        # Create the language model and prompt
        llm = GoogleGenerativeAI(model='gemini-pro')
        prompt = ChatPromptTemplate.from_template("""
Answer the following questions based on the provided context.
Try to give lenghty answers.Do not give false answer
<context>
{context}
</context>
Question: {input}           
""")

        retriever = vector_db.as_retriever()

        document_chain = create_stuff_documents_chain(llm, prompt)

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Chat interface

        user_input = st.text_input("Ask Question to your Document:")
        submit = st.button("Submit")
        if submit:
            result = retrieval_chain.invoke(
                {"input": user_input})
            st.write("Answer:\n\n", result["answer"])


if __name__ == "__main__":
    main()
