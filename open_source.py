# import os

# from transformers import AutoTokenizer
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores.faiss import FAISS
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_hub import HuggingFaceHub
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate

# loader = PyPDFLoader("attention-is-all-you-need-Paper.pdf")
# documents = loader.load()


# from langchain_community.embeddings import (
#     HuggingFaceEmbeddings
# )

# embeddings = HuggingFaceEmbeddings(
#     model_name = 'sentence-transformers/all-MiniLM-L12-v2', 
#     model_kwargs = {'device': "cpu"}
# )

# text_splitter = RecursiveCharacterTextSplitter(
#      chunk_size=1000,
#      chunk_overlap=100,
# )

# docs = text_splitter.split_documents(documents)

# vector_db = FAISS.from_documents(
#     documents=docs, embedding=embeddings
# )


# retriever = vector_db.as_retriever(
#     search_kwargs={'k': 5,}
#     )

# relavent_docs = retriever.get_relevant_documents("tell me core concepts about this paper")

# from langchain.chains import RetrievalQA
    
# custom_prompt_template = """
# ### System:
# You are an AI assistant that follows instructions extremely well. Help as much as you can.
# ### User:
# You are a research assistant for an artificial intelligence student. Use only the following information to answer user queries:
# Context= {context}
# Question= {question}
# ### Assistant:
# """

# prompt = PromptTemplate(template=custom_prompt_template,
#                         input_variables=["question", "context"])
# LLM_FALCON_SMALL = "tiiuae/falcon-7b-instruct"
# llm = HuggingFaceHub(repo_id=LLM_FALCON_SMALL)

# from langchain.chains import ConversationalRetrievalChain

# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory

# memory = ConversationBufferMemory(
#   memory_key="chat_history", output_key='answer', return_messages=False)
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vector_db.as_retriever(search_kwargs={"k": 1}),
#     get_chat_history=lambda o:o,
#     memory=memory,
#     combine_docs_chain_kwargs={'prompt': prompt})

# query = "What are the core concepts of presented in these papers, and how do the approaches differ or align?"



# result = qa_chain("What is this about?")
# print(result['answer'])
# from langchain_community.document_loaders.pdf import PyPDFLoader


from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
import os
from langchain.document_loaders.pdf import PyMuPDFLoader

from dotenv import load_dotenv

load_dotenv()

loader = PyMuPDFLoader('resume.pdf')
# loader = PyMuPDFLoader('attention-is-all-you-need-Paper.pdf')

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
documents = text_splitter.split_documents(docs)

vector_db = FAISS.from_documents(documents=documents, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# print(vector_db.similarity_search('attention')[0].page_content)

llm = GoogleGenerativeAI(model='gemini-pro')

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the following questions based on the provided context.
Try to give lenghty answers.Do not give false answer
<context>
{context}
</context>
Question: {input}           
""")


from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector_db.as_retriever()

from langchain.chains.retrieval import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever, document_chain)

res = retrieval_chain.invoke({"input": "tell me skills of Suman in document"})
print(res['answer'])