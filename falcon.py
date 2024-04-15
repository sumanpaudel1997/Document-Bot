from langchain.vectorstores.faiss import FAISS 
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from langchain.chains import RetrievalQA

loader = PyMuPDFLoader("attention-is-all-you-need-Paper.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# 
# sentence-transformers/all-mpnet-base-v2
vector_db = FAISS.from_documents(docs, embedding_function)
retriever = vector_db.as_retriever(search_type='similarity',search_kwargs={'k':5})
EMB_MPNET_V2 = 'all-mpnet-base-v2'



from langchain_core.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""

prompt = PromptTemplate.from_template(template)

LLM_FALCON_SMALL = "tiiuae/falcon-7b-instruct"
LLM_MISTRAL = 'mistralai/Mistral-7B-Instruct-v0.2'

llm = HuggingFaceHub(repo_id=LLM_FALCON_SMALL)

qa = RetrievalQA.from_chain_type(llm=llm, 
                                chain_type="stuff",
                                retriever=retriever)
qa.combine_documents_chain.llm_chain.prompt = prompt

question = "tell me about transformers"
qa.combine_documents_chain.verbose = True
qa.return_source_documents = True
x = qa({"query":question,})
print(x)


# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# x = rag_chain.invoke("tell me about transformers")
# print(x)

