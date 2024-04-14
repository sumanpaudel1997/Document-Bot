# Import necessary modules and define env variables

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import chainlit as cl


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



# text_splitter and system template

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""


messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


@cl.on_chat_start
async def on_chat_start():

    
    await cl.Message(content="Hello there, Welcome to AskAnyQuery related to Data!",).send()
    files = None

    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="I am SuVin, you can ask me anything !!!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    loader = PyPDFLoader(file.path)
    documents = loader.load()
    # Split the text into chunks
    texts = text_splitter.split_documents(documents)


    # Create a Chroma vector store
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': False})
    
    docsearch = await cl.make_async(FAISS.from_documents)(
        texts, embeddings
    )

    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        GoogleGenerativeAI(model="gemini-pro", temperature=0, streaming=True,convert_system_message_to_human=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )


    # Save the metadata and texts in the user session
    
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message:str):

    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []
    
    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()