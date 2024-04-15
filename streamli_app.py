from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl



@cl.on_chat_start
async def on_chat_start():
    model = HuggingFaceHub(repo_id='meta-llama/Llama-2-7b')
    prompt = ChatPromptTemplate.from_messages(
        [
            (
               "system",
                "You're a very friendly AI provides accurate and lengthy answer.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
