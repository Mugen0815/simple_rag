

from operator import itemgetter
from typing import List, Tuple

from fastapi import Depends, FastAPI, Request, Response
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_community.vectorstores import FAISS

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
import os
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms import Ollama

CHROMA_PATH = "chroma"
DATA_PATH = os.environ.get('DATA_PATH')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
)
retriever2 = vectorstore.as_retriever()

embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
retriever = db.as_retriever()


_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(openai_api_key=OPENAI_API_KEY,temperature=0)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | ChatOpenAI(openai_api_key=OPENAI_API_KEY) | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)

ollama = Ollama(model="llama2",  base_url="http://172.30.0.1:8001")
#model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
model = ollama
prompt = ChatPromptTemplate.from_template("Give me a summary about {topic} in a paragraph or less.")
chain2 = prompt | model
chain3 = ChatOpenAI(openai_api_key=OPENAI_API_KEY) | StrOutputParser()


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain, path="/chat", enable_feedback_endpoint=True) # results in /chat/invoke, /chat/batch, /chat/stream

add_routes(app, chain2, path="/chat2", enable_feedback_endpoint=False) # results in
add_routes(app, chain3, path="/chat3", enable_feedback_endpoint=False) # results in


@app.post("/completion", include_in_schema=False)
async def simple_invoke(request: Request) -> Response:
    """Handle a request."""
    requestbody = await request.json()
    topic = requestbody.get('topic')
    if topic:
        text_to_summarize = f"Give me a summary about {topic} in a paragraph or less."
        result = model.invoke(text_to_summarize)
    else:
        result = "No topic provided"
    
    return result



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)