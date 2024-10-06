# ============================================
# Import Necessary Libraries
# ============================================

# Standard Libraries
import logging
import uuid
from functools import partial, lru_cache
from threading import Lock
import os
import asyncio

# Third-party Libraries
from fasthtml.common import *
from fasthtml.components import Zero_md
from dotenv import load_dotenv
import json

# New Imports for RAG functionalities
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain.schema import Document
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Literal, List
from langchain_mistralai.chat_models import ChatMistralAI

# Language detection
from langdetect import detect

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

# Model IDs
FINE_TUNED_MODEL_ID = 'ft:open-mistral-nemo:5aa386c9:20241006:34d17884'  # Replace with your model ID

# Initialize the session state for conversation
if 'messages' not in globals():
    messages = []

# Function to render local Markdown (used for messages)
def render_local_md(md, css=''):
    css_template = Template(Style(css), data_append=True)
    return Zero_md(css_template, Script(md, type="text/markdown"))

# CSS to fix styling issues
css = '.markdown-body {background-color: unset!important; color: unset!important;}'
_render_local_md = partial(render_local_md, css=css)

# Headers
zeromd_headers = [Script(type="module", src="https://cdn.jsdelivr.net/npm/zero-md@3?register")]
chat_headers = [
    Script(src="https://cdn.tailwindcss.com"),
    Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css"),
    Script(src="https://unpkg.com/htmx.org@1.9.10"),
    Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&display=swap"),
    Style(f"""
    body, html {{
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
    }}
    body {{
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center center;
    }}
    .chat-bubble-user {{ background-color: rgba(246, 246, 246, 0.8) !important; color: black !important; }}
    .chat-bubble-assistant {{ background-color: rgba(236, 236, 236, 0.8) !important; color: black !important; }}
    .font-fira-code {{ font-family: 'Fira Code', monospace !important; }}
    .chat-bubble {{ font-size: 0.9rem !important; backdrop-filter: blur(5px); }}
    #msg-input {{ font-size: 0.9rem !important; }}
    .card {{ backdrop-filter: blur(10px); background-color: rgba(255, 255, 255, 0.5) !important; }}
    """),
    Script("""
    function scrollToBottom() {
        const chatlist = document.getElementById('chatlist');
        chatlist.scrollTop = chatlist.scrollHeight;
    }

    function scrollWithDelay() {
        requestAnimationFrame(() => {
            scrollToBottom();
            setTimeout(scrollToBottom, 100);
        });
    }

    document.body.addEventListener('htmx:afterSwap', function(event) {
        if (event.detail.target.id === 'chatlist') {
            scrollWithDelay();
        }
    });

    window.addEventListener('load', scrollWithDelay);
    const observer = new MutationObserver(scrollWithDelay);
    observer.observe(document.getElementById('chatlist'), { childList: true, subtree: true });
    """)
]

all_headers = zeromd_headers + chat_headers

app, rt = fast_app()

# Store the default system prompt (will be adjusted based on language)
default_system_prompt = """
You are an assistant that answers user questions.
"""

if len(messages) == 0:
    messages.append({"role": "system", "content": default_system_prompt})

# Chat message component (renders a chat bubble)
def ChatMessage(msg, user):
    bubble_class = "chat-end" if user else "chat-start"
    bubble_color = "chat-bubble-user" if user else "chat-bubble-assistant"
    role = "You" if user else "Assistant"

    md = _render_local_md(msg)
    return Div(
        Div(role, cls="chat-header"),
        Div(md, cls=f"chat-bubble {bubble_color} font-fira-code p-3 rounded-lg shadow-md"),
        cls=f"chat {bubble_class}"
    )

# The input field for the user message
def ChatInput():
    return Input(name='msg', id='msg-input', placeholder="Type your message here...",
                 cls="input input-bordered w-full font-fira-code")

# ============================================
# Setup Vector Store and RAG Components
# ============================================

# Function to set up the vector store
def setup_vector_store():
    import os

    persist_directory = './chroma_db'
    embeddings = MistralAIEmbeddings()

    if os.path.exists(persist_directory):
        # Load existing vectorstore
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        # Read the text files
        files = [
            "../data/finetune_data/all_content.txt",
            '../data/finetune_data/all_content_2.txt',
            '../data/finetune_data/all_content_3.txt',
            '../data/finetune_data/all_content_4.txt',
            '../data/finetune_data/all_content_5.txt',
            '../data/finetune_data/all_content_6.txt',
            '../data/finetune_data/all_content_7.txt'
        ]

        docs_list = []
        for file in files:
            loader = TextLoader(file)
            docs = loader.load()
            docs_list.extend(docs)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()

    # Create retriever
    retriever = vectorstore.as_retriever()
    return retriever

# Initialize the retriever
retriever = setup_vector_store()

# Define LLM

mistral_model = FINE_TUNED_MODEL_ID  # Use your fine-tuned model ID
llm = ChatMistralAI(model=mistral_model, temperature=0)

# Router Data Model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question, choose to route it to web search or a vectorstore.",
    )

# LLM with structured output for routing
structured_llm_router = llm.with_structured_output(RouteQuery)

# Router instructions
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to the content in the files provided.

Use the vectorstore for questions on these topics. For all else, use web-search."""

# Retrieval Grader Data Model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# LLM with structured output for grading
structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.

Give a binary 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# Grader prompt
doc_grader_prompt = "Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}"

# RAG Prompt (will be adjusted based on language)
rag_prompt_template = """You are an assistant for question-answering tasks.

Use the following retrieved context to answer the question.

If you don't know the answer, just say that you don't know.

Use at most three sentences and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ============================================
# Asynchronous function to generate assistant response using RAG
# ============================================

async def generate_assistant_response():
    global messages

    # Get the last user message
    user_message = messages[-1]['content']

    # Detect language
    try:
        user_language = detect(user_message)
    except:
        user_language = 'en'  # Default to English if detection fails

    # Set system prompt and RAG prompt based on language
    if user_language == 'hi':
        system_prompt = "आप एक सहायक हैं जो उपयोगकर्ता के प्रश्नों का उत्तर हिंदी में देते हैं।"
        rag_prompt = """आप एक सहायक हैं जो प्रश्नों के उत्तर देते हैं।

निम्नलिखित संदर्भ का उपयोग करके प्रश्न का उत्तर दें।

यदि आपको उत्तर नहीं पता है, तो बस कहें कि आपको नहीं पता।

अधिकतम तीन वाक्यों का उपयोग करें और उत्तर संक्षिप्त रखें।

प्रश्न: {question}

संदर्भ: {context}

उत्तर:"""
    else:
        system_prompt = "You are an assistant that answers user questions in English."
        rag_prompt = rag_prompt_template

    # Replace the system prompt in messages
    if messages[0]['role'] == 'system':
        messages[0]['content'] = system_prompt
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Route the question
    routing_prompt = [SystemMessage(content=router_instructions), HumanMessage(content=user_message)]
    routing_decision = structured_llm_router.invoke(routing_prompt)
    datasource = routing_decision.datasource

    # Initialize documents
    documents = []

    if datasource == 'vectorstore':
        # Retrieve documents from vectorstore
        docs = retriever.get_relevant_documents(user_message)

        # Grade documents
        filtered_docs = []
        web_search_flag = False
        for doc in docs:
            doc_grader_prompt_formatted = doc_grader_prompt.format(document=doc.page_content, question=user_message)
            grading_prompt = [SystemMessage(content=doc_grader_instructions), HumanMessage(content=doc_grader_prompt_formatted)]
            score = structured_llm_doc_grader.invoke(grading_prompt)
            grade = score.binary_score.lower()
            if grade == 'yes':
                filtered_docs.append(doc)
            else:
                web_search_flag = True  # Set flag to use web search
        documents = filtered_docs

    # If no relevant documents or web search is needed
    if datasource == 'websearch' or not documents:
        # Implement web search functionality or return a default message
        if user_language == 'hi':
            assistant_reply = "क्षमा करें, मुझे कोई प्रासंगिक जानकारी नहीं मिली। क्या आप अपना प्रश्न स्पष्ट कर सकते हैं?"
        else:
            assistant_reply = "Sorry, I couldn't find any relevant information. Could you please clarify your question?"
        return assistant_reply

    # Generate answer using retrieved documents
    context = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=context, question=user_message)
    generation_prompt = [SystemMessage(content=system_prompt), HumanMessage(content=rag_prompt_formatted)]
    response = llm.invoke(generation_prompt)
    assistant_reply = response.content

    return assistant_reply

# ============================================
# Routes and Server Setup
# ============================================

@rt('/')
def index():
    # Retrieve existing chat messages
    chat_items = []
    for msg in messages:
        if msg['role'] == 'user':
            chat_items.append(ChatMessage(msg['content'], True))
        elif msg['role'] == 'assistant':
            chat_items.append(ChatMessage(msg['content'], False))

    page = Div(cls="min-h-screen flex items-center justify-center bg-[#e1e1e1]")(
        Div(cls="w-full max-w-6xl px-4")(
            Div(cls="card bg-base-100 shadow-xl")(
                Div(cls="card-body")(
                    Div(cls="flex flex-col mb-4")(
                        Div(cls="flex items-center justify-between mb-2")(
                            Div(cls="flex items-center gap-2")(
                                Img(src="../logo.png", alt="Logo", cls="h-8 w-8"),  # Adjust size as needed
                                H2("ReadWise", cls="card-title text-2xl font-bold font-fira-code")
                            ),
                        ),
                        Div(cls="border-b border-gray-300 w-full")
                    ),
                    Div(id="chatlist", cls="space-y-8 h-[65vh] overflow-y-auto mb-4")(
                        *chat_items  # Display existing messages
                    ),
                    Form(hx_post="/send", hx_target="#chatlist", hx_swap="beforeend", cls="flex gap-2")(
                        ChatInput(),
                        Button("Send", cls="btn bg-black text-white hover:bg-gray-800 font-fira-code", type="submit")
                    )
                )
            )
        )
    )
    return Html(*all_headers, Body(cls="font-fira-code")(page))

@rt('/send', methods=['POST'])
async def send(msg: str):
    global messages
    if not msg:
        return ""

    # Add user's message to the conversation history
    messages.append({"role": "user", "content": msg})

    # Create the user's chat bubble
    user_bubble = ChatMessage(msg, True)

    # Create a placeholder for the assistant's response with hx attributes
    assistant_placeholder = Div(
        ChatMessage("...", False),
        hx_post="/update_response",
        hx_trigger="load",
        hx_target="this",
        hx_swap="outerHTML"
    )

    # Return the user's message and assistant's placeholder
    return Html(
        user_bubble,
        assistant_placeholder,
        Script("document.getElementById('msg-input').value = '';")  # Clear input box after message is sent
    )

# New route to update the assistant's message using RAG
@rt('/update_response', methods=['POST'])
async def update_response():
    global messages

    # Asynchronous request to generate assistant response
    retry_attempts = 3
    assistant_response = None
    for attempt in range(retry_attempts):
        try:
            assistant_response = await generate_assistant_response()
            break  # If the request is successful, break out of the loop
        except Exception as e:
            print(f"Error during attempt {attempt + 1}: {e}")
            if attempt == retry_attempts - 1:
                assistant_response = "An error occurred, and the assistant could not respond."
                break

    # Update the assistant's message in the messages list
    messages.append({"role": "assistant", "content": assistant_response})

    # Return the assistant's bubble
    assistant_bubble = ChatMessage(assistant_response, False)

    # Also, scroll to bottom
    return Html(
        assistant_bubble,
        Script("scrollToBottom();")
    )

# Start the server using Uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
