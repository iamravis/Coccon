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
from mistralai import Mistral
import json

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')


#FINE_TUNED_MODEL_ID = 'ft:open-mistral-7b:5aa386c9:20241006:254ae550'  # open-mistral
FINE_TUNED_MODEL_ID = 'ft:open-mistral-nemo:5aa386c9:20241006:34d17884' # nemo
#FINE_TUNED_MODEL_ID = 'ft:mistral-small-latest:5aa386c9:20241006:12e3d2bd' # mistral-small
#FINE_TUNED_MODEL_ID = 'ft:mistral-large-latest:5aa386c9:20241006:1efa1a26' # mistral-large

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

# Store the system prompt
system_prompt = "You are a healthcare chatbot which answers the asked healthcare questions strictly in hindi"  # Replace with your actual prompt
if len(messages) == 0:
    messages.append({"role": "system", "content": system_prompt})

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

# Asynchronous function to send the user's message to the model
async def chat_with_model(prompt):
    try:
        # Initialize the Mistral client inside the function
        client = Mistral(api_key=MISTRAL_API_KEY)
        conversation_history = messages
        response = await client.chat.complete_async(
            model=FINE_TUNED_MODEL_ID,
            messages=conversation_history,
        )
        assistant_reply = response.choices[0].message.content
        return assistant_reply
    except asyncio.TimeoutError:
        return "The request timed out. Please try again."
    except Exception as e:
        return f"An error occurred: {e}"

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


# New route to update the assistant's message
@rt('/update_response', methods=['POST'])
async def update_response():
    global messages

    # The last message is the assistant's placeholder
    # The second to last message is the user's message
    last_user_message = messages[-2]['content']

    # Asynchronous request to the model for assistant response
    retry_attempts = 3
    assistant_response = None
    for attempt in range(retry_attempts):
        try:
            assistant_response = await chat_with_model(last_user_message)
            break  # If the request is successful, break out of the loop
        except Exception as e:
            print(f"Error during attempt {attempt + 1}: {e}")
            if attempt == retry_attempts - 1:
                assistant_response = "An error occurred and the assistant couldn't respond."
                break

    # Update the assistant's message in the messages list
    messages[-1]['content'] = assistant_response

    # Return the assistant's bubble
    assistant_bubble = ChatMessage(assistant_response, False)

    # Also, scroll to bottom
    return Html(
        assistant_bubble,
        Script("scrollToBottom();")
    )

serve()
