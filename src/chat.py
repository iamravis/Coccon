import streamlit as st
import os
from mistralai import Mistral
from dotenv import load_dotenv
import asyncio

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

# Initialize the Mistral client
client = Mistral(api_key=MISTRAL_API_KEY)

# Fine-tuned model ID from your previous output
FINE_TUNED_MODEL_ID = 'ft:open-mistral-7b:5aa386c9:20241006:254ae550'

# Streamlit app configuration
st.set_page_config(
    page_title="Advanced Chatbot with Mistral AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit footer and hamburger menu
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Initialize session state for conversation history and system prompt
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    st.session_state['system_prompt'] = ''

# Sidebar for user inputs
st.sidebar.title("Chatbot Settings")

# Input for personal prompt
st.sidebar.subheader("Personal Prompt")
personal_prompt = st.sidebar.text_area("Enter your personal prompt:", st.session_state['system_prompt'], height=100)
if st.sidebar.button("Set Personal Prompt"):
    st.session_state['system_prompt'] = personal_prompt
    st.session_state['messages'] = []  # Clear conversation history when prompt changes
    if personal_prompt.strip():
        st.session_state['messages'].append({
            "role": "system",
            "content": personal_prompt
        })
    st.success("Personal prompt set successfully.")

# User prompt input
user_prompt = st.sidebar.text_area("Enter your prompt:", "", height=150)

# Adjustable settings
st.sidebar.subheader("Model Parameters")
max_tokens = st.sidebar.slider("Max Tokens:", min_value=50, max_value=500, value=150)
temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7)
top_p = st.sidebar.slider("Top P:", min_value=0.0, max_value=1.0, value=0.9)

if st.sidebar.button("Clear Chat History"):
    st.session_state['messages'] = []
    st.success("Chat history cleared.")

# Main app
st.title("🤖 Advanced Chatbot with Fine-Tuned Mistral AI Model")

# Function to interact with the fine-tuned model
async def chat_with_model(prompt):
    try:
        messages = st.session_state['messages'] + [{"role": 'user', "content": prompt}]
        response = await client.chat.complete_async(
            model=FINE_TUNED_MODEL_ID,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        assistant_reply = response.choices[0].message.content
        return assistant_reply
    except Exception as e:
        return f"An error occurred: {e}"

# Display conversation history
for message in st.session_state['messages']:
    if message['role'] == 'system':
        st.markdown(f"**System Prompt:** {message['content']}")
    elif message['role'] == 'user':
        st.markdown(f"**You:** {message['content']}")
    elif message['role'] == 'assistant':
        st.markdown(f"**Assistant:** {message['content']}")

# When the user submits a prompt
if st.sidebar.button("Get Response"):
    if user_prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            assistant_response = asyncio.run(chat_with_model(user_prompt))
            # Append user message and assistant response to conversation history
            st.session_state['messages'].append({"role": 'user', "content": user_prompt})
            st.session_state['messages'].append({"role": 'assistant', "content": assistant_response})
            st.markdown(f"**You:** {user_prompt}")
            st.markdown(f"**Assistant:** {assistant_response}")
