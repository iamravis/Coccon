import streamlit as st
import requests
import os

# Set up Streamlit app with a title and image
st.title("Cocoon")

# Verify if the image file exists
if os.path.isfile("../images/cocoon.png"):
    st.image("../images/cocoon.png", width=200)
else:
    st.error("Image file not found.")

# Function to read the system prompt from a file
def get_system_prompt():
    with open("system_prompt.txt", "r") as file:
        return file.read()

# Load the system prompt
system_prompt = get_system_prompt()

# Function to call Pixtral API via OpenRouter
def call_pixtral_api(user_input):
    # OpenRouter Pixtral API URL
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer pqmKVrIjJjkKQMhvRslPapP7QzNV2A1I",  # Your actual API key
        "Content-Type": "application/json"
    }

    # Prepare the API request body
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{system_prompt}\n\nUser: {user_input}\nChatbot:"}
            ]
        }
    ]

    data = {
        "model": "mistralai/pixtral-12b",
        "messages": messages
    }

    # Send request to OpenRouter API
    response = requests.post(api_url, headers=headers, json=data)

    # Handle API response
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from the model.")
    else:
        return f"Error: {response.status_code} - {response.text}"

# Initialize the app with a welcome message from the bot
if "bot_message" not in st.session_state:
    st.session_state.bot_message = "Hello, I’m Cocoon! How can I assist you today with your wellbeing or any questions?"

# Display bot message
st.markdown(f"**Cocoon**: {st.session_state.bot_message}")

# User input
user_input = st.text_input("Ask me about parenthood ...")

# When user submits input
if st.button("Send"):
    if user_input:
        # Call Pixtral API with user input
        bot_response = call_pixtral_api(user_input)
        
        # Update bot message with the response
        st.session_state.bot_message = bot_response

        # Display the updated bot response
        st.markdown(f"**You**: {user_input}")
        st.markdown(f"**Cocoon**: {bot_response}")
