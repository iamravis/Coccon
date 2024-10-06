from mistralai import Mistral
import os
from dotenv import load_dotenv

load_dotenv()

# Get the API key from environment variable
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')


client = Mistral(api_key=MISTRAL_API_KEY)

chat_train = client.files.upload(file={
    "file_name": "train.jsonl",
    "content": open("../data/train/train.jsonl", "rb"),
})
chat_eval = client.files.upload(file={
    "file_name": "eval.jsonl",
    "content": open("../data/eval/eval.jsonl", "rb"),
})