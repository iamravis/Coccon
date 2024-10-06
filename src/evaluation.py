from mistralai import Mistral
import os
from dotenv import load_dotenv
import json

load_dotenv()

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
client = Mistral(api_key=MISTRAL_API_KEY)

# Base model ID
BASE_MODEL_ID = 'open-mistral-7b'
# Fine-tuned model ID
FINE_TUNED_MODEL_ID = 'ft:open-mistral-7b:5aa386c9:20241006:254ae550'

# Load evaluation dataset
with open('../data/eval/eval.jsonl', 'r') as f:
    eval_data = [json.loads(line) for line in f]

# Function to get model response
def get_model_response(model_id, prompt):
    response = client.chat.complete(
        model=model_id,
        messages=[{"role": 'user', "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Collect responses from both models
base_model_responses = []
fine_tuned_model_responses = []

for item in eval_data:
    prompt = item['content']
    expected_response = item.get('response', '')

    # Base model response
    base_response = get_model_response(BASE_MODEL_ID, prompt)
    base_model_responses.append({
        'prompt': prompt,
        'response': base_response,
        'expected': expected_response
    })

    # Fine-tuned model response
    fine_tuned_response = get_model_response(FINE_TUNED_MODEL_ID, prompt)
    fine_tuned_model_responses.append({
        'prompt': prompt,
        'response': fine_tuned_response,
        'expected': expected_response
    })

# Save responses to files for analysis
with open('base_model_responses.jsonl', 'w') as f:
    for item in base_model_responses:
        f.write(json.dumps(item) + '\n')

with open('fine_tuned_model_responses.jsonl', 'w') as f:
    for item in fine_tuned_model_responses:
        f.write(json.dumps(item) + '\n')
