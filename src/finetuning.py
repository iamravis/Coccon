import os
import json
import time
from rich import print
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.client import MistralClient


load_dotenv()

# Get the API key from environment variable
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

client = Mistral(api_key=MISTRAL_API_KEY)

def pprint(obj):
    print(json.dumps(obj.dict(), indent=4))

# 1. Upload the dataset
with open("../data/train/train.jsonl", "rb") as f:
    train_data = client.files.create(file=("../data/train/train_data.jsonl", f))
with open("../data/eval/eval.jsonl", "rb") as f:
    eval_data = client.files.create(file=("../data/eval/eval_data.jsonl", f))

print("Data:")
pprint(train_data)
pprint(eval_data)

# 2. Create Fine Tuning Job
created_jobs = client.jobs.create(
    model="open-mistral-7b",
    training_files=[ultrachat_chunk_train.id],
    validation_files=[ultrachat_chunk_eval.id],
    hyperparameters=TrainingParameters(
        training_steps=10,
        learning_rate=0.0001,
    ),
    integrations=[
        WandbIntegrationIn(
            project="test_ft_api",
            run_name="test",
            api_key=os.environ.get("WANDB_API_KEY"),
        ).dict()
    ],
)
print("\nCreated Jobs:")
pprint(created_jobs)

# 3. Check the Status of the Job
print("\nChecking Job Status:")
retrieved_job = client.jobs.retrieve(created_jobs.id)
while retrieved_job.status in ["RUNNING", "QUEUED"]:
    retrieved_job = client.jobs.retrieve(created_jobs.id)
    pprint(retrieved_job)
    print(f"Job is {retrieved_job.status}, waiting 10 seconds")
    time.sleep(10)

jobs = client.jobs.list()
pprint(jobs)

retrieved_jobs = client.jobs.retrieve(created_jobs.id)
pprint(retrieved_jobs)

# 4. Use the Fine Tuned Model
chat_response = client.chat(
    model=retrieved_jobs.fine_tuned_model,
    messages=[ChatMessage(role='user', content='What is gestational diabetes?')]
)
print("\nTesting Fine Tuned Model:")
pprint(chat_response)