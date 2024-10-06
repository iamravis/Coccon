import os
import json
import time
import wandb
from rich import print
from mistralai.client import MistralClient
from dotenv import load_dotenv
from mistralai.models.jobs import TrainingParameters
from mistralai.models.chat_completion import ChatMessage
from mistralai.models.jobs import WandbIntegrationIn

load_dotenv()

# Get the API key from environment variable
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
WANDB_API_KEY = os.getenv('WANDB_API_KEY')


client = MistralClient(api_key=MISTRAL_API_KEY)

def pprint(obj):
    print(json.dumps(obj.dict(), indent=4))

# 1. Upload the dataset
def upload_file(file_path, purpose="fine-tune"):
    with open(file_path, "rb") as f:
        response = client.files.create(file=(os.path.basename(file_path), f))
    return response

ultrachat_chunk_train = upload_file("../data/train/train.jsonl")
ultrachat_chunk_eval = upload_file("./data/eval/eval.jsonl")

print("Data:")
pprint(ultrachat_chunk_train)
pprint(ultrachat_chunk_eval)

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
            api_key=WANDB_API_KEY,
        ).dict()
    ],
)
print("\nCreated Jobs:")
pprint(created_jobs)

# 3. Check the Status of the Job
def monitor_job(job_id):
    while True:
        retrieved_job = client.jobs.retrieve(job_id)
        print(f"\nJob Status: {retrieved_job.status}")
        pprint(retrieved_job)
        if retrieved_job.status not in ["RUNNING", "QUEUED"]:
            return retrieved_job
        print("Waiting 60 seconds before next check...")
        time.sleep(60)

print("\nMonitoring Job Status:")
final_job = monitor_job(created_jobs.id)

print("\nListing all jobs:")
jobs = client.jobs.list()
pprint(jobs)

print("\nRetrieving final job details:")
pprint(final_job)

# 4. Use the Fine Tuned Model
if final_job.status == "SUCCEEDED":
    chat_response = client.chat(
        model=final_job.fine_tuned_model,
        messages=[ChatMessage(role='user', content='What is the best French cheese?')]
    )
    print("\nTesting Fine Tuned Model:")
    pprint(chat_response)
else:
    print(f"\nFine-tuning job did not succeed. Final status: {final_job.status}")
