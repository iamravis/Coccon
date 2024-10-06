from mistralai import Mistral
import os
import time

from dotenv import load_dotenv

load_dotenv()

# Get the API keys and project names from environment variables
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
WANDB_PROJECT = os.getenv('WANDB_PROJECT')

client = Mistral(api_key=MISTRAL_API_KEY)

chunk_train = client.files.upload(file={
    "file_name": "train.jsonl",
    "content": open("../data/train/train.jsonl", "rb"),
})
chunk_eval = client.files.upload(file={
    "file_name": "eval.jsonl",
    "content": open("../data/eval/eval.jsonl", "rb"),
})

print("Data:")
print(chunk_train)
print(chunk_eval)

# Create a fine-tuning job with W&B integration and auto-start
created_jobs = client.fine_tuning.jobs.create(
    model="mistral-large-latest",
    training_files=[{"file_id": chunk_train.id, "weight": 1}],
    validation_files=[chunk_eval.id],
    hyperparameters={
        "training_steps": 10,
        "learning_rate": 0.0001
    },
    integrations=[
        {
            "project": WANDB_PROJECT,
            "api_key": WANDB_API_KEY,
        }
    ],
    auto_start=False  # We start the job manually after validation
)

# Wait for the job to be validated
job_id = created_jobs.id
job_status = None

while True:
    retrieved_job = client.fine_tuning.jobs.get(job_id=job_id)
    job_status = retrieved_job.status
    print(f"Job Status: {job_status}")
    if job_status == 'VALIDATED':
        print("Fine-tuning job validated.")
        break
    elif job_status == 'FAILED':
        print("Fine-tuning job validation failed.")
        break
    else:
        # Sleep before checking again
        time.sleep(30)  # Check every 30 seconds

# Start the fine-tuning job if validated
if job_status == 'VALIDATED':
    client.fine_tuning.jobs.start(job_id=job_id)
    print("Fine-tuning job started.")
else:
    print("Fine-tuning job cannot be started.")

# Monitor the job status
while True:
    retrieved_job = client.fine_tuning.jobs.get(job_id=job_id)
    job_status = retrieved_job.status
    print(f"Job Status: {job_status}")
    if job_status in ['SUCCESS', 'COMPLETED']:
        print("Fine-tuning job completed successfully.")
        break
    elif job_status == 'FAILED':
        print("Fine-tuning job failed.")
        break
    else:
        # Sleep before checking again
        time.sleep(60)  # Check every 60 seconds

# Retrieve the fine-tuned model ID
fine_tuned_model_id = retrieved_job.fine_tuned_model
print(f"Fine-tuned model ID: {fine_tuned_model_id}")

# Attempt to download the fine-tuned model
if job_status in ['SUCCESS', 'COMPLETED'] and fine_tuned_model_id:
    try:
        os.makedirs('finetunedmodels', exist_ok=True)

        # Download the model (replace with actual API call if available)
        # Assuming client.models.download() is the correct method
        client.models.download(
            model_id=fine_tuned_model_id,
            destination_path='finetunedmodels/'
        )

        print(f"Fine-tuned model {fine_tuned_model_id} downloaded to finetunedmodels/")
    except AttributeError:
        print("Downloading the fine-tuned model is not supported by the Mistral API.")
    except Exception as e:
        print(f"An error occurred while downloading the model: {e}")
else:
    print("Fine-tuned model not available for download.")

# Use the fine-tuned model for inference
try:
    chat_response = client.chat.complete(
        model=fine_tuned_model_id,
        messages=[{"role": 'user', "content": 'Why is iron important during pregnancy?'}]
    )

    print("Chat Response:")
    print(chat_response)
except Exception as e:
    print(f"An error occurred during inference: {e}")
