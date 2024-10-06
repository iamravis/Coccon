import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
API_KEY = os.getenv('JINA_AI_READER_API_KEY')

urls = [
    "https://www.nhs.uk/conditions/pre-eclampsia/",
    "https://www.nhs.uk/conditions/gestational-diabetes/",
    "https://www.nhs.uk/conditions/baby/",
    "https://www.nhs.uk/conditions/caesarean-section/"
    
]


def fetch_content(url):
    # Construct the reader URL
    reader_url = f"https://r.jina.ai/{url}"

    try:
        response = requests.get(reader_url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return ""


# Create the directory if it doesn't exist
os.makedirs('../data/finetune_data', exist_ok=True)

# File to store all content
output_file = '../data/finetune_data/all_content_7.txt'

with open(output_file, 'w', encoding='utf-8') as file:
    for url in urls:
        print(f"Fetching content from {url}")
        content = fetch_content(url)
        if content:
            file.write(f"Content from {url}:\n\n")
            file.write(content)
            file.write("\n\n" + "=" * 50 + "\n\n")  # Separator between contents
        print(f"Content from {url} added to {output_file}")

print(f"All content has been saved to {output_file}")