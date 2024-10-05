import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
API_KEY = os.getenv('JINA_AI_READER_API_KEY')

urls = [
    "https://www.nhs.uk/pregnancy/keeping-well/have-a-healthy-diet/",
    "https://www.nhs.uk/pregnancy/keeping-well/vegetarian-or-vegan-and-pregnant/",
    "https://www.nhs.uk/pregnancy/keeping-well/foods-to-avoid/",
    "https://www.nhs.uk/pregnancy/keeping-well/vitamins-supplements-and-nutrition/",
    "https://www.nhs.uk/pregnancy/keeping-well/exercise/",
    "https://www.nhs.uk/pregnancy/keeping-well/medicines/",
    "https://www.nhs.uk/pregnancy/keeping-well/mental-health/",
    "https://www.nhs.uk/pregnancy/keeping-well/depression/",
    "https://www.nhs.uk/pregnancy/keeping-well/sex/",
    "https://www.nhs.uk/pregnancy/keeping-well/travelling/",
    "https://www.nhs.uk/pregnancy/keeping-well/your-health-at-work/",
    "https://www.nhs.uk/pregnancy/keeping-well/your-babys-movements/",
    "https://www.nhs.uk/pregnancy/keeping-well/reducing-the-risk-of-stillbirth/",
    "https://www.nhs.uk/pregnancy/keeping-well/drinking-alcohol-while-pregnant/",
    "https://www.nhs.uk/pregnancy/keeping-well/stop-smoking/",
    "https://www.nhs.uk/pregnancy/keeping-well/illegal-drugs/",
    "https://www.nhs.uk/pregnancy/keeping-well/vaccinations/",
    "https://www.nhs.uk/pregnancy/keeping-well/flu-jab/",
    "https://www.nhs.uk/pregnancy/keeping-well/whooping-cough-vaccination/",
    "https://www.nhs.uk/pregnancy/keeping-well/infections-that-may-affect-your-baby/",
    "https://www.nhs.uk/pregnancy/keeping-well/pregnancy-and-covid-19/"
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
os.makedirs('../data/rag_data', exist_ok=True)

# File to store all content
output_file = '../data/rag_data/all_content.txt'

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