import requests
import json

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


def fetch_and_save_content(url, index):
    reader_url = f"https://r.jina.ai/{url}"
    try:
        response = requests.get(reader_url)
        response.raise_for_status()

        content = response.text

        filename = f"content_{index + 1}.txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"Content from {url} saved to {filename}")
    except requests.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
    except IOError as e:
        print(f"Error saving content to file: {e}")


for index, url in enumerate(urls):
    fetch_and_save_content(url, index)