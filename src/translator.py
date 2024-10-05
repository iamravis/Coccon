import json
import googletrans
from tqdm import tqdm


def translate_to_hindi(text):
    translator = googletrans.Translator()
    try:
        return translator.translate(text, dest='hi').text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def process_dataset(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in tqdm(data, desc="Translating"):
        # Translate context
        item['context_hindi'] = translate_to_hindi(item['context'])

        # Translate query
        item['query_hindi'] = translate_to_hindi(item['query'])

        # Translate answer
        item['answer_hindi'] = translate_to_hindi(item['answer'])

    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Processed dataset saved to {input_file}")


# Usage
input_file = 'rag_varied_fine_tuning_dataset-1.json'
process_dataset(input_file)