import json

def reformat_for_mistral_multilingual(input_file, output_file):
    with open(input_file, "r", encoding='utf-8') as infile, open(output_file, "w", encoding='utf-8') as outfile:
        for line in infile:
            # Load each line as a dictionary
            data = json.loads(line)

            # Initialize the messages list
            messages = []

            # Add context in both languages (if present)
            if "context_en" in data:
                messages.append({
                    "role": "system",
                    "content": data["context_en"]
                })
            if "context_hi" in data:
                messages.append({
                    "role": "system",
                    "content": data["context_hi"]
                })

            # Add the user query in both languages
            if "query_en" in data:
                messages.append({
                    "role": "user",
                    "content": data["query_en"]
                })
            if "query_hi" in data:
                messages.append({
                    "role": "user",
                    "content": data["query_hi"]
                })

            # Add the assistant answer in both languages
            if "answer_en" in data:
                messages.append({
                    "role": "assistant",
                    "content": data["answer_en"]
                })
            if "answer_hi" in data:
                messages.append({
                    "role": "assistant",
                    "content": data["answer_hi"]
                })

            # Create the reformatted structure
            reformatted_data = {
                "messages": messages
            }

            # Write reformatted data as a single line in the new JSONL file
            outfile.write(json.dumps(reformatted_data, ensure_ascii=False) + "\n")


# Reformat both train and eval files for English and Hindi
reformat_for_mistral_multilingual("../data/finetune_data/multilingual_finetuning_dataset_test.jsonl", "../data/train/reformatted_multilingual.jsonl")
#reformat_for_mistral_multilingual("../data/eval/eval.jsonl", "../data/eval/eval_reformatted_multilingual.jsonl")

print("Reformatting completed for both languages!")
