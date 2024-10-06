import pandas as pd
import json

# Load your existing multilingual JSONL file
df = pd.read_json("../data/finetune_data/multilingual.jsonl", lines=True)

# Use a more balanced split between train and eval
df_train = df.sample(frac=0.9, random_state=200)  # 90% for training
df_eval = df.drop(df_train.index)  # Remaining 10% for evaluation

# Function to save JSONL in a compact format (no pretty-printing)
def save_jsonl(df, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in df.to_dict(orient="records"):
            json.dump(record, f, ensure_ascii=False)  # Compact format, no indent
            f.write('\n')  # Ensure each record is on a new line

# Save the splits into JSONL files
save_jsonl(df_train, "../data/train/train.jsonl")
save_jsonl(df_eval, "../data/eval/eval.jsonl")

print("Data split and saved successfully!")
