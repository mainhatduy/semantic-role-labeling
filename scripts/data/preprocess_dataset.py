import os
import json
import random
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

load_dotenv()
token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN not found in environment or .env file.")

# Define head selection rules from inference.ipynb
def find_head_index(span, pos_tags, words):
    start, end = span
    span_len = end - start
    if span_len <= 1:
        return start
    
    sub_pos = pos_tags[start:end]
    sub_words = words[start:end]
    
    # Rule 1: Split by prepositions, conjunctions, subordinators, and infinitival 'to'
    split_indices = [i for i, pos in enumerate(sub_pos) if pos in ('IN', 'TO', 'CC', 'WDT', 'WP', 'WRB')]
    if split_indices:
        limit = split_indices[0]
        if limit > 0:
            sub_pos = sub_pos[:limit]
            sub_words = sub_words[:limit]
            span_len = limit
        else:
            # Skip the leading preposition and recurse on the rest
            sub_pos = sub_pos[1:]
            sub_words = sub_words[1:]
            start += 1
            span_len -= 1
            return find_head_index((start, start + span_len), pos_tags, words)
            
    # Rule 2: Find the last noun or pronoun in the remaining span
    noun_indices = [i for i, pos in enumerate(sub_pos) if pos.startswith('NN') or pos == 'PRP']
    if noun_indices:
        return start + noun_indices[-1]
        
    # Rule 3: Find the last verb
    verb_indices = [i for i, pos in enumerate(sub_pos) if pos.startswith('VB')]
    if verb_indices:
        return start + verb_indices[-1]
        
    # Rule 4: Find the last adjective
    adj_indices = [i for i, pos in enumerate(sub_pos) if pos.startswith('JJ')]
    if adj_indices:
        return start + adj_indices[-1]
        
    return start + span_len - 1

def main():
    print("Loading dataset kiil-lab/english_srl...")
    dataset = load_dataset("kiil-lab/english_srl", token=token)
    train_data = list(dataset["train"])
    print(f"Loaded {len(train_data)} samples.")

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(train_data)

    # Split: 80% train, 10% val, 10% test
    n_total = len(train_data)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    splits = {
        "train": train_data[:n_train],
        "val": train_data[n_train:n_train + n_val],
        "test": train_data[n_train + n_val:]
    }

    # Output directory
    out_dir = "/teamspace/studios/this_studio/semantic-role-labeling/dataset/propbank"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Extract unique roles (excluding 'rel')
    unique_roles = set()

    for split_name, samples in splits.items():
        for sample in samples:
            for verb in sample.get("verbs", []):
                for arg in verb.get("arguments", []):
                    label = arg["label"]
                    if label != "rel":
                        unique_roles.add(label)

    # Sort roles: ARGs first (ARG0, ARG1, ...) then others alphabetically
    sorted_roles = sorted(
        list(unique_roles),
        key=lambda r: (0 if r.startswith('ARG') and r[3:].isdigit() else 1, r)
    )

    roles_file_path = os.path.join(out_dir, "unique_roles.json")
    with open(roles_file_path, "w", encoding="utf-8") as f:
        json.dump(sorted_roles, f, indent=2)
    print(f"Wrote {len(sorted_roles)} unique roles to {roles_file_path}")

    # 2. Process and write JSONL files
    for split_name, samples in splits.items():
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        print(f"Processing split '{split_name}' -> {out_path}...")
        
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in tqdm(samples):
                words = sample["words"]
                pos_tags = sample["pos_tags"]
                n_words = len(words)
                if n_words == 0:
                    continue

                predicates = []
                for verb_info in sample.get("verbs", []):
                    pred_idx = verb_info["clean_index"]
                    if pred_idx is None or pred_idx >= n_words:
                        continue

                    argument_spans = []
                    for arg in verb_info.get("arguments", []):
                        role = arg["label"]
                        if role == "rel":
                            continue  # skip relation to self
                        
                        spans = arg.get("clean_spans", [])
                        for span in spans:
                            start, end = span
                            for token_index in range(start, end):
                                if 0 <= token_index < n_words:
                                    argument_spans.append({
                                        "token_index": token_index,
                                        "role": role
                                    })

                    predicates.append({
                        "predicate_index": pred_idx,
                        "argument_spans": argument_spans
                    })

                record = {
                    "words": words,
                    "pos_tags": pos_tags,
                    "predicates": predicates
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
