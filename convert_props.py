import os
import re
import json
import argparse

def parse_filepath(rel_path):
    """
    Extract metadata from the relative path of the annotation file.
    Example: "english/annotations/bc/cctv/00/cctv_0001.v4_gold_prop"
    """
    parts = rel_path.split(os.sep)
    if not parts:
        return None
    
    language = parts[0]
    
    try:
        ann_idx = parts.index("annotations")
    except ValueError:
        return None
        
    if len(parts) <= ann_idx + 4:
        return None
        
    genre = parts[ann_idx + 1]
    source = parts[ann_idx + 2]
    group_id = parts[ann_idx + 3]
    filename = parts[ann_idx + 4]
    
    # Extract document_name and quality (gold or auto)
    match = re.match(r"^(.*?)\.v4_(gold|auto)_prop$", filename)
    if not match:
        return None
    doc_name, quality = match.groups()
    
    return {
        "language": language,
        "genre": genre,
        "source": source,
        "group_id": group_id,
        "document_name": doc_name,
        "quality": quality
    }

def parse_lemma_pos(lemma_pos):
    """
    Extract lemma and POS tag from roleset string.
    Example: "invite-v" -> ("invite", "v")
    """
    if "-" in lemma_pos:
        parts = lemma_pos.rsplit("-", 1)
        return parts[0], parts[1]
    return lemma_pos, None

def parse_argument(arg_str):
    """
    Parse argument strings like "2:0-rel", "3:2,6:1-ARG1", or "8:1*10:1-ARG0".
    """
    match = re.match(r"^([0-9:,*&]+)-(.*)$", arg_str)
    if match:
        node, role = match.groups()
        return {"role": role, "node": node}
    return {"role": arg_str, "node": None}

def convert_prop_file(filepath, rel_path, out_file):
    """
    Parse a single prop file and write to out_file in JSONL format.
    """
    meta = parse_filepath(rel_path)
    if not meta:
        print(f"Warning: Could not parse metadata from path: {rel_path}")
        return 0
        
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 7:
                print(f"Warning: Malformed line in {rel_path}:{line_num} -> '{line}'")
                continue
                
            doc_id_field = parts[0]
            try:
                sent_idx = int(parts[1])
                pred_idx = int(parts[2])
            except ValueError:
                print(f"Warning: Non-integer sentence/word index in {rel_path}:{line_num} -> '{line}'")
                continue
                
            annotator = parts[3]
            lemma_pos = parts[4]
            roleset_id = parts[5]
            separator = parts[6]
            
            if separator != "-----":
                print(f"Warning: Expected '-----' separator in {rel_path}:{line_num} -> '{line}'")
                # We can still proceed if the structure is correct, but let's check
                
            # Rest of the elements are argument mappings
            raw_args = parts[7:]
            arguments = []
            for arg in raw_args:
                arguments.append(parse_argument(arg))
                
            lemma, pos = parse_lemma_pos(lemma_pos)
            
            record = {
                "language": meta["language"],
                "genre": meta["genre"],
                "source": meta["source"],
                "group_id": meta["group_id"],
                "document_name": meta["document_name"],
                "quality": meta["quality"],
                "document_id_field": doc_id_field,
                "sentence_index": sent_idx,
                "predicate_word_index": pred_idx,
                "annotator": annotator,
                "lemma": lemma,
                "pos": pos,
                "roleset_id": roleset_id,
                "arguments": arguments
            }
            
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            
    return count

def main():
    parser = argparse.ArgumentParser(description="Convert CoNLL-2012 _prop files to JSONL")
    parser.add_argument("--data_dir", type=str, default="dataset/conll-2012/extracted_train/conll-2012/v4/data/train/data",
                        help="Root directory of extracted train data")
    parser.add_argument("--output_file", type=str, default="dataset/conll-2012/processed_props.jsonl",
                        help="Output JSONL filepath")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
        
    print(f"Scanning for _prop files in: {args.data_dir}")
    prop_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith("_prop"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, args.data_dir)
                prop_files.append((full_path, rel_path))
                
    print(f"Found {len(prop_files)} _prop files.")
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    total_records = 0
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for idx, (full_path, rel_path) in enumerate(prop_files, 1):
            count = convert_prop_file(full_path, rel_path, out_f)
            total_records += count
            if idx % 50 == 0 or idx == len(prop_files):
                print(f"Processed {idx}/{len(prop_files)} files... Cumulative records: {total_records}")
                
    print(f"Done! Successfully converted {total_records} propositions to {args.output_file}")

if __name__ == "__main__":
    main()
