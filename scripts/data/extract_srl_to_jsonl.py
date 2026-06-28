import os
import re
import json
from tqdm import tqdm

class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []
        self.parent = None
        self.start_idx = None
        self.end_idx = None
        
    def is_leaf(self):
        return len(self.children) == 1 and isinstance(self.children[0], str)

def parse_sexpr(s):
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    tokens = s.split()
    
    stack = []
    for token in tokens:
        if token == '(':
            stack.append([])
        elif token == ')':
            if not stack:
                raise ValueError("Unbalanced parentheses")
            curr = stack.pop()
            label = curr[0]
            children = curr[1:]
            node = Node(label, children)
            for child in children:
                if isinstance(child, Node):
                    child.parent = node
            if stack:
                stack[-1].append(node)
            else:
                return node
        else:
            stack[-1].append(token)
    raise ValueError("Incomplete S-expression")

def assign_indices(node, start_idx=0):
    if node.is_leaf():
        node.start_idx = start_idx
        node.end_idx = start_idx + 1
        return start_idx + 1
    
    curr_idx = start_idx
    for child in node.children:
        if isinstance(child, Node):
            curr_idx = assign_indices(child, curr_idx)
    node.start_idx = start_idx
    node.end_idx = curr_idx
    return curr_idx

def get_leaves(node):
    leaves = []
    def collect(n):
        if n.is_leaf():
            leaves.append(n)
        else:
            for child in n.children:
                if isinstance(child, Node):
                    collect(child)
    collect(node)
    return leaves

def read_trees(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    trees = []
    curr = []
    balance = 0
    in_tree = False
    
    i = 0
    n = len(content)
    while i < n:
        c = content[i]
        if c == '(':
            if balance == 0:
                in_tree = True
                curr = []
            balance += 1
            curr.append(c)
        elif c == ')':
            if in_tree:
                balance -= 1
                curr.append(c)
                if balance == 0:
                    trees.append("".join(curr))
                    in_tree = False
        else:
            if in_tree:
                curr.append(c)
        i += 1
    return trees

def process_document(parse_file, prop_file, doc_id_rel):
    try:
        trees_str = read_trees(parse_file)
    except Exception as e:
        print(f"Error reading parse file {parse_file}: {e}")
        return []
    
    trees = []
    for i, t_str in enumerate(trees_str):
        try:
            tree = parse_sexpr(t_str)
            assign_indices(tree)
            trees.append(tree)
        except Exception as e:
            trees.append(None)
            print(f"Error parsing tree {i} in {parse_file}: {e}")

    propositions = {}
    if os.path.exists(prop_file):
        try:
            with open(prop_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    sent_idx = int(parts[1])
                    pred_idx = int(parts[2])
                    lemma = parts[4]
                    roleset = parts[5]
                    args = parts[7:]
                    
                    if sent_idx not in propositions:
                        propositions[sent_idx] = []
                    propositions[sent_idx].append({
                        "pred_idx": pred_idx,
                        "lemma": lemma,
                        "roleset": roleset,
                        "args": args
                    })
        except Exception as e:
            print(f"Error reading prop file {prop_file}: {e}")

    documents_sentences = []
    
    for sent_idx, tree in enumerate(trees):
        if tree is None:
            continue
            
        leaves = get_leaves(tree)
        full_words = [l.children[0] for l in leaves]
        full_pos_tags = [l.label for l in leaves]
        
        clean_words = []
        full_to_clean = {}
        for i, (word, pos) in enumerate(zip(full_words, full_pos_tags)):
            if pos == '-NONE-':
                full_to_clean[i] = None
            else:
                full_to_clean[i] = len(clean_words)
                clean_words.append(word)
                
        def map_span(start, end):
            clean_start = None
            clean_end = None
            for k in range(start, end):
                clean_idx = full_to_clean.get(k)
                if clean_idx is not None:
                    if clean_start is None:
                        clean_start = clean_idx
                    clean_end = clean_idx + 1
            return clean_start, clean_end

        verbs_list = []
        sent_props = propositions.get(sent_idx, [])
        for prop in sent_props:
            pred_idx = prop["pred_idx"]
            roleset = prop["roleset"]
            lemma = prop["lemma"]
            
            clean_pred_idx = full_to_clean.get(pred_idx)
            
            arguments = []
            for arg in prop["args"]:
                if '-' not in arg:
                    continue
                addr, label = arg.split('-', 1)
                
                addrs = re.split(r'[*&,]', addr)
                full_spans = []
                clean_spans = []
                for a in addrs:
                    if not a or ':' not in a:
                        continue
                    try:
                        leaf_idx, height = map(int, a.split(':'))
                        leaf_node = leaves[leaf_idx]
                        curr = leaf_node
                        for _ in range(height):
                            if curr.parent:
                                curr = curr.parent
                        
                        full_start, full_end = curr.start_idx, curr.end_idx
                        full_spans.append([full_start, full_end])
                        
                        clean_start, clean_end = map_span(full_start, full_end)
                        if clean_start is not None and clean_end is not None:
                            clean_spans.append([clean_start, clean_end])
                    except Exception as e:
                        pass
                
                arg_text_parts = []
                for cs in clean_spans:
                    arg_text_parts.append(" ".join(clean_words[cs[0]:cs[1]]))
                arg_text = " ".join(arg_text_parts)
                
                arguments.append({
                    "label": label,
                    "full_spans": full_spans,
                    "clean_spans": clean_spans,
                    "text": arg_text
                })
                
            verbs_list.append({
                "verb": clean_words[clean_pred_idx] if clean_pred_idx is not None else full_words[pred_idx],
                "clean_index": clean_pred_idx,
                "full_index": pred_idx,
                "roleset": roleset,
                "arguments": arguments
            })
            
        documents_sentences.append({
            "doc_id": doc_id_rel,
            "sentence_id": sent_idx,
            "words": clean_words,
            "pos_tags": [pos for pos in full_pos_tags if pos != '-NONE-'],
            "full_words": full_words,
            "full_pos_tags": full_pos_tags,
            "verbs": verbs_list
        })
        
    return documents_sentences

def main():
    root_dir = "/teamspace/studios/this_studio/semantic-role-labeling/scripts/notebooks/my_ontonotes_folder/ontonotes-release-5.0/data/files/data/english/annotations"
    output_file = "/teamspace/studios/this_studio/semantic-role-labeling/english_srl.jsonl"
    
    print("Finding all matching .prop and .parse files...")
    file_groups = {}
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.prop'):
                base = file[:-5]
                prop_path = os.path.join(root, file)
                parse_path = os.path.join(root, base + '.parse')
                if os.path.exists(parse_path):
                    rel_dir = os.path.relpath(root, root_dir)
                    doc_id_rel = os.path.join(rel_dir, base)
                    file_groups[doc_id_rel] = (parse_path, prop_path)
                    
    print(f"Found {len(file_groups)} documents to process.")
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for doc_id_rel, (parse_path, prop_path) in tqdm(sorted(file_groups.items()), desc="Processing docs"):
            sentences = process_document(parse_path, prop_path, doc_id_rel)
            for sent in sentences:
                out_f.write(json.dumps(sent, ensure_ascii=False) + '\n')
                count += 1
                
    print(f"\nSuccessfully extracted {count} sentences to {output_file}")

if __name__ == "__main__":
    main()
