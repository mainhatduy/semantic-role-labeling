"""
SRL Graph Dataset — PropBank JSONL → PyG Data objects.

Each sentence produces exactly ONE Data object (1 sentence = 1 graph):
    - data.x: XLM-RoBERTa embeddings (N, 768) — continuous, frozen
    - data.edge_index: sparse directed edges (2, num_edges) — ALL predicate→arg edges
    - data.edge_attr: one-hot edge labels (num_edges, num_edge_classes)
    - data.y: empty global features (1, 0)
    - data.predicate_mask: multi-hot boolean mask (N,) — True for every predicate word
      E[i][j] = role_label  if word[i] is a predicate and word[j] is its argument
      E[i][j] = 0 (No-Edge) otherwise

Joint fine-tuning mode (SRLGraphDatasetWithTokens):
    - data.input_ids: token ids for online re-embedding (1, seq_len)
    - data.word_starts / data.word_ends: subword → word span boundaries (N, max_subwords)
    The embedding model is re-run every forward pass, enabling joint optimization.
"""

import os
import json
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.lightning import LightningDataset
from tqdm import tqdm

from src.diffusion.distributions import DistributionNodes


class SRLGraphDataset(InMemoryDataset):
    """Converts PropBank JSONL into PyG graph Data objects for edge-only diffusion."""

    def __init__(self, root, jsonl_path, roles_path, max_seq_len=64,
                 embedding_model='FacebookAI/xlm-roberta-base', embedding_dim=768,
                 split='train', transform=None, pre_transform=None):
        self.jsonl_path = jsonl_path
        self.roles_path = roles_path
        self.max_seq_len = max_seq_len
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.split = split

        # Load role mapping
        with open(roles_path, 'r') as f:
            roles = json.load(f)
        self.role_to_idx = {"No-Edge": 0}
        for i, role in enumerate(roles):
            self.role_to_idx[role] = i + 1
        self.num_edge_classes = len(self.role_to_idx)  # 58 (0=No-Edge + 57 roles)

        # Reuse existing processed train/full files if available to avoid re-generating
        if self.split == 'train':
            processed_dir = os.path.join(root, 'processed')
            train_pt = os.path.join(processed_dir, 'srl_train.pt')
            full_pt = os.path.join(processed_dir, 'srl_full.pt')
            if not os.path.exists(train_pt) and os.path.exists(full_pt):
                print(f"Found existing srl_full.pt, creating a link to srl_train.pt to avoid reprocessing...")
                os.makedirs(processed_dir, exist_ok=True)
                try:
                    os.link(full_pt, train_pt)
                except Exception:
                    import shutil
                    shutil.copy(full_pt, train_pt)

            cache_train = os.path.join(root, 'embeddings_cache_train.pt')
            cache_full = os.path.join(root, 'embeddings_cache.pt')
            if not os.path.exists(cache_train) and os.path.exists(cache_full):
                print(f"Found existing embeddings_cache.pt, creating a link to embeddings_cache_train.pt...")
                try:
                    os.link(cache_full, cache_train)
                except Exception:
                    import shutil
                    shutil.copy(cache_full, cache_train)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        # Note: Keeping dataset tensors on CPU memory to prevent slow GPU slicing and collation bottlenecks.
        # if torch.cuda.is_available():
        #     print(f"[{self.split.capitalize()} Dataset] Loading all data tensors directly to GPU VRAM...")
        #     self._data = self._data.to('cuda')

    @property
    def processed_file_names(self):
        return [f'srl_{self.split}.pt']

    def process(self):
        """Parse JSONL, compute embeddings, build PyG Data objects."""
        print(f"Processing SRL dataset from {self.jsonl_path}...")

        # Load sentences
        sentences = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                sentences.append(json.loads(line.strip()))

        # Compute or load cached embeddings
        cache_name = 'embeddings_cache.pt' if self.split == 'full' else f'embeddings_cache_{self.split}.pt'
        embeddings_cache = os.path.join(self.root, cache_name)
        if os.path.exists(embeddings_cache):
            print(f"Loading cached embeddings from {embeddings_cache}")
            all_embeddings = torch.load(embeddings_cache, weights_only=True)
        else:
            print(f"Computing embeddings with {self.embedding_model}...")
            all_embeddings = self._compute_embeddings(sentences)
            torch.save(all_embeddings, embeddings_cache)
            print(f"Cached embeddings to {embeddings_cache}")

        # Build Data objects — 1 sentence = 1 graph
        data_list = []
        skipped = 0

        for sent_idx, sentence in enumerate(tqdm(sentences, desc="Building graphs")):
            words = sentence['words']
            n_words = len(words)

            # Skip sentences longer than max_seq_len
            if n_words > self.max_seq_len:
                skipped += 1
                continue

            # Get embeddings for this sentence (N, 768)
            X = all_embeddings[sent_idx]
            if X is None:
                skipped += 1
                continue
            # Ensure X has the correct number of words
            X = X[:n_words]
            if X.shape[0] != n_words:
                skipped += 1
                continue

            # Collect ALL edges from ALL predicates into one edge list.
            # E[i][j] = role_label  if word[i] is a predicate and word[j] is its argument
            # E[i][j] = 0 (No-Edge) otherwise — no collisions since each predicate is a unique row.
            edge_index_list = []
            edge_attr_list = []
            predicate_mask = torch.zeros(n_words, dtype=torch.bool)  # multi-hot

            for pred_info in sentence.get('predicates', []):
                pred_idx = pred_info['predicate_index']
                if pred_idx >= n_words:
                    continue

                predicate_mask[pred_idx] = True  # mark every predicate word

                for arg in pred_info.get('argument_spans', []):
                    arg_idx = arg['token_index']
                    role = arg['role']

                    if arg_idx >= n_words:
                        continue
                    if role not in self.role_to_idx:
                        continue

                    role_idx = self.role_to_idx[role]
                    edge_index_list.append([pred_idx, arg_idx])
                    edge_attr_list.append(role_idx)


            # Build sparse edge representation
            if len(edge_index_list) > 0:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).T  # (2, E)
                edge_attr = F.one_hot(
                    torch.tensor(edge_attr_list, dtype=torch.long),
                    num_classes=self.num_edge_classes
                ).float()  # (E, num_edge_classes)
            else:
                # Predicates exist but none have arguments (e.g. "be" with no args)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, self.num_edge_classes), dtype=torch.float)

            # Empty global features
            y = torch.zeros(1, 0)

            data = Data(
                x=X.clone(),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                predicate_mask=predicate_mask,  # multi-hot: True for each predicate word
            )
            data_list.append(data)

        print(f"Built {len(data_list)} graph samples ({skipped} sentences skipped)")

        torch.save(self.collate(data_list), self.processed_paths[0])

    def _compute_embeddings(self, sentences):
        """Compute XLM-RoBERTa word-level embeddings with subword averaging."""
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        model = AutoModel.from_pretrained(self.embedding_model)
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        all_embeddings = []

        with torch.no_grad():
            for sentence in tqdm(sentences, desc="Computing embeddings"):
                words = sentence['words']
                n_words = len(words)

                if n_words > self.max_seq_len:
                    all_embeddings.append(None)
                    continue

                # Tokenize each word separately to track subword → word mapping
                word_ids_to_subword_ids = []
                all_subwords = [tokenizer.cls_token_id]

                for word_idx, word in enumerate(words):
                    subword_ids = tokenizer.encode(word, add_special_tokens=False)
                    word_ids_to_subword_ids.append(
                        list(range(len(all_subwords), len(all_subwords) + len(subword_ids)))
                    )
                    all_subwords.extend(subword_ids)

                all_subwords.append(tokenizer.sep_token_id)

                # Truncate if too long for the model
                max_model_len = tokenizer.model_max_length
                if len(all_subwords) > max_model_len:
                    all_subwords = all_subwords[:max_model_len - 1] + [tokenizer.sep_token_id]

                input_ids = torch.tensor([all_subwords], device=device)
                attention_mask = torch.ones_like(input_ids)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state[0]  # (seq_len, 768)

                # Average subword embeddings per word
                word_embeddings = []
                for word_idx in range(n_words):
                    subword_indices = word_ids_to_subword_ids[word_idx]
                    # Filter indices that are within bounds
                    valid_indices = [i for i in subword_indices if i < hidden_states.shape[0]]
                    if len(valid_indices) > 0:
                        word_emb = hidden_states[valid_indices].mean(dim=0)
                    else:
                        word_emb = torch.zeros(self.embedding_dim, device=device)
                    word_embeddings.append(word_emb)

                word_embeddings = torch.stack(word_embeddings).cpu()  # (N, 768)
                all_embeddings.append(word_embeddings)

        return all_embeddings


# ---------------------------------------------------------------------------
# Joint fine-tuning dataset: stores token ids + word boundaries for online
# re-embedding. Works as a wrapper over the processed Data objects.
# ---------------------------------------------------------------------------

class SRLGraphDatasetWithTokens(SRLGraphDataset):
    """
    Extends SRLGraphDataset by also saving tokenizer outputs (input_ids,
    attention_mask, and word-boundary spans) into every Data object.

    When joint_finetune=True, the embedding model recomputes embeddings
    from these ids every forward pass so that gradients flow into it.
    The cached `data.x` is kept as a fallback for stage-1 (frozen) mode.
    """

    # Override processed file names so we write a separate .pt file
    @property
    def processed_file_names(self):
        return [f'srl_{self.split}_with_tokens.pt']

    def process(self):
        """Parse JSONL, build PyG Data objects that carry both cached
        embeddings AND token ids for online re-embedding."""
        from transformers import AutoTokenizer

        print(f"Processing SRL dataset (with tokens) from {self.jsonl_path}...")

        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)

        # Load sentences
        sentences = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                sentences.append(json.loads(line.strip()))

        # Compute or load cached embeddings (same logic as parent)
        cache_name = 'embeddings_cache.pt' if self.split == 'full' else f'embeddings_cache_{self.split}.pt'
        embeddings_cache = os.path.join(self.root, cache_name)
        if os.path.exists(embeddings_cache):
            print(f"Loading cached embeddings from {embeddings_cache}")
            all_embeddings = torch.load(embeddings_cache, weights_only=True)
        else:
            print(f"Computing embeddings with {self.embedding_model}...")
            all_embeddings = self._compute_embeddings(sentences)
            torch.save(all_embeddings, embeddings_cache)
            print(f"Cached embeddings to {embeddings_cache}")

        data_list = []
        skipped = 0
        max_model_len = tokenizer.model_max_length

        for sent_idx, sentence in enumerate(tqdm(sentences, desc="Building graphs (with tokens)")):
            words = sentence['words']
            n_words = len(words)

            if n_words > self.max_seq_len:
                skipped += 1
                continue

            X = all_embeddings[sent_idx]
            if X is None:
                skipped += 1
                continue
            X = X[:n_words]
            if X.shape[0] != n_words:
                skipped += 1
                continue

            # ------------------------------------------------------------------
            # Tokenize and build word-boundary info for online re-embedding
            # ------------------------------------------------------------------
            word_ids_to_subword_ids = []
            all_subwords = [tokenizer.cls_token_id]

            for word in words:
                subword_ids = tokenizer.encode(word, add_special_tokens=False)
                word_ids_to_subword_ids.append(
                    list(range(len(all_subwords), len(all_subwords) + len(subword_ids)))
                )
                all_subwords.extend(subword_ids)

            all_subwords.append(tokenizer.sep_token_id)

            if len(all_subwords) > max_model_len:
                all_subwords = all_subwords[:max_model_len - 1] + [tokenizer.sep_token_id]

            seq_len = len(all_subwords)

            # Pad input_ids to max_model_len so that all Data objects have the
            # same-length tensor → PyG batch collation gives a regular 2D block.
            pad_len = max_model_len - seq_len
            pad_id  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            padded_input_ids = all_subwords + [pad_id] * pad_len
            padded_attn_mask = [1] * seq_len + [0] * pad_len

            input_ids      = torch.tensor(padded_input_ids, dtype=torch.long)  # (max_model_len,)
            attention_mask = torch.tensor(padded_attn_mask, dtype=torch.long)  # (max_model_len,)

            # word_spans: (N, 2) — [start, end) subword indices per word
            # We store start/end as two tensors so PyG collation works fine.
            word_starts = torch.zeros(n_words, dtype=torch.long)
            word_ends   = torch.zeros(n_words, dtype=torch.long)
            for wi, subword_ids_list in enumerate(word_ids_to_subword_ids):
                valid = [i for i in subword_ids_list if i < seq_len]
                if valid:
                    word_starts[wi] = valid[0]
                    word_ends[wi]   = valid[-1] + 1   # exclusive end
                else:
                    # Fallback: subwords were truncated — mark as invalid (0, 0)
                    word_starts[wi] = 0
                    word_ends[wi]   = 0

            # ------------------------------------------------------------------
            # Build edges (identical to parent class)
            # ------------------------------------------------------------------
            edge_index_list = []
            edge_attr_list = []
            predicate_mask = torch.zeros(n_words, dtype=torch.bool)

            for pred_info in sentence.get('predicates', []):
                pred_idx = pred_info['predicate_index']
                if pred_idx >= n_words:
                    continue
                predicate_mask[pred_idx] = True

                for arg in pred_info.get('argument_spans', []):
                    arg_idx = arg['token_index']
                    role = arg['role']
                    if arg_idx >= n_words:
                        continue
                    if role not in self.role_to_idx:
                        continue
                    role_idx = self.role_to_idx[role]
                    edge_index_list.append([pred_idx, arg_idx])
                    edge_attr_list.append(role_idx)

            if len(edge_index_list) > 0:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).T
                edge_attr = F.one_hot(
                    torch.tensor(edge_attr_list, dtype=torch.long),
                    num_classes=self.num_edge_classes
                ).float()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, self.num_edge_classes), dtype=torch.float)

            y = torch.zeros(1, 0)

            data = Data(
                x=X.clone(),                   # cached embeddings (N, 768) — stage-1 fallback
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                predicate_mask=predicate_mask,
                # --- online-embedding extras ---
                input_ids=input_ids,           # (seq_len,)  — token ids
                attention_mask=attention_mask, # (seq_len,)  — attention mask
                word_starts=word_starts,       # (N,)        — first subword idx per word
                word_ends=word_ends,           # (N,)        — last+1 subword idx per word
            )
            data_list.append(data)

        print(f"Built {len(data_list)} graph samples with tokens ({skipped} sentences skipped)")
        torch.save(self.collate(data_list), self.processed_paths[0])


class SRLDataModule(LightningDataset):
    """DataModule that loads PropBank SRL data with train/val/test splits.

    When ``cfg.train.joint_finetune_epoch`` is set, the datamodule uses
    ``SRLGraphDatasetWithTokens`` so that the embedding model can be
    fine-tuned jointly with the GraphTransformer in stage 2.
    """

    def __init__(self, cfg):
        dataset_cfg = cfg.dataset

        # Resolve paths relative to the project root
        project_root = Path(__file__).resolve().parents[2]  # models/MySRLGraph/ level
        data_dir = project_root / dataset_cfg.datadir
        roles_path = str(data_dir / dataset_cfg.roles_file)

        processed_root = str(project_root / 'processed_data')

        max_seq_len = getattr(dataset_cfg, 'max_seq_len', 64)
        embedding_model = getattr(dataset_cfg, 'embedding_model', 'FacebookAI/xlm-roberta-base')
        embedding_dim = getattr(dataset_cfg, 'embedding_dim', 768)

        # Use token-storing dataset if joint fine-tuning is requested
        joint_finetune_epoch = getattr(cfg.train, 'joint_finetune_epoch', None)
        DatasetClass = SRLGraphDatasetWithTokens if joint_finetune_epoch is not None else SRLGraphDataset
        if joint_finetune_epoch is not None:
            print(f"[Joint Fine-tuning] Using SRLGraphDatasetWithTokens — "
                  f"embedding model will be unfrozen at epoch {joint_finetune_epoch}.")

        # Check if val_file and test_file are specified or if they exist in dataset directory
        val_file = getattr(dataset_cfg, 'val_file', None)
        if val_file is None and (data_dir / 'val.jsonl').exists():
            val_file = 'val.jsonl'
        test_file = getattr(dataset_cfg, 'test_file', None)
        if test_file is None and (data_dir / 'test.jsonl').exists():
            test_file = 'test.jsonl'

        if val_file is not None:
            print(f"Loading separate datasets: train={dataset_cfg.train_file}, val={val_file}, test={test_file}")
            train_dataset = DatasetClass(
                root=processed_root,
                jsonl_path=str(data_dir / dataset_cfg.train_file),
                roles_path=roles_path,
                max_seq_len=max_seq_len,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                split='train',
            )
            val_dataset = DatasetClass(
                root=processed_root,
                jsonl_path=str(data_dir / val_file),
                roles_path=roles_path,
                max_seq_len=max_seq_len,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                split='val',
            )
            if test_file is not None:
                test_dataset = DatasetClass(
                    root=processed_root,
                    jsonl_path=str(data_dir / test_file),
                    roles_path=roles_path,
                    max_seq_len=max_seq_len,
                    embedding_model=embedding_model,
                    embedding_dim=embedding_dim,
                    split='test',
                )
            else:
                test_dataset = val_dataset

            self.full_dataset = train_dataset
            self.num_edge_classes = train_dataset.num_edge_classes
        else:
            print("No separate val_file found. Splitting train_file randomly.")
            jsonl_path = str(data_dir / dataset_cfg.train_file)
            full_dataset = DatasetClass(
                root=processed_root,
                jsonl_path=jsonl_path,
                roles_path=roles_path,
                max_seq_len=max_seq_len,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                split='full',
            )

            # Split into train/val/test
            split_ratios = getattr(dataset_cfg, 'split_ratios', [0.8, 0.1, 0.1])
            total = len(full_dataset)
            train_size = int(total * split_ratios[0])
            val_size = int(total * split_ratios[1])
            test_size = total - train_size - val_size

            # Use a fixed generator for reproducibility
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size, test_size], generator=generator
            )

            self.full_dataset = full_dataset
            self.num_edge_classes = full_dataset.num_edge_classes

        self.cfg = cfg

        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=cfg.train.batch_size if 'debug' not in cfg.general.name else 2,
            num_workers=cfg.train.num_workers,
            pin_memory=getattr(cfg.dataset, 'pin_memory', True) if torch.cuda.is_available() else False,
        )

    def node_counts(self, max_nodes_possible=300):
        """Compute distribution of number of nodes (words) per sample."""
        all_counts = torch.zeros(max_nodes_possible)
        for data in self.full_dataset:
            n = data.x.shape[0]
            if n < max_nodes_possible:
                all_counts[n] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        """For continuous embeddings, return a dummy single-class distribution."""
        return torch.ones(1)

    def edge_counts(self):
        """Compute distribution of edge types across all training data."""
        num_classes = self.num_edge_classes
        device = self.full_dataset[0].edge_attr.device if len(self.full_dataset) > 0 else 'cpu'
        d = torch.zeros(num_classes, dtype=torch.float, device=device)

        for data in self.full_dataset:
            n = data.x.shape[0]
            num_edges = data.edge_index.shape[1]
            # Count non-edges: all possible directed pairs minus actual edges
            all_pairs = n * (n - 1)  # directed: no self-loops
            num_non_edges = all_pairs - num_edges

            if num_edges > 0:
                edge_types = data.edge_attr.sum(dim=0)
                d[0] += num_non_edges
                d[1:] += edge_types[1:]
            else:
                d[0] += all_pairs

        d = d / d.sum()
        return d.cpu()


class SRLDatasetInfos:
    """Dataset metadata for SRL graphs."""

    def __init__(self, datamodule, cfg):
        self.datamodule = datamodule
        self.cfg = cfg

        self.num_edge_classes = datamodule.num_edge_classes

        # Compute distributions
        n_nodes = datamodule.node_counts()
        node_types = datamodule.node_types()
        edge_types = datamodule.edge_counts()

        # Store
        self.node_types = node_types
        self.edge_types = edge_types
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

        self.input_dims = None
        self.output_dims = None

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        """Compute input/output dims for the model."""
        import src.utils as utils

        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(
            example_batch.x, example_batch.edge_index,
            example_batch.edge_attr, example_batch.batch
        )

        # For SRL: X is continuous 768d, E is one-hot 58d
        x_dim = example_batch['x'].size(1)  # 768
        e_dim = example_batch['edge_attr'].size(1)  # 58

        # y is empty (0) + 1 for time conditioning
        y_size = example_batch['y'].size(1) if example_batch['y'].numel() > 0 else 0
        y_dim = y_size + 1  # +1 for time

        self.input_dims = {'X': x_dim, 'E': e_dim, 'y': y_dim}

        # Add extra feature dims
        example_data = {
            'X_t': ex_dense.X,
            'E_t': ex_dense.E,
            'y_t': torch.zeros(ex_dense.X.shape[0], 0),
            'node_mask': node_mask
        }

        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_domain_feat = domain_features(example_data)
        self.input_dims['X'] += ex_domain_feat.X.size(-1)
        self.input_dims['E'] += ex_domain_feat.E.size(-1)
        self.input_dims['y'] += ex_domain_feat.y.size(-1)

        # Output dims: only predict edges (not nodes or global features)
        self.output_dims = {
            'X': x_dim,    # Need to keep X output dim for transformer internal compatibility
            'E': e_dim,    # 58 edge classes
            'y': 0         # No global features to predict
        }
