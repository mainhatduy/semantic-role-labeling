# MySRLGraph: Edge-only Discrete Diffusion for SRL Graph Generation

## Problem Description

Build `models/MySRLGraph` — a **conditional, edge-only discrete denoising diffusion model** that generates SRL (Semantic Role Labeling) graphs from input sentences. Unlike DiGress which generates **both nodes and edges from scratch** (unconditional), MySRLGraph:

- **Fixes nodes** as the words of an input sentence (using DeBERTa embeddings as static features)
- **Only diffuses edges** — learning to predict the SRL role relationships between word pairs
- Uses a **directed, asymmetric edge matrix** (predicate→argument relationships are directional)
- Trains on `dataset/propbank/train.jsonl` (10,149 sentences, 57 unique role labels)

### Key Differences from DiGress

| Aspect | DiGress (Original) | MySRLGraph (Ours) |
|---|---|---|
| Generation type | Unconditional | Conditional on input sentence |
| Node features $X$ | Discrete, diffused | Continuous (DeBERTa 768d), **frozen** |
| Edge features $E$ | Symmetric (undirected) | **Asymmetric (directed)** |
| Edge classes | 5 (No-bond, Single, Double, Triple, Aromatic) | **58** (No-Edge + 57 SRL roles) |
| Loss | $\lambda_1 CE(X) + \lambda_2 CE(E) + \lambda_3 CE(y)$ | **$CE(E)$ only** |
| Noise process | Both $X$ and $E$ | **$E$ only** |
| Extra features | Cycles, Laplacian eigenvalues | **Not needed** (DeBERTa provides context) |

---

### 1. Edge Matrix Symmetry: Approved (Remove Constraint)

* **Verdict:** **Completely Correct.** * **Rationale:** In molecular graph generation (DiGress's original domain), chemical bonds are undirected (Carbon-Oxygen is the same as Oxygen-Carbon), which is why the source code forces $E = \frac{E + E^T}{2}$. In SRL, relations are strictly directional from the **Predicate** to the **Argument** (e.g., `killing` $\rightarrow$ `cleric` via `ARG1`). Enforcing symmetry would break the model by falsely teaching it that `cleric` also performs the `ARG1` role on `killing`.
* **Action Item:** In the Denoising Graph Transformer network (specifically where the edge logits are predicted/post-processed), delete or comment out the symmetry enforcement step (`E = (E + E.transpose(1, 2)) / 2`) to let the matrix remain asymmetric.

### 2. Node Representation: Option 1 Confirmed

* **Verdict:** **Option 1 is absolutely the superior path.**
* **Rationale:** As established, text tokens are immutable static conditions. Forcing Option 2 means the model wastes massive capacity learning a diffusion loop over a simple binary token classification (`Normal` vs `Predicate`), providing zero contextual value. Utilizing frozen DeBERTa-v3 embeddings ($768$-dimensional vectors) forces the model to leverage deep contextual, semantic, and syntactic relationships between words directly when denoising the edges.
* **Action Item:** Pass the DeBERTa hidden states directly as node features $X$. Ensure they bypass the forward noise loop (keep $\beta_t = 0$ for $X$).

---

### 3. Sentence Length and Memory: Approved Strategy

* **Verdict:** **Valid warning, strategy accepted.**
* **Rationale:** While the raw tensor allocation for $64 \times 64 \times 64 \times 58$ floats occupies a relatively small fraction of VRAM (under 100 MB), the internal intermediate activations during the Graph Transformer's attention layers scale quadratically, easily bloating memory consumption by $10\times$ to $20\times$.
* **Action Item:** 1. Set `max_seq_len = 64`.
2. Start training with a `batch_size = 16` or `32` to monitor your VRAM headroom.
3. Use **Gradient Accumulation** if you need a larger effective batch size without blowing up your hardware.

---

### 4. Multi-Predicate Handling: Option B Confirmed (Highly Recommended)

* **Verdict:** **Proceed with Option B (One Sample per Predicate).**
* **Rationale:** Option A introduces severe structural layout conflicts. In your image example, `killing` treats `cleric` as `ARG1`, but `causing` treats `killing` as `ARG0`. Merging everything into a single matrix makes the target matrix incredibly complex and structurally chaotic for the Graph Transformer. Option B splits the sentence into $P$ distinct training graphs (where $P$ is the number of predicates). This keeps each sample highly target-oriented: **"Given this sentence and *this specific predicate*, what are the arguments?"**

#### How Option B Reshapes Your Data Pipeline:

If a sentence has 3 predicates, your dataset generates 3 independent graphs for that single sentence:

1. **Graph 1:** Node features (DeBERTa tokens) + a special indicator channel (an extra scalar appended to $X$ where the 1st predicate token is marked as `1`, others `0`) + Edge Matrix for Predicate 1.
2. **Graph 2:** Node features + indicator channel marking the 2nd predicate token as `1` + Edge Matrix for Predicate 2.

This gives the network a crystal-clear focal point during the reverse sampling phase.
### Embedding model
Embedding model we use FacebookAI/xlm-roberta-base with embeding size 768d

---

## Proposed Changes

Changes are organized into 7 phases. All files are under `models/MySRLGraph/`.

---

### Phase 1: Dataset Module — PropBank → PyG Graph Data

This is the most critical phase. We need to convert `dataset/propbank/train.jsonl` into `torch_geometric.data.Data` objects compatible with the diffusion pipeline.

#### [NEW] [srl_dataset.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/datasets/srl_dataset.py)

New dataset class `SRLGraphDataset(InMemoryDataset)` that:

1. **Reads `train.jsonl`** — parses each sentence's words and predicates
2. **Tokenizes with DeBERTa** — uses `AutoTokenizer` + `AutoModel` to get contextual word embeddings
   - Handle subword tokens: average subword embeddings to get one embedding per word
   - Cache the embeddings to `.pt` files to avoid recomputation
3. **Builds edge matrix per predicate**:
   - For each predicate in a sentence, create a `(N, N)` matrix
   - Edge labels: `0 = No-Edge`, `1..57 = role indices` (loaded from `unique_roles.json`)
   - Row = predicate index, Column = argument indices → set the role label
   - **Directed**: only set `E[predicate_idx, argument_idx] = role_label`, do NOT mirror
4. **Creates `Data` objects**:
   - `data.x`: DeBERTa embeddings `(N, 768)` — continuous float tensor
   - `data.edge_index`: sparse representation for non-zero edges `(2, num_edges)`
   - `data.edge_attr`: one-hot edge attributes `(num_edges, 58)` — including No-Edge class
   - `data.y`: global features tensor `(1, 0)` — empty, since we condition on embeddings
   - `data.predicate_mask`: boolean mask `(N,)` indicating which word is the predicate for this sample

```python
# Pseudocode for data construction
role_to_idx = {"No-Edge": 0, "ARG0": 1, "ARG1": 2, ...}  # 58 classes total

for sentence in train_jsonl:
    words = sentence["words"]          # N words
    X = deberta_embed(words)           # (N, 768)
    
    for predicate in sentence["predicates"]:
        pred_idx = predicate["predicate_index"]
        edge_index_list = []
        edge_attr_list = []
        
        for arg in predicate["argument_spans"]:
            arg_idx = arg["token_index"]
            role = role_to_idx[arg["role"]]
            edge_index_list.append([pred_idx, arg_idx])
            edge_attr_list.append(role)
        
        # Build PyG Data
        edge_index = torch.tensor(edge_index_list).T    # (2, E)
        edge_attr = F.one_hot(torch.tensor(edge_attr_list), 58)
        
        data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=empty_y)
        data_list.append(data)
```

#### [NEW] [srl_dataset.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/datasets/srl_dataset.py) — `SRLDataModule` and `SRLDatasetInfos`

```python
class SRLDataModule(AbstractDataModule):
    """DataModule that loads PropBank SRL data."""
    # Split: 80% train, 10% val, 10% test
    # Uses AbstractDataModule's DataLoader creation

class SRLDatasetInfos(AbstractDatasetInfos):
    """Dataset metadata for SRL graphs."""
    # edge_types: distribution of 58 edge classes across training set
    # node_types: not meaningful (continuous), set to dummy
    # n_nodes: distribution of sentence lengths in the dataset
    # max_n_nodes: 64 (cutoff)
```

#### [MODIFY] [abstract_dataset.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/datasets/abstract_dataset.py)

- Add handling for **continuous node features** (DeBERTa embeddings are not one-hot)
- The `node_types()` method currently counts discrete classes — needs adaptation for continuous features (return a dummy distribution)
- `compute_input_output_dims` needs to handle `X_dim = 768` (embedding dim) instead of counting one-hot classes

---

### Phase 2: Diffusion Pipeline — Edge-Only Noise

The core change: **only apply noise to edges, keep nodes frozen**.

#### [MODIFY] [diffusion_model_discrete.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/diffusion_model_discrete.py)

This is the heart of the model. Major changes:

**`apply_noise()`** (line 407-442):
```diff
- # Apply noise to BOTH X and E
- probX = X @ Qtb.X
- probE = E @ Qtb.E.unsqueeze(1)
- sampled_t = sample_discrete_features(probX=probX, probE=probE, ...)
- X_t = F.one_hot(sampled_t.X, ...)
- E_t = F.one_hot(sampled_t.E, ...)
+ # Apply noise ONLY to E, keep X frozen
+ probE = E @ Qtb.E.unsqueeze(1)
+ sampled_E = sample_edge_only_features(probE=probE, node_mask=node_mask)
+ E_t = F.one_hot(sampled_E, num_classes=self.Edim_output)
+ X_t = X  # Keep DeBERTa embeddings unchanged!
```

**`training_step()`** (line 103-120):
```diff
  dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
- X, E = dense_data.X, dense_data.E
+ X_embed = dense_data.X    # (bs, n, 768) continuous embeddings — NOT one-hot
+ E = dense_data.E           # (bs, n, n, 58) one-hot edge labels
  
- noisy_data = self.apply_noise(X, E, data.y, node_mask)
+ noisy_data = self.apply_noise_edges_only(X_embed, E, data.y, node_mask)
  
  pred = self.forward(noisy_data, extra_data, node_mask)
  
- loss = self.train_loss(pred_X=pred.X, pred_E=pred.E, true_X=X, true_E=E, ...)
+ loss = self.train_loss_edges_only(pred_E=pred.E, true_E=E, ...)
```

**`forward()`** (line 485-489):
```diff
- X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
- E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
+ X = noisy_data['X_t'].float()   # Already 768d DeBERTa embeddings, no extra features
+ E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()  # noisy edge + time embedding
```

**`sample_batch()`** (line 491-595) — for inference:
```diff
- # Sample random X_T and E_T from prior
- z_T = sample_discrete_feature_noise(limit_dist, node_mask)
+ # X is given (DeBERTa embeddings of input sentence), only sample E_T
+ X = input_embeddings  # Fixed, from input sentence
+ E_T = sample_edge_only_noise(limit_dist_E, node_mask)
```

**`sample_p_zs_given_zt()`** (line 597-655):
```diff
- # Compute posterior for BOTH X and E
- p_s_given_0_X = compute_batched_over0_posterior(X_t, Qt.X, ...)
- p_s_given_0_E = compute_batched_over0_posterior(E_t, Qt.E, ...)
+ # Compute posterior for E ONLY
+ p_s_given_0_E = compute_batched_over0_posterior(E_t, Qt.E, ...)
+ # X remains fixed throughout
```

**Remove symmetry enforcement** throughout:
```diff
- E = 1/2 * (E + torch.transpose(E, 1, 2))  # REMOVE THIS — SRL edges are directed
- assert (E == torch.transpose(E, 1, 2)).all()  # REMOVE THIS
- U_E = torch.triu(E_t, diagonal=1)  # REMOVE — no longer symmetric
- U_E = (U_E + torch.transpose(U_E, 1, 2))  # REMOVE
```

**Transition model initialization** — only need edge transition:
```diff
  self.transition_model = MarginalUniformTransition(
-     x_marginals=x_marginals,
+     x_marginals=torch.ones(1),  # dummy, not used
      e_marginals=e_marginals,    # computed from SRL dataset
      y_classes=0
  )
```

#### [MODIFY] [diffusion_utils.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/diffusion/diffusion_utils.py)

**`sample_discrete_features()`** (line 233-266):
- Create a new function `sample_edge_only_features()` that only samples edges
- Remove upper-triangular + transpose logic (was for symmetric graphs)
- Keep diagonal mask (no self-loops)

**`sample_discrete_feature_noise()`** (line 366-394):
- Create `sample_edge_only_noise()` — only samples edge noise from limit distribution
- Remove symmetry enforcement

**`mask_distributions()`** (line 324-356):
- Adapt to handle continuous X (skip X masking) and asymmetric E

#### [MODIFY] [noise_schedule.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/diffusion/noise_schedule.py)

- `DiscreteUniformTransition`: set `x_classes=0` or bypass X transition matrix computation
- `MarginalUniformTransition`: same — only compute `q_e`, skip `q_x`
- Remove all `q_x` and `q_y` computations in `get_Qt()` and `get_Qt_bar()`

---

### Phase 3: Graph Transformer — Accept Continuous X + Directed E

#### [MODIFY] [transformer_model.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/models/transformer_model.py)

**`GraphTransformer.__init__()`**:
```diff
  # Input MLP for X now takes continuous 768d embeddings
  self.mlp_in_X = nn.Sequential(
-     nn.Linear(input_dims['X'], hidden_mlp_dims['X']),  # X is one-hot count
+     nn.Linear(input_dims['X'], hidden_mlp_dims['X']),  # X is 768d DeBERTa
      act_fn_in,
      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']),
      act_fn_in
  )
```

**`GraphTransformer.forward()`**:
```diff
  # Remove skip connection for X (output X is not used in edge-only mode)
- X_to_out = X[..., :self.out_dim_X]
  E_to_out = E[..., :self.out_dim_E]
  
  # Remove symmetry enforcement on E output
- E = 1/2 * (E + torch.transpose(E, 1, 2))  # REMOVE for directed graphs
  
  # Remove diagonal zeroing for E — keep it for self-loops
  # (self-loop = word relating to itself = No-Edge, already handled by data)
```

**`NodeEdgeBlock.forward()`**:
- The attention mechanism naturally works with both continuous X and discrete E
- Key change: **do NOT symmetrize** the updated edge features `newE`
- The `e_out` projection maps from `dx → de` — output dims change since de is now 58

**Output dims**:
```python
output_dims = {
    'X': 0,    # We don't predict X (nodes are fixed)
    'E': 58,   # Predict 58-class edge labels
    'y': 0     # No global features to predict
}
```

> [!NOTE]
> Since we don't predict X, we can either: (a) still have `mlp_out_X` but ignore its output, or (b) skip it entirely. Option (b) saves compute. We'll still update X internally through transformer layers for message passing, but won't project X to an output prediction.

---

### Phase 4: Training Module — Edge-Only Loss

#### [MODIFY] [train_metrics.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/metrics/train_metrics.py)

**`TrainLossDiscrete`**: Simplify to edge-only loss:

```python
class TrainLossEdgeOnly(nn.Module):
    """Train with Cross entropy on edges only."""
    def __init__(self):
        super().__init__()
        self.edge_loss = CrossEntropyMetric()
    
    def forward(self, pred_E, true_E, log: bool):
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))
        pred_E = torch.reshape(pred_E, (-1, pred_E.size(-1)))
        
        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = pred_E[mask_E, :]
        
        loss_E = self.edge_loss(flat_pred_E, flat_true_E)
        
        if log:
            wandb.log({"train_loss/E_CE": self.edge_loss.compute()})
        
        return loss_E
```

#### [NEW] [srl_metrics.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/metrics/srl_metrics.py)

SRL-specific metrics for evaluation:
- **Edge-level F1 / Precision / Recall**: For each role class
- **Macro-averaged F1**: Across all role classes
- **Exact Match Accuracy**: What % of edge matrices are perfectly predicted

---

### Phase 5: Extra Features — Simplify

#### [MODIFY] [extra_features.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/diffusion/extra_features.py)

- Use `DummyExtraFeatures` — DeBERTa embeddings already encode rich linguistic features
- Cycle features / Laplacian eigenvalues are designed for molecular graphs, not useful for NLP
- Only append time embedding `t` to the global feature `y`

---

### Phase 6: Configuration & Entrypoint

#### [NEW] [configs/dataset/srl_propbank.yaml](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/configs/dataset/srl_propbank.yaml)

```yaml
name: 'srl_propbank'
datadir: '../../dataset/propbank'       # relative to project root
train_file: 'train.jsonl'
roles_file: 'unique_roles.json'
max_seq_len: 64                         # truncate sentences longer than this
deberta_model: 'microsoft/deberta-v3-base'
deberta_dim: 768
split_ratios: [0.8, 0.1, 0.1]          # train/val/test
pin_memory: False
```

#### [NEW] [configs/experiment/srl.yaml](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/configs/experiment/srl.yaml)

```yaml
# @package _global_
general:
    name: 'srl_edge_diffusion'
    gpus: 1
    wandb: 'online'
    test_only: null
    check_val_every_n_epochs: 5
    sample_every_val: 10
    samples_to_generate: 32
    samples_to_save: 10
    chains_to_save: 1
    log_every_steps: 50

train:
    n_epochs: 500
    batch_size: 32           # Smaller due to N×N×58 edge tensor
    save_model: True
    lr: 1e-4
    num_workers: 4

model:
    n_layers: 6
    type: 'discrete'
    transition: 'marginal'
    diffusion_steps: 500
    diffusion_noise_schedule: 'cosine'
    extra_features: null     # No extra features needed
    edge_only: true          # New flag for edge-only diffusion
    
    hidden_mlp_dims: {'X': 256, 'E': 128, 'y': 128}
    hidden_dims: {'dx': 256, 'de': 128, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}

dataset:
    name: 'srl_propbank'
```

#### [MODIFY] [configs/config.yaml](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/configs/config.yaml)

```yaml
defaults:
    - _self_
    - general: general_default
    - model: discrete
    - train: train_default
    - dataset: srl_propbank    # Changed from qm9
```

#### [MODIFY] [main.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/main.py)

Add SRL dataset branch:

```python
if dataset_config["name"] == 'srl_propbank':
    from datasets.srl_dataset import SRLDataModule, SRLDatasetInfos
    from metrics.srl_metrics import SRLSamplingMetrics
    from metrics.train_metrics import TrainLossEdgeOnly
    from diffusion.extra_features import DummyExtraFeatures
    
    datamodule = SRLDataModule(cfg)
    dataset_infos = SRLDatasetInfos(datamodule, cfg)
    
    extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    
    dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)
    
    train_metrics = TrainAbstractMetricsDiscrete()
    sampling_metrics = SRLSamplingMetrics(dataset_infos)
    visualization_tools = None  # or SRLVisualization
    
    model_kwargs = { ... }
```

---

### Phase 7: `utils.py` — Adapt for Continuous Nodes + Directed Edges

#### [MODIFY] [utils.py](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/src/utils.py)

**`to_dense()`**:
- Currently converts sparse PyG format to dense `(bs, N, N, de)` tensor
- Needs to handle:
  - Continuous `x` (not one-hot) — `to_dense_batch` already works for continuous features
  - **Directed edges** — `to_dense_adj` creates adjacency, but we need to ensure NO symmetrization
  - `encode_no_edge()` still works (sets the No-Edge class at index 0)

**`PlaceHolder.mask()`**:
```diff
- assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))  # REMOVE — asymmetric
+ # No symmetry assertion for directed SRL edges
```

#### [MODIFY] [requirements.txt](file:///home/myduy/Workspace/research/semantic-role-labeling/models/MySRLGraph/requirements.txt)

Add:
```
transformers>=4.30.0
```

---

## File Change Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/datasets/srl_dataset.py` | **NEW** | PropBank JSONL parser, DeBERTa embedder, PyG Data builder |
| `src/diffusion_model_discrete.py` | **MODIFY** | Edge-only diffusion, remove node noise, remove symmetry |
| `src/diffusion/diffusion_utils.py` | **MODIFY** | Add edge-only sampling, remove symmetry constraints |
| `src/diffusion/noise_schedule.py` | **MODIFY** | Edge-only transition matrices |
| `src/diffusion/extra_features.py` | **MODIFY** | Use DummyExtraFeatures only |
| `src/models/transformer_model.py` | **MODIFY** | Accept 768d continuous X, directed E, output E-only |
| `src/metrics/train_metrics.py` | **MODIFY** | Edge-only CrossEntropy loss |
| `src/metrics/srl_metrics.py` | **NEW** | SRL F1/Precision/Recall evaluation |
| `src/utils.py` | **MODIFY** | Remove symmetry assertions, handle continuous X |
| `src/main.py` | **MODIFY** | Add SRL dataset branch |
| `configs/dataset/srl_propbank.yaml` | **NEW** | SRL dataset config |
| `configs/experiment/srl.yaml` | **NEW** | SRL experiment config |
| `configs/config.yaml` | **MODIFY** | Default to SRL dataset |
| `requirements.txt` | **MODIFY** | Add `transformers` dependency |

---

## Verification Plan

### Automated Tests

1. **Dataset unit test**: Verify that `SRLGraphDataset` correctly parses `train.jsonl` and produces valid PyG Data objects with correct shapes:
   ```bash
   python -c "from datasets.srl_dataset import SRLDataModule; dm = SRLDataModule(cfg); print(next(iter(dm.train_dataloader())))"
   ```

2. **Edge matrix verification**: For a known sentence (e.g., sentence 2 from train.jsonl), manually verify the edge matrix matches the expected SRL roles.

3. **Forward pass smoke test**: Run a single training step to ensure no shape mismatches:
   ```bash
   python src/main.py +experiment=srl general.name=test train.n_epochs=1 train.batch_size=2
   ```

4. **Loss convergence**: Train for 10 epochs and verify loss decreases.

### Manual Verification

5. **Visualize generated graphs**: After training, sample some SRL graphs and compare with ground truth
6. **Check edge predictions**: For a known input sentence, verify the model predicts reasonable SRL roles
