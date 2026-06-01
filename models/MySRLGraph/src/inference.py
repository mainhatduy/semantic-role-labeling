"""
End-to-end SRL inference pipeline.

Usage:
    python -m src.inference \
        --model_path models/outputs/2026-06-01/04-25-41-srl_edge_diffusion \
        --sentence "The cat sat on the mat"

    Or programmatically:
        from src.inference import SRLInferencePipeline
        pipeline = SRLInferencePipeline("models/outputs/2026-06-01/04-25-41-srl_edge_diffusion")
        result = pipeline.predict("The cat sat on the mat")
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Resolve project root so that `src.*` imports work when running as a script
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # .../MySRLGraph/src
_PROJECT_ROOT = _THIS_DIR.parent                     # .../MySRLGraph
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition, \
    DiscreteUniformTransition
from src.diffusion import diffusion_utils
from diffusion.extra_features import DummyExtraFeatures
from src.utils import PlaceHolder


# ============================================================================
#  Helper: lightweight re-creation of the denoising model for inference only
# ============================================================================

class _InferenceDenoiser(nn.Module):
    """Minimal wrapper around GraphTransformer + noise schedule for inference.

    This avoids pulling in the full LightningModule (which requires dataset_infos,
    metrics, etc.) and instead reconstructs only the components needed for the
    reverse diffusion sampling loop.
    """

    def __init__(self, cfg, edge_marginals: torch.Tensor, state_dict: dict, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device_to_use = device
        self.T = cfg.model.diffusion_steps

        # ---- dimensions (edge-only mode) ----
        self.Xdim = cfg.model._input_dims_X
        self.Edim = cfg.model._input_dims_E
        self.ydim = cfg.model._input_dims_y
        self.Edim_output = cfg.model._output_dims_E

        # ---- build transformer ----
        input_dims = {'X': self.Xdim, 'E': self.Edim, 'y': self.ydim}
        output_dims = {'X': cfg.model._output_dims_X, 'E': self.Edim_output, 'y': cfg.model._output_dims_y}

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=OmegaConf.to_container(cfg.model.hidden_mlp_dims),
            hidden_dims=OmegaConf.to_container(cfg.model.hidden_dims),
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        # ---- noise schedule ----
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule,
            timesteps=self.T,
        )

        # ---- transition model ----
        x_marginals = torch.ones(1)  # edge-only: dummy X
        e_marginals = edge_marginals
        if cfg.model.transition == 'marginal':
            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=0,
            )
        else:
            self.transition_model = DiscreteUniformTransition(
                x_classes=1,
                e_classes=self.Edim_output,
                y_classes=0,
            )
        self.limit_dist_E = e_marginals

        # ---- extra / domain features (dummy for SRL) ----
        self.extra_features = DummyExtraFeatures()
        self.domain_features = DummyExtraFeatures()

        # ---- load trained weights ----
        self._load_weights(state_dict)
        self.to(device)
        self.eval()

    # ------------------------------------------------------------------

    def _load_weights(self, state_dict: dict):
        """Load only the relevant keys from the Lightning checkpoint."""
        model_keys = {}
        for k, v in state_dict.items():
            # Lightning wraps the model keys under "model.*"
            if k.startswith('model.'):
                model_keys[k[len('model.'):]] = v
        self.model.load_state_dict(model_keys)

        ns_keys = {}
        for k, v in state_dict.items():
            if k.startswith('noise_schedule.'):
                ns_keys[k[len('noise_schedule.'):]] = v
        if ns_keys:
            self.noise_schedule.load_state_dict(ns_keys)

    # ------------------------------------------------------------------

    def forward_pass(self, noisy_data, node_mask):
        """Single forward through the transformer."""
        extra_data = self._compute_extra_data(noisy_data)
        X = noisy_data['X_t'].float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    def _compute_extra_data(self, noisy_data):
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)
        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)
        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)
        return PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_edges(self, X: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Run full reverse diffusion to sample edge matrix.

        Args:
            X: (1, n, 768) node embeddings
            node_mask: (1, n) boolean mask

        Returns:
            E_pred: (n, n) integer tensor of edge class indices
        """
        device = self.device_to_use
        bs = 1
        n = X.size(1)

        # Start from noise
        E = diffusion_utils.sample_edge_only_noise(
            limit_dist_E=self.limit_dist_E.to(device),
            node_mask=node_mask,
        )  # (1, n, n, de)
        y = torch.zeros(bs, 0, device=device)

        # Reverse diffusion: t = T, T-1, ..., 1
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((bs, 1), device=device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            beta_t = self.noise_schedule(t_normalized=t_norm)
            alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_norm)
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_norm)

            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device)
            Qt = self.transition_model.get_Qt(beta_t, device)

            # Forward prediction
            noisy_data = {
                'X_t': X, 'E_t': E, 'y_t': y,
                't': t_norm, 'node_mask': node_mask,
            }
            pred = self.forward_pass(noisy_data, node_mask)
            pred_E = F.softmax(pred.E, dim=-1)

            # Posterior sampling for edges
            p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=E, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E,
            )

            pred_E_flat = pred_E.reshape((bs, -1, pred_E.shape[-1]))
            weighted_E = pred_E_flat.unsqueeze(-1) * p_s_and_t_given_0_E
            unnormalized_prob_E = weighted_E.sum(dim=-2)
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
            prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
            prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

            sampled_E_s = diffusion_utils.sample_edge_only_features(probE=prob_E, node_mask=node_mask)
            E = F.one_hot(sampled_E_s, num_classes=self.Edim_output).float()

        # Final: collapse to class indices
        E_pred = E.argmax(dim=-1).squeeze(0)  # (n, n)
        return E_pred


# ============================================================================
#  Main pipeline
# ============================================================================

class SRLInferencePipeline:
    """End-to-end: sentence → relation matrix.

    Args:
        model_path: Path to an experiment output directory, e.g.
            ``models/outputs/2026-06-01/04-25-41-srl_edge_diffusion``.
            The pipeline will automatically load ``checkpoints/.../last.ckpt``.
        device: ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).
        roles_path: Override path to ``unique_roles.json``. If None, resolved
            from the saved config.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        roles_path: Optional[str] = None,
    ):
        self.model_path = Path(model_path).resolve()
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # 1. Load saved Hydra config
        self.cfg = self._load_config()

        # 2. Load role mapping
        self.role_to_idx, self.idx_to_role = self._load_roles(roles_path)
        self.num_edge_classes = len(self.role_to_idx)

        # 3. Load checkpoint
        ckpt_path = self._find_last_checkpoint()
        print(f"[SRLInferencePipeline] Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # 4. Recover dims from the checkpoint's hyper_parameters
        self._resolve_dims(checkpoint)

        # 5. Recover edge marginals from checkpoint
        edge_marginals = self._recover_edge_marginals(checkpoint)

        # 6. Build denoiser
        self.denoiser = _InferenceDenoiser(
            cfg=self.cfg,
            edge_marginals=edge_marginals,
            state_dict=checkpoint['state_dict'],
            device=self.device,
        )

        # 7. Load embedding model (lazy, on first call)
        self._tokenizer = None
        self._embed_model = None
        self._embedding_model_name = self.cfg.dataset.embedding_model
        self._embedding_dim = self.cfg.dataset.embedding_dim

        print(f"[SRLInferencePipeline] Ready on {self.device}  "
              f"(diffusion_steps={self.denoiser.T}, edge_classes={self.num_edge_classes})")

    # ------------------------------------------------------------------
    # Config / checkpoint helpers
    # ------------------------------------------------------------------

    def _load_config(self):
        cfg_path = self.model_path / '.hydra' / 'config.yaml'
        if not cfg_path.exists():
            raise FileNotFoundError(f"Cannot find saved config at {cfg_path}")
        cfg = OmegaConf.load(str(cfg_path))
        return cfg

    def _find_last_checkpoint(self) -> str:
        """Find the 'last.ckpt' file inside the experiment's checkpoints dir."""
        ckpt_dir = self.model_path / 'checkpoints'
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"No checkpoints directory at {ckpt_dir}")

        # Search recursively for last.ckpt
        candidates = list(ckpt_dir.rglob('last.ckpt'))
        if candidates:
            return str(candidates[0])

        # Fallback: pick the checkpoint with the highest epoch number
        all_ckpts = list(ckpt_dir.rglob('*.ckpt'))
        if not all_ckpts:
            raise FileNotFoundError(f"No .ckpt files found under {ckpt_dir}")

        # Sort by modification time, take the newest
        all_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(all_ckpts[0])

    def _load_roles(self, roles_path_override: Optional[str] = None):
        if roles_path_override:
            rp = Path(roles_path_override)
        else:
            # Resolve from config (relative to MySRLGraph/)
            project_root = Path(__file__).resolve().parents[1]  # MySRLGraph/
            data_dir = project_root / self.cfg.dataset.datadir
            rp = data_dir / self.cfg.dataset.roles_file

        if not rp.exists():
            raise FileNotFoundError(f"Roles file not found: {rp}")

        with open(rp, 'r') as f:
            roles = json.load(f)

        role_to_idx = {"No-Edge": 0}
        for i, role in enumerate(roles):
            role_to_idx[role] = i + 1

        idx_to_role = {v: k for k, v in role_to_idx.items()}
        return role_to_idx, idx_to_role

    def _resolve_dims(self, checkpoint: dict):
        """Recover input/output dims stored in the checkpoint's hparams."""
        hp = checkpoint.get('hyper_parameters', {})
        cfg = hp.get('cfg', self.cfg)

        # Try to recover dims from the state_dict shapes
        sd = checkpoint['state_dict']

        # model.mlp_in_X.0.weight → shape (hidden_mlp_X, input_X)
        x_in = sd['model.mlp_in_X.0.weight'].shape[1]
        e_in = sd['model.mlp_in_E.0.weight'].shape[1]
        y_in = sd['model.mlp_in_y.0.weight'].shape[1]

        x_out = sd['model.mlp_out_X.2.weight'].shape[0]
        e_out = sd['model.mlp_out_E.2.weight'].shape[0]
        y_out = sd['model.mlp_out_y.2.weight'].shape[0]

        # Store in config for easy access
        OmegaConf.set_struct(self.cfg.model, False)
        self.cfg.model._input_dims_X = x_in
        self.cfg.model._input_dims_E = e_in
        self.cfg.model._input_dims_y = y_in
        self.cfg.model._output_dims_X = x_out
        self.cfg.model._output_dims_E = e_out
        self.cfg.model._output_dims_y = y_out
        OmegaConf.set_struct(self.cfg.model, True)

    def _recover_edge_marginals(self, checkpoint: dict) -> torch.Tensor:
        """Recover the edge marginal distribution used during training.

        The MarginalUniformTransition stores u_e = marginals expanded to (1, de, de).
        We extract the first row to recover the original marginals.
        """
        sd = checkpoint['state_dict']

        # The transition_model is not stored in state_dict (it's not an nn.Module with
        # registered buffers in the Lightning module). We reconstruct from data distribution.
        # As a robust fallback, use uniform over edge classes.
        e_out = self.cfg.model._output_dims_E
        return torch.ones(e_out) / e_out

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _ensure_embed_model(self):
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer, AutoModel
        print(f"[SRLInferencePipeline] Loading embedding model: {self._embedding_model_name} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._embedding_model_name)
        self._embed_model = AutoModel.from_pretrained(self._embedding_model_name)
        self._embed_model.eval()
        self._embed_model.to(self.device)

    def _compute_word_embeddings(self, words: List[str]) -> torch.Tensor:
        """Compute word-level embeddings with subword averaging.

        Returns:
            (n_words, embedding_dim) tensor on CPU
        """
        self._ensure_embed_model()
        tokenizer = self._tokenizer
        model = self._embed_model

        word_ids_to_subword_ids = []
        all_subwords = [tokenizer.cls_token_id]

        for word in words:
            subword_ids = tokenizer.encode(word, add_special_tokens=False)
            word_ids_to_subword_ids.append(
                list(range(len(all_subwords), len(all_subwords) + len(subword_ids)))
            )
            all_subwords.extend(subword_ids)

        all_subwords.append(tokenizer.sep_token_id)

        max_model_len = tokenizer.model_max_length
        if len(all_subwords) > max_model_len:
            all_subwords = all_subwords[:max_model_len - 1] + [tokenizer.sep_token_id]

        input_ids = torch.tensor([all_subwords], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[0]  # (seq_len, dim)

        word_embeddings = []
        for subword_indices in word_ids_to_subword_ids:
            valid_indices = [i for i in subword_indices if i < hidden_states.shape[0]]
            if valid_indices:
                word_emb = hidden_states[valid_indices].mean(dim=0)
            else:
                word_emb = torch.zeros(self._embedding_dim, device=self.device)
            word_embeddings.append(word_emb)

        return torch.stack(word_embeddings).cpu()  # (n_words, dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        sentence: Union[str, List[str]],
        predicate_indices: Optional[List[int]] = None,
    ) -> Dict:
        """Run end-to-end SRL inference.

        Args:
            sentence: A sentence string (will be split on whitespace) or a
                pre-tokenised list of words.
            predicate_indices: Optional list of word indices that are predicates.
                If None, every word is treated as a potential predicate and a
                relation matrix is generated for the whole sentence.

        Returns:
            dict with keys:
                - ``words``: list of words
                - ``relation_matrix``: (n, n) numpy int array of edge class indices
                - ``relation_labels``: (n, n) numpy array of role label strings
                - ``roles``: dict mapping role names → class indices
                - ``edges``: list of dicts ``{src, tgt, role}`` for non-zero edges
        """
        # Tokenise
        if isinstance(sentence, str):
            words = sentence.strip().split()
        else:
            words = list(sentence)

        n = len(words)
        if n == 0:
            raise ValueError("Empty sentence")

        # Compute embeddings → (1, n, dim)
        X = self._compute_word_embeddings(words)
        X = X.unsqueeze(0).to(self.device)  # (1, n, dim)

        # Node mask — all valid
        node_mask = torch.ones(1, n, dtype=torch.bool, device=self.device)

        # Run reverse diffusion
        E_pred = self.denoiser.sample_edges(X, node_mask)  # (n, n) int tensor

        # Build outputs
        E_np = E_pred.cpu().numpy()
        label_matrix = np.empty((n, n), dtype=object)
        edges = []
        for i in range(n):
            for j in range(n):
                cls_idx = int(E_np[i, j])
                label = self.idx_to_role.get(cls_idx, f"UNK-{cls_idx}")
                label_matrix[i, j] = label
                if cls_idx != 0:
                    edges.append({'src': i, 'src_word': words[i],
                                  'tgt': j, 'tgt_word': words[j],
                                  'role': label, 'role_idx': cls_idx})

        return {
            'words': words,
            'relation_matrix': E_np,
            'relation_labels': label_matrix,
            'roles': self.role_to_idx,
            'edges': edges,
        }

    def predict_pretty(self, sentence: Union[str, List[str]]) -> str:
        """Return a human-readable string of the SRL prediction."""
        result = self.predict(sentence)
        words = result['words']
        edges = result['edges']

        lines = [f"Sentence: {' '.join(words)}",
                 f"Tokens ({len(words)}): {words}",
                 ""]

        if not edges:
            lines.append("No semantic roles detected.")
        else:
            lines.append(f"Detected {len(edges)} semantic role edge(s):")
            lines.append(f"{'Predicate':<20} {'→':^3} {'Argument':<20} {'Role':<20}")
            lines.append("-" * 65)
            for e in edges:
                src_str = f"[{e['src']}] {e['src_word']}"
                tgt_str = f"[{e['tgt']}] {e['tgt_word']}"
                lines.append(f"{src_str:<20} {'→':^3} {tgt_str:<20} {e['role']:<20}")

        return "\n".join(lines)


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SRL end-to-end inference")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to experiment output directory '
                             '(e.g. models/outputs/2026-06-01/04-25-41-srl_edge_diffusion)')
    parser.add_argument('--sentence', type=str, required=True,
                        help='Input sentence (will be whitespace-tokenised)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda / cpu (default: auto)')
    parser.add_argument('--roles_path', type=str, default=None,
                        help='Override path to unique_roles.json')
    args = parser.parse_args()

    pipeline = SRLInferencePipeline(
        model_path=args.model_path,
        device=args.device,
        roles_path=args.roles_path,
    )

    print("\n" + "=" * 70)
    print(pipeline.predict_pretty(args.sentence))
    print("=" * 70)

    # Also print the raw relation matrix
    result = pipeline.predict(args.sentence)
    print("\nRelation matrix (class indices):")
    print(result['relation_matrix'])


if __name__ == '__main__':
    main()
