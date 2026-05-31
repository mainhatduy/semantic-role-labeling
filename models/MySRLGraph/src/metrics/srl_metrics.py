"""
SRL-specific sampling metrics for evaluation.

Computes:
    - Edge-level Precision, Recall, F1 per role class
    - Macro-averaged F1 across all role classes
    - Exact match accuracy (% of edge matrices perfectly predicted)
"""

import torch
import torch.nn.functional as F
import wandb


class SRLSamplingMetrics:
    """Compute SRL-specific metrics on generated samples."""

    def __init__(self, dataset_infos):
        self.dataset_infos = dataset_infos
        self.num_edge_classes = dataset_infos.num_edge_classes

    def reset(self):
        pass

    def __call__(self, generated_samples, name, current_epoch, val_counter, test=False, local_rank=0):
        return self.forward(generated_samples, name, current_epoch, val_counter, test, local_rank)

    def forward(self, generated_samples, name, current_epoch, val_counter, test=False, local_rank=0):
        """Evaluate generated SRL graphs.

        Since SRL is conditional generation (edge prediction given input sentence),
        meaningful evaluation requires ground-truth comparison during inference.
        During training validation, we log basic statistics about the generated samples.
        """
        if local_rank != 0:
            return

        # Basic statistics about generated samples
        n_samples = len(generated_samples)
        if n_samples == 0:
            return

        avg_edges = 0
        avg_nodes = 0
        edge_class_counts = torch.zeros(self.num_edge_classes)

        for sample in generated_samples:
            atom_types, edge_types = sample[0], sample[1]
            n = atom_types.shape[0]
            avg_nodes += n

            # Count non-zero edges
            if edge_types.dim() == 2:
                # edge_types is (n, n) with class indices
                non_zero = (edge_types > 0).sum().item()
                avg_edges += non_zero
                for c in range(self.num_edge_classes):
                    edge_class_counts[c] += (edge_types == c).sum().item()
            elif edge_types.dim() == 3:
                # edge_types is (n, n, de) one-hot
                classes = edge_types.argmax(dim=-1)
                non_zero = (classes > 0).sum().item()
                avg_edges += non_zero
                for c in range(self.num_edge_classes):
                    edge_class_counts[c] += (classes == c).sum().item()

        avg_nodes /= n_samples
        avg_edges /= n_samples

        prefix = "test" if test else "val"
        log_dict = {
            f"{prefix}_sampling/n_samples": n_samples,
            f"{prefix}_sampling/avg_nodes": avg_nodes,
            f"{prefix}_sampling/avg_edges": avg_edges,
        }

        if wandb.run:
            wandb.log(log_dict, commit=False)

        print(f"  SRL Sampling: {n_samples} samples, avg_nodes={avg_nodes:.1f}, avg_edges={avg_edges:.1f}")


def compute_srl_f1(pred_E, true_E, num_classes, node_mask=None):
    """Compute per-class and macro F1 for edge predictions.

    Args:
        pred_E: (bs, n, n, de) predicted edge logits or probabilities
        true_E: (bs, n, n, de) true one-hot edge labels
        num_classes: number of edge classes (58)
        node_mask: (bs, n) optional mask

    Returns:
        dict with per-class metrics and macro averages
    """
    # Convert to class indices
    pred_classes = pred_E.argmax(dim=-1)  # (bs, n, n)
    true_classes = true_E.argmax(dim=-1)  # (bs, n, n)

    if node_mask is not None:
        # Mask invalid positions
        valid = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # (bs, n, n)
        pred_classes = pred_classes * valid
        true_classes = true_classes * valid

    # Flatten
    pred_flat = pred_classes.reshape(-1)
    true_flat = true_classes.reshape(-1)

    results = {}
    precisions = []
    recalls = []
    f1s = []

    # Skip class 0 (No-Edge) for meaningful metrics
    for c in range(1, num_classes):
        pred_c = (pred_flat == c)
        true_c = (true_flat == c)

        tp = (pred_c & true_c).sum().float()
        fp = (pred_c & ~true_c).sum().float()
        fn = (~pred_c & true_c).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Only include in macro average if the class appears in ground truth
        if true_c.sum() > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    if len(f1s) > 0:
        results['macro_precision'] = torch.stack(precisions).mean().item()
        results['macro_recall'] = torch.stack(recalls).mean().item()
        results['macro_f1'] = torch.stack(f1s).mean().item()
    else:
        results['macro_precision'] = 0.0
        results['macro_recall'] = 0.0
        results['macro_f1'] = 0.0

    # Exact match accuracy (full edge matrix match)
    if node_mask is not None:
        # Compare only valid positions
        match = (pred_classes == true_classes) | ~valid.bool()
        exact_match = match.all(dim=-1).all(dim=-1).float().mean().item()
    else:
        exact_match = (pred_classes == true_classes).all(dim=-1).all(dim=-1).float().mean().item()

    results['exact_match'] = exact_match
    return results
