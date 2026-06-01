import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError


def get_alpha_weights(freq: torch.Tensor, epsilon: float = 1e-10,
                       max_weight: float = 15.0, min_weight: float = 0.1) -> torch.Tensor:
    """Compute class-balancing alpha weights from a marginal class distribution.

    Uses a **log-inverse-frequency** scheme normalised by the **median** class,
    which avoids the pathology of pure 1/freq weighting where the dominant class
    (99.7% background) gets weight ≈ 0 and the model receives almost no gradient.

    The formula is::

        raw_w = log(1 / (freq + ε))          # log dampens extreme ratios
        weights = raw_w / median(raw_w)       # median-normalise (robust to outliers)
        weights = clamp(weights, min, max)    # bound to [min_weight, max_weight]

    With the SRL edge distribution this gives roughly:
    - Background (99.7%):  weight ≈ 0.1  (min-clamped, not zero!)
    - Common classes:      weight ≈ 1–3
    - Rare classes:        weight ≈ 8–15 (max-clamped)

    Args:
        freq: 1-D tensor ``[num_classes]`` of marginal class probabilities.
        epsilon: Added to freq before log to handle zero-frequency classes.
        max_weight: Upper clamp for weights (prevents gradient explosion).
        min_weight: Lower clamp for weights (ensures even the dominant class
            still contributes gradient).

    Returns:
        1-D ``torch.Tensor`` of shape ``[num_classes]``.
    """
    # log-inverse-frequency (log dampens the extreme range of 1/freq)
    raw = torch.log(1.0 / (freq.float() + epsilon))

    # Normalise by median — robust to the extreme outliers from zero-freq classes
    median_val = raw.median()
    weights = raw / median_val

    # Clamp both ends so no class is silenced or explodes
    weights = weights.clamp(min=min_weight, max=max_weight)

    return weights


class TrainAbstractMetricsDiscrete(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class TrainAbstractMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchMSE(MeanSquaredError):
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: Tensor, target: Tensor):
            """ Updates and returns variables required to compute Mean Squared Error. Checks for same shape of input
            tensors.
                preds: Predicted tensor
                target: Ground truth tensor
            """
            diff = preds - target
            sum_squared_error = torch.sum(diff * diff)
            n_obs = preds.shape[0]
            return sum_squared_error, n_obs


class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class FocalLossMetric(Metric):
    """Focal Loss as a TorchMetrics Metric with per-class alpha weighting.

    Designed for extreme class-imbalance scenarios (e.g. SRL edge classification
    where the background class dominates at ~99.7%).

    Args:
        alpha: 1-D tensor of shape ``[num_classes]`` with per-class weights.
            Registered as a buffer so PyTorch Lightning automatically moves it to
            the correct device. Pass ``None`` to disable class weighting.
        gamma: Focusing parameter. Higher values down-weight easy (well-classified)
            examples more aggressively. Default: ``2.0``.
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

        # Register alpha as a buffer so it auto-moves to GPU/TPU with PL
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.register_buffer("alpha", None)

        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate focal loss over a batch.

        Args:
            preds: Raw logits of shape ``(N, C)``.
            target: One-hot encoded targets of shape ``(N, C)``.
        """
        # Convert one-hot -> class indices
        target_indices = torch.argmax(target, dim=-1)  # (N,)

        # Per-sample cross-entropy (unreduced)
        ce_loss = F.cross_entropy(preds, target_indices, reduction="none")  # (N,)

        # Probability of the true class
        pt = torch.exp(-ce_loss)  # (N,)

        # Focal modulation factor
        focal_weight = (1.0 - pt) ** self.gamma  # (N,)

        # Apply per-class alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha[target_indices]  # (N,)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        self.total_loss += focal_loss.sum()
        self.total_samples += preds.size(0)

    def compute(self) -> Tensor:
        """Return the mean focal loss over all accumulated samples."""
        return self.total_loss / self.total_samples


class ProbabilityMetric(Metric):
    def __init__(self):
        """ This metric is used to track the marginal predicted probability of a class during training. """
        super().__init__()
        self.add_state('prob', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        self.prob += preds.sum()
        self.total += preds.numel()

    def compute(self):
        return self.prob / self.total


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples