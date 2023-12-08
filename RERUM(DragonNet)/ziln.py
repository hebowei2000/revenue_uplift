import torch
import torch.nn.functional as F
import torch.distributions as tdist


def zero_inflated_lognormal_pred(logits):
    """Calculates predicted mean of zero inflated lognormal logits.

    Arguments:
        logits: [batch_size, 3] tensor of logits.

    Returns:
        preds: [batch_size, 1] tensor of predicted mean.
    """
    positive_probs = torch.sigmoid(logits[..., :1])
    loc = logits[..., 1:2]
    scale = torch.nn.functional.softplus(logits[..., 2:])
    preds = positive_probs * torch.exp(loc + 0.5 * scale**2)
    return preds


def zero_inflated_lognormal_loss(labels, logits):
    """Computes the zero inflated lognormal loss.

    Arguments:
        labels: True targets, tensor of shape [batch_size, 1].
        logits: Logits of output layer, tensor of shape [batch_size, 3].

    Returns:
        Zero inflated lognormal loss value.
    """
    positive = (labels > 0).float()

    positive_logits = logits[..., :1]
    classification_loss = F.binary_cross_entropy_with_logits(
        positive_logits, positive, reduction='mean')

    loc = logits[..., 1:2]
    scale = torch.max(
        torch.nn.functional.softplus(logits[..., 2:]),
        torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps)))
    safe_labels = positive * labels + (1 - positive) * torch.ones_like(labels)
    log_prob = tdist.LogNormal(loc=loc, scale=scale).log_prob(safe_labels)
    regression_loss = -torch.mean(positive * log_prob, dim=-1)

    return classification_loss + regression_loss