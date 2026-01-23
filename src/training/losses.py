"""Loss functions for VAE training."""

import torch
import torch.nn.functional as F


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse"
) -> torch.Tensor:
    """Compute reconstruction loss between reconstructed and target tensors.

    Args:
        recon: Reconstructed tensor from decoder.
        target: Original input tensor.
        loss_type: Type of loss - 'mse' or 'bce'.

    Returns:
        Scalar loss tensor.
    """
    if loss_type == "mse":
        return F.mse_loss(recon, target)
    elif loss_type == "bce":
        return F.binary_cross_entropy(recon, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence for VAE latent space.

    KL(q(z|x) || p(z)) where q is the encoder distribution and p is N(0,1).

    Args:
        mu: Mean of the latent distribution.
        logvar: Log variance of the latent distribution.

    Returns:
        Scalar KL divergence loss.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    loss_type: str = "mse"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total VAE loss (reconstruction + KL divergence).

    Args:
        recon: Reconstructed tensor from decoder.
        target: Original input tensor.
        mu: Mean of the latent distribution.
        logvar: Log variance of the latent distribution.
        beta: Weight for KL divergence term (beta-VAE).
        loss_type: Type of reconstruction loss - 'mse' or 'bce'.

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss).
    """
    recon_loss = reconstruction_loss(recon, target, loss_type)
    kl_loss = kl_divergence(mu, logvar)
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
