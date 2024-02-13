import torch
import numpy as np

def _add_dict(d, new_d):
    """Add values of dictionary new_d to dictionary d."""
    for key, value in new_d.items():
        d.setdefault(key, []).append(value)
    return d

def compute_unnormalized_kl(p, q):
    """Compute unnormalized Kullback-Leibler divergence between two vectors.

    Parameters
    ----------
    p: torch tensor
    q: torch tensor
        Should have the same size as p

    Returns
    -------
    unnormalized_kl: float"""
    # By convention: 0 log 0 = 0
    entropy = torch.nan_to_num(
        p * (p / q).log(), nan=0.0, posinf=0.0, neginf=0.0
    ).sum()
    return entropy

def compute_divergence(p, q):
    """Compute Kullback-Leibler divergence between two vectors p and q

    Parameters
    ----------
    p: torch tensor
    q: torch tensor
        Should have the same size as p
    divergence: "kl", compute KL(p, q).
    
    Returns
    -------
    div: float
    """
    # By convention: 0 log 0 = 0
    
    entropy = torch.nan_to_num(
        p * (p / q).log(), nan=0.0, posinf=0.0, neginf=0.0
    ).sum()
    
    return entropy - p.sum() + q.sum()

def compute_quad_kl(mu, nu, alpha, beta):
    """
    Calculate the KL divergence between two product measures:
    KL(mu otimes nu, alpha otimes beta) =
    m_mu * KL(nu, beta)
    + m_nu * KL(mu, alpha)
    + (m_mu - m_alpha) * (m_nu - m_beta)

    Parameters
    ----------
    mu: torch tensor
    nu: torch tensor
    alpha: torch tensor
        Should have the same size as mu
    beta: torch tensor
        Should have the same size as nu

    Returns
    ----------
    kl: float
        KL divergence between two product measures
    """

    m_mu = mu.sum()
    m_nu = nu.sum()
    m_alpha = alpha.sum()
    m_beta = beta.sum()
    const = (m_mu - m_alpha) * (m_nu - m_beta)
    kl = m_nu * compute_divergence(mu, alpha) + m_mu * compute_divergence(nu, beta) + const

    return kl