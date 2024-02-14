import torch

def init_test_distribution(n_features, n_points):
    """Initializes test distribution"""
    mean = torch.normal(0, 3, size=(n_features,))

    # Generate random covariance matrix from Wishart distribution
    m = torch.distributions.wishart.Wishart(df=torch.tensor(n_features), covariance_matrix=torch.eye(n_features))
    cov = m.sample()

    # Generate random multivariate normal
    m = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
    features = torch.stack([m.sample() for _ in range(n_points)]).T
    embeddings = torch.rand(n_points, 3)
    geometry = torch.cdist(embeddings, embeddings)

    return features, geometry, embeddings


def compute_unnormalized_kl(p, q):
    """Compute unnormalized Kullback-Leibler divergence between two vectors"""
    # By convention: 0 log 0 = 0
    entropy = torch.nan_to_num(
        p * (p / q).log(), nan=0.0, posinf=0.0, neginf=0.0
    ).sum()
    return entropy


def compute_divergence(p, q):
    """Compute Kullback-Leibler divergence between two vectors p and q"""
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
    """

    m_mu = mu.sum()
    m_nu = nu.sum()
    m_alpha = alpha.sum()
    m_beta = beta.sum()
    const = (m_mu - m_alpha) * (m_nu - m_beta)
    kl = m_nu * compute_divergence(mu, alpha) + m_mu * compute_divergence(nu, beta) + const

    return kl