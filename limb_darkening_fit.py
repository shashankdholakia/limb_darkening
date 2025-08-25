import jax
from jax import config
config.update("jax_enable_x64", True)

from jaxoplanet.experimental.limb_dark_poly_coeffs import calc_poly_coeffs

import jax.numpy as jnp

def four_param_limb_dark(mu, coeffs):
    """
    Four-parameter nonlinear limb darkening law (Claret 2000).

    The stellar intensity profile is given by:

        I(mu) / I(mu=1) = 1
                          - c1 * (1 - mu**0.5)
                          - c2 * (1 - mu)
                          - c3 * (1 - mu**1.5)
                          - c4 * (1 - mu**2)

    where mu = cos(theta) is the cosine of the angle between the
    line of sight and the stellar surface normal.

    Parameters
    ----------
    mu : array-like
        Cosine of the angle between the line of sight and the surface normal.
        Must satisfy 0 <= mu <= 1.
    coeffs : array-like of shape (4,)
        Limb-darkening coefficients [c1, c2, c3, c4].

    Returns
    -------
    I_norm : array-like
        Normalized stellar intensity at each value of mu, such that I(mu=1) = 1.

    References
    ----------
    Claret, A. (2000), "A new non-linear limb-darkening law for LTE stellar
    atmosphere models: Calculations for -5.0 <= log[M/H] <= +1,
    2000 K <= Teff <= 50000 K at several surface gravities".
    A&A, 363, 1081–1190.
    """
    c1, c2, c3, c4 = coeffs
    return (
        1
        - c1 * (1 - mu**0.5)
        - c2 * (1 - mu)
        - c3 * (1 - mu**1.5)
        - c4 * (1 - mu**2)
    )
    

def sqrt_limb_dark(mu, coeffs):
    """
    Square-root limb darkening law:
    I(mu) / I(mu=1) = 1 - c1*(1 - mu) - c2*(1 - sqrt(mu))

    Parameters
    ----------
    mu : array-like
        Cosine of the angle between line of sight and surface normal (0 <= mu <= 1).
    c1, c2 : float
        Limb-darkening coefficients.

    Returns
    -------
    I_norm : array-like
        Normalized intensity at each mu.
    """
    c1, c2 = coeffs
    
    return 1.0 - c1 * (1.0 - mu) - c2 * (1.0 - jnp.sqrt(mu))


def log_limb_dark(mu, coeffs, eps=1e-12):
    """
    Logarithmic limb darkening law:
    I(mu) / I(mu=1) = 1 - c1 * (1 - mu) - c2 * mu * log(mu)

    Parameters
    ----------
    mu : array-like
        Cosine of the angle between line of sight and surface normal (0 <= mu <= 1).
    c1, c2 : float
        Limb-darkening coefficients.
    eps : float, optional
        Small value to avoid log(0). Default is 1e-12.

    Returns
    -------
    I_norm : array-like
        Normalized intensity at each mu.
    """
    c1, c2 = coeffs
    mu_safe = jnp.clip(mu, eps, 1.0)  # keep inside [eps,1]
    return 1.0 - c1 * (1.0 - mu_safe) - c2 * mu_safe * jnp.log(mu_safe)

 