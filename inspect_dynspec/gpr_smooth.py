import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky
from jax.scipy.sparse.linalg import cg

jax.config.update("jax_enable_x64", True)
from typing import Optional, Tuple


# Taken from Quartical and written by Landman Bester
# https://github.com/ratt-ru/QuartiCal/blob/1fc6e5ff61365ef4164be209a970bdb4483703b0/quartical/utils/maths.py#L70
def fit_hyperplane(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Approximate a surface by a hyperplane in D dimensions

    inputs:
        x - D x N array of coordinates.
        y - N array of (possibly noisy) observations.
            Can be complex valued.

    outputs:
        theta - a vector of coefficients suct that X.T.dot(theta)
                is the hyperplane approximation of y and X is x
                with a row of ones appended as the final axis
    """
    D, N = x.shape
    y = y.squeeze()[None, :]
    z = jnp.vstack((x, y))
    centroid = jnp.zeros((D + 1, 1), dtype=y.dtype)
    for d in range(D + 1):
        if d < D:
            centroid = centroid.at[d, 0].set(jnp.sum(x[d]) / N)
        else:
            centroid = centroid.at[d, 0].set(jnp.sum(y) / N)
    diff = z - centroid
    cov = diff.dot(diff.conj().T)
    s, V = jnp.linalg.eigh(cov)
    n = V[:, 0].conj()  # defines normal to the plane
    theta = jnp.zeros(D + 1, dtype=y.dtype)
    for d in range(D + 1):
        if d < D:
            theta = theta.at[d].set(-n[d] / n[-1])
        else:
            # we need to take the mean here because y can be noisy
            # i.e. we do not have a point exactly in the plane
            theta = theta.at[d].set(jnp.mean(n[None, 0:-1].dot(x) / n[-1] + y))
    return theta


def kron_mv(Ls : tuple, z : jnp.ndarray) -> jnp.ndarray:
    """
    Generalized Kronecker matvec for JAX, matching the utils.py kron_matvec logic.
    Ls: list of matrices (e.g. [Lv, Lt])
    z: flattened input vector
    """
    x = z
    for A in Ls:
        Gd = A.shape[0]
        NGd = x.size // Gd
        X = x.reshape(Gd, NGd)
        Z = A @ X
        x = Z.T.ravel()
    return x.reshape(z.shape)


def rbf_kernel(grid : jnp.ndarray, lengthscale: float, variance: float=1.0) -> jnp.ndarray:
    """
    RBF kernel K_ij = variance * exp(-0.5 * (xi - xj)^2 / lengthscale^2)
    """
    d2 = jnp.subtract.outer(grid, grid) ** 2
    return variance * jnp.exp(-0.5 * d2 / lengthscale**2)


class Mask:
    def __init__(self, mask: jnp.ndarray):
        """
        A JAX-compatible mask operator.
        mask: boolean array of shape (nx, ny), True for observed pixels.
        """
        self.shape = mask.shape
        # Flattened boolean mask
        self.mask_flat = mask.ravel()
        # Precompute number of observations
        self.n_obs = int(self.mask_flat.sum())

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        R @ x: pick out observed pixels based on the mask
        x can be shape (nx, ny) or flattened (nx*ny,) - costs nothing to flatten if already flattened
        """
        x_flat = x.ravel()
        return x_flat[self.mask_flat]  # shape (n_obs,)

    def adjoint(self, y_obs: jnp.ndarray) -> jnp.ndarray:
        """
        R_T @ y_obs: scatter residuals back into full image.
        Returns array of shape (nx, ny,).
        """
        # start with zeros in flattened space
        full = jnp.zeros(self.mask_flat.shape)
        # scatter observed values back
        full = full.at[self.mask_flat].set(y_obs)
        return full.reshape(self.shape)


def make_A_matvec(Ls : tuple, mask : Mask, prec_flat : jnp.ndarray) -> callable:
    """
    Returns a function A_matvec(z) that computes
      (I + Lᵀ Rᵀ diag(prec_flat) R L) z
    where R is the mask operator and Σ⁻¹ = diag(prec_flat).
    Ls: list of matrices (e.g. [Lv, Lt])
    """

    @jax.jit
    def A_matvec(z):
        # 1) x = kron_mv(Ls, z)
        x_flat = kron_mv(Ls, z)
        # 2) Rx = mask.forward(x_flat)
        Rx = mask.forward(x_flat)
        # 3) Σ⁻¹ Rx  ← use per-pixel precisions at observed locs
        prec_obs = mask.forward(prec_flat)
        Sinv_Rx = Rx * prec_obs
        # 4) Rᵀ (Σ⁻¹ Rx)
        RT_Sinv_Rx = mask.adjoint(Sinv_Rx)
        # 5) Lᵀ term:
        LT_term = kron_mv([A.T for A in Ls], RT_Sinv_Rx)
        # 6) Return (I + …) z
        return z + LT_term

    return A_matvec


def gpr_smooth(
    data: jnp.ndarray,  # shape (Nv, Nt)
    mask: jnp.ndarray,  # shape (Nv, Nt), bool or {0,1}
    weights: Optional[
        jnp.ndarray
    ] = None,  # shape (Nv, Nt), per-pixel noise-precision (1/σ²)
    l_length_nu: float = 1.0,
    l_length_t: float = 1.0,
    sigma2: float = 1.0,
    jitter: float = 1e-6,
    cg_tol: float = 1e-6,
    cg_maxiter: int = 200,
    nof_weight_samples: int = 20,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    2D Gaussian‐Process smoothing via an RBF kernel with optional heteroscedastic noise.

    Args:
      data     : float[H, W]  — observed brightness.
      mask     : float[H, W]  — per-pixel mask (1 for valid pixels, 0 for invalid).
      weights  : float[H, W]  — per-pixel noise-precision (1/sigma^2). If None, uniform weights are used.
      l_length_nu : float     — RBF length-scale for frequency axis.
      l_length_t : float      — RBF length-scale for time axis.
      sigma2   : float        — RBF signal variance sigma^2.
      jitter   : float        — small value added to diagonal for numerical stability.
      cg_tol   : float        — CG solver tolerance.
      cg_maxiter: int         — maximum number of CG iterations.
      nof_weight_samples: int — number of latent samples to estimate posterior variance.

    Returns:
      float[H, W] — posterior mean (“smoothed”) image.
      float[H, W] — posterior variance image.
    """

    # Prep:
    Nv, Nt = data.shape
    t_grid = jnp.linspace(0, 1, Nt, dtype=jnp.float64)
    v_grid = jnp.linspace(0, 1, Nv, dtype=jnp.float64)
    R = Mask(mask)

    weights = jnp.ones_like(data) if weights is None else weights

    data_flat = data.ravel()
    prec_flat = weights.ravel()

    y_obs = R.forward(data_flat)  # observed data
    prec_obs = R.forward(prec_flat)  # observed precisions - should have no entities with zero weight

    # Σ = diag(1/prec_flat)  ⇒  σ_obs = 1/√prec_obs
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    y_weighted = y_obs * prec_obs  # weighted observations
    RT_Sinv_y = R.adjoint(y_weighted)  # back to full grid

    # Define the covariance kernels for nu and t:
    Kt = rbf_kernel(t_grid, lengthscale=l_length_t, variance=sigma2)
    Kv = rbf_kernel(v_grid, lengthscale=l_length_nu, variance=1.0)

    # LᵀL = K so then we can compute Cholesky factors
    Lt = cholesky(Kt + jitter * jnp.eye(Nt, dtype=jnp.float64), lower=True)
    Lv = cholesky(Kv + jitter * jnp.eye(Nv, dtype=jnp.float64), lower=True)
    Ls = (Lv, Lt)
    Ls_T = tuple(A.T for A in Ls)

    # (I + Lᵀ Rᵀ Σ⁻¹ R L) * Eta = Lᵀ Rᵀ Σ⁻¹ * data
    # A * Eta = b:
    # A = (I + Lᵀ Rᵀ Σ⁻¹ R L) is the operator we want to apply
    # However now Σ⁻¹ = diag(prec_flat) here (heteroscedastic noise),
    A_matvec = make_A_matvec(Ls, R, prec_flat)

    b = kron_mv(Ls_T, RT_Sinv_y)

    # Conjugate gradient solver for Ax = b -> |Ax - b| < tol
    z0 = jnp.zeros(Nv * Nt, dtype=jnp.float64)
    Eta_map, info = cg(A_matvec, b, x0=z0, tol=cg_tol, maxiter=cg_maxiter)
    if info != 0:
        print("Jax CG result, info =", info)

    # finally we just need to compute x = kron_mv(Ls, Eta_map)
    x_map_flat = kron_mv(Ls, Eta_map)
    # and reshape to the original image shape
    x_map = x_map_flat.reshape(Nv, Nt)

    def sample_latent(key : jax.random.PRNGKey) -> jnp.ndarray:
        """
        Sample from the latent distribution:
            ξ ∼ N(0, A⁻¹)
        by solving A ξ = φ where φ = ψ + η
        with ψ = Lᵀ Rᵀ ε_obs, ε_obs ∼ N(0, σ_obs²)
        and η ∼ N(0, I)
        """
        # split for noise vs η
        key_n, key_eta = jax.random.split(key)

        # 1) draw ε_obs ∼ N(0, σ_obs²)
        eps_obs = jax.random.normal(key_n, shape=(R.n_obs,)) * sigma_obs

        # 2) embed into full grid: n_full = Rᵀ ε_obs
        n_full = R.adjoint(eps_obs)  # shape (Nv*Nt,)

        # 3) ψ = Lᵀ Rᵀ ε_obs  = kron_mv([A.T for A in Ls[::-1]], n_full)
        psi = kron_mv([A.T for A in Ls], n_full)

        # 4) η ∼ N(0, I) in latent space
        eta = jax.random.normal(key_eta, shape=(Nv * Nt,))

        # 5) φ = ψ + η
        phi = psi + eta

        # 6) solve A ξ = φ by CG
        xi, _ = cg(A_matvec, phi, tol=cg_tol, maxiter=cg_maxiter)

        return xi

    # nof samples to estimate diag(A⁻¹)?
    keys = jax.random.split(jax.random.PRNGKey(0), nof_weight_samples)

    # draw all latent samples
    xis = jax.vmap(sample_latent, in_axes=(0,))(keys)
    x_n = jax.vmap(lambda xi: kron_mv(Ls, xi))(
        xis
    )  # for each sample, map xi through L to get sample in original space

    var_x = jnp.var(x_n, axis=0).reshape(
        Nv, Nt
    )  # compute var across nof_weight_samples

    return x_map, var_x, xis
