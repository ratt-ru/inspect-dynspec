import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky
from jax.scipy.sparse.linalg import cg
import numpy as np

def kron_mv(Ls, z):
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


def kron_mtv(Lt, Lv, w):
    """
    Compute (Lt.T ⊗ Lv.T) @ w efficiently.
    """
    Nt = Lt.shape[0]
    Nv = Lv.shape[0]
    W = w.reshape(Nv, Nt)
    Y = Lv.T @ W @ Lt  # shape (Nv, Nt)
    return Y.ravel()


def rbf_kernel(grid, lengthscale):
    d2 = jnp.subtract.outer(grid, grid) ** 2
    return jnp.exp(-0.5 * d2 / lengthscale**2)


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
        R @ x: pick out observed pixels.
        x can be shape (nx, ny) or flattened (nx * ny,)
        """
        x_flat = x.ravel()
        return x_flat[self.mask_flat]  # shape (n_obs,)

    def adjoint(self, y_obs: jnp.ndarray) -> jnp.ndarray:
        """
        R_T @ y_obs: scatter residuals back into full image.
        Returns array of shape (nx * ny,), which you can reshape to (nx, ny).
        """
        # start with zeros in flattened space
        full = jnp.zeros(self.mask_flat.shape)
        # scatter observed values back
        full = full.at[self.mask_flat].set(y_obs)
        return full  # still flattened

    def adjoint_image(self, y_obs: jnp.ndarray) -> jnp.ndarray:
        """
        Convenience: same as adjoint but reshaped to (nx, ny).
        """
        return self.adjoint(y_obs).reshape(self.shape)


def make_A_matvec(Ls, mask, sigma2):
    """
    Returns a function A_matvec(z) that computes
      (I + Lᵀ Rᵀ Σ⁻¹ R L) z
    where R is the mask operator.
    Ls: list of matrices (e.g. [Lv, Lt])
    """
    # JIT-compile the inner function for speed
    @jax.jit
    def A_matvec(z):
        # 1) x = (L_t ⊗ L_v) @ z
        x_flat = kron_mv(Ls, z)

        # 2) Rx = mask.forward(x_flat)
        Rx = mask.forward(x_flat)

        # 3) Σ⁻¹ Rx
        Sinv_Rx = Rx / sigma2

        # 4) Rᵀ (Σ⁻¹ Rx)
        RT_Sinv_Rx = mask.adjoint(Sinv_Rx)

        # 5) Lᵀ term: (Lᵀ_t ⊗ Lᵀ_v) @ RT_Sinv_Rx
        LT_term = kron_mv([A.T for A in Ls[::-1]], RT_Sinv_Rx)

        # 6) Return (I + …) z
        return z + LT_term

    return A_matvec


def gpr_smooth(
    data: np.ndarray,
    mask: np.ndarray,
    l_length_nu: float = 1.0,
    l_length_t: float = 1.0,
    sigma2: float = 1.0,
    jitter:      float = 1e-6,
    cg_tol:      float = 1e-6,
    cg_maxiter:  int   = 200
) -> np.ndarray:
    """
    2D Gaussian‐Process smoothing via an RBF kernel.

    Args:
      data     : float[H, W]  — observed brightness.
      mask     : float[H, W]  — per-pixel mask (1 for valid pixels, 0 for invalid).
      l_length_nu : float     — RBF length-scale for frequency axis.
      l_length_t : float      — RBF length-scale for time axis.
      sigma2   : float        — RBF signal variance sigma^2.
      jitter   : float        — small value added to diagonal for numerical stability.
      cg_tol   : float        — CG solver tolerance.
      cg_maxiter: int         — maximum number of CG iterations.

    Returns:
      float[H, W] — posterior mean (“smoothed”) image.
    """
    # Prep:
    Nt = data.shape[1]
    Nv = data.shape[0]
    t_grid = jnp.linspace(0, 1, Nt)
    v_grid = jnp.linspace(0, 1, Nv)
    R = Mask(mask)
    Sinv_y = data / sigma2
    Sinv_y_obs = R.forward(Sinv_y)
    RT_Sinv_y = R.adjoint(Sinv_y_obs)

    # Define the covariance kernels for nu and t:
    K_t = rbf_kernel(t_grid, lengthscale=l_length_t)
    K_v = rbf_kernel(v_grid, lengthscale=l_length_nu)

    # LᵀL = K so then we can compute Cholesky factors
    L_t = cholesky(K_t + jitter * jnp.eye(Nt), lower=True)
    L_v = cholesky(K_v + jitter * jnp.eye(Nv), lower=True)
    Ls = [L_v, L_t]

    # (I+Lᵀ Rᵀ Σ⁻¹ R L) * Eta = Lᵀ Rᵀ Σ⁻¹ * data
    # A * Eta = b:
    # A = (I+Lᵀ Rᵀ Σ⁻¹ R L) is the operator we want to apply
    A_matvec = make_A_matvec(Ls, R, sigma2)

    # b = kron_mv([A.T for A in Ls[::-1]], RT_Sinv_y)
    b = kron_mv([A.T for A in Ls[::-1]], RT_Sinv_y)

    # Conjugate gradient solver for Ax = b -> |Ax - b| < tol
    z0 = jnp.zeros(Nv * Nt)  # our objective
    Eta_map, info = cg(A_matvec, b, x0=z0, tol=cg_tol, maxiter=cg_maxiter)
    if info != 0:
        print("Jax CG result, info =", info)

    # finally we just need to compute x = kron_mv(Ls, Eta_map)
    x_map_flat = kron_mv(Ls, Eta_map)
    # and reshape to the original image shape
    x_map = x_map_flat.reshape(Nv, Nt)

    return x_map