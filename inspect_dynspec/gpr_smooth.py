import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky
import sys
import numpy as np

jax.config.update("jax_enable_x64", True)
from typing import Optional, Tuple

#Clean console callbacks for JAX loops
def _print_cg_progress(k, g, d, c, p):
    g_val = np.mean(g)
    d_val = np.mean(d)
    c_val = np.max(c)
    k_val = np.max(k)
    
    # \r goes to start of line, \033[K clears the rest of the line
    sys.stdout.write(f"\r\033[KCG Progress: Iter {int(k_val)} | Mean Resid^2={g_val:.2e} | Mean delta_2={d_val:.2e} | Max Patience={int(c_val)}/{int(p)}")
    sys.stdout.flush()

def _print_cg_done(k, g, c):
    sys.stdout.write(f"\r\033[KCG Finished: Iter {int(np.max(k))} | Final Mean Resid^2={np.mean(g):.2e} | Stagnated? {bool(np.max(c))}\n")
    sys.stdout.flush()


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

def custom_cg(
    A, 
    b, 
    x0=None, 
    tol=1e-1, 
    maxiter=None,
    patience=10, # Number of times delta_2 < tol**2 to stop
    print_rate=50
):
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if maxiter is None:
        maxiter = min(10 * b.size, 20000)

    r0 = b - A(x0)
    p0 = r0
    gamma0 = jnp.vdot(r0, r0).real
    
    # state: (x, r, p, gamma, k, delta_2, counter)
    init_state = (x0, r0, p0, gamma0, 0, 1.0, 0)

    def cond_fun(state):
        _, _, _, _, k, _, count = state
        # Continue if we haven't hit maxiter AND haven't hit our patience limit
        return (k < maxiter) & (count < patience)

    def body_fun(state):
        x, r, p, gamma, k, _, count = state
        
        Ap = A(p)
        alpha = gamma / (jnp.vdot(p, Ap).real + 1e-15)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        
        gamma_new = jnp.vdot(r_new, r_new).real
        beta = gamma_new / (gamma + 1e-15)
        p_new = r_new + beta * p

        # Calculate squared relative step change
        x_diff = x_new - x
        delta_2 = jnp.vdot(x_diff, x_diff).real / (jnp.vdot(x_new, x_new).real + 1e-15)
        
        # Increment counter if delta_2 is below tolerance, else reset it to 0
        new_count = jnp.where(delta_2 < tol**2, count + 1, 0)
        
        def do_print():
            jax.debug.callback(_print_cg_progress, k, gamma_new, delta_2, new_count, patience)
        
        jax.lax.cond(k % print_rate == 0, do_print, lambda: None)
        
        return (x_new, r_new, p_new, gamma_new, k + 1, delta_2, new_count)

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    
    x_final, _, _, gamma_final, k_final, _, count_final = final_state
    
    # We consider it converged if the patience counter was the reason we stopped
    converged = count_final >= patience
    
    jax.debug.callback(_print_cg_done, k_final, gamma_final, converged)

    return x_final

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

def test_forward_backward(R : Mask, Ls : Tuple) -> None:

    key = jax.random.PRNGKey(31)
    k1, k2 = jax.random.split(key,2)

    Nv, Nt = R.shape
    n_obs = R.n_obs

    x = jax.random.normal(k1, (Nv, Nt)) # has shape of data "latent space"
    y = jax.random.normal(k2, (n_obs,)) # has length of "data space" (unmasked values)

    Lx = kron_mv(Ls, x)
    Rx = R.forward(Lx)

    Ls_T = [A.T for A in Ls]
    RTy = R.adjoint(y)
    LTRTy = kron_mv(Ls_T, RTy)

    y1 = jnp.vdot(y,Rx)
    x2 = jnp.vdot(x,LTRTy)

    diff = jnp.abs(y1 - x2)

    if diff <= 1e-10:
        jax.debug.print(f"got diff: {diff}, passed")
    else:
        jax.debug.print(f"got diff: {diff}, failed")


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
    cg_patience: int = 5,
    cg_maxiter: Optional[int] = None,
    nof_weight_samples: int = 20,
    test : bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    2D Gaussian‐Process smoothing via an RBF kernel with optional heteroscedastic noise.

    Args:
      data     : float[Nv, Nt]            — observed brightness.
      mask     : float[Nv, Nt]            — per-pixel mask (1 for valid pixels, 0 for invalid).
      weights  : float[Nv, Nt]            — per-pixel noise-precision (1/sigma^2). If None, uniform weights are used.
      l_length_nu : float               — RBF length-scale for frequency axis.
      l_length_t : float                — RBF length-scale for time axis.
      sigma2   : float                  — RBF signal variance sigma^2.
      jitter   : float                  — small value added to diagonal for numerical stability.
      cg_tol   : float                  — CG solver tolerance.
      cg_patience : int                 — Number of times that the cg step may be smaller than the tolerance in a row
      cg_maxiter: Optional[int]         — maximum number of CG iterations.
      nof_weight_samples: int           — number of latent samples to estimate posterior variance.
      test: bool                        — if True, returns additional intermediate values for testing purposes.

    Returns:
      float[Nv, Nt] — posterior mean (“smoothed”) image.
      float[Nsamp, Nv, Nt] — posterior samples in the original space, used to estimate variance. Nsamp = nof_weight_samples.
    """

    # Prep:
    Nv, Nt = data.shape
    t_grid = jnp.linspace(0, 1, Nt, dtype=jnp.float64)
    v_grid = jnp.linspace(0, 1, Nv, dtype=jnp.float64)
    R = Mask(mask)

    weights = jnp.ones_like(data) if weights is None else weights

    y_obs = R.forward(data)  # observed data
    prec_obs = R.forward(weights)  # observed precisions - should have no entities with zero weight

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

    if test:
        test_forward_backward(R,Ls)

    # (I + Lᵀ Rᵀ Σ⁻¹ R L) * Eta = Lᵀ Rᵀ Σ⁻¹ * data
    # A * Eta = b:
    # A = (I + Lᵀ Rᵀ Σ⁻¹ R L) is the operator we want to apply
    # However now Σ⁻¹ = diag(prec_flat) here (heteroscedastic noise),
    A_matvec = make_A_matvec(Ls, R, weights)

    b = kron_mv(Ls_T, RT_Sinv_y)

    # Conjugate gradient solver for Ax = b -> |Ax - b| < tol
    z0 = jnp.zeros((Nv, Nt), dtype=jnp.float64)
    Eta_map = custom_cg(A_matvec, b, x0=z0, tol=cg_tol, patience=cg_patience, maxiter=cg_maxiter)

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
        eta = jax.random.normal(key_eta, shape=(Nv, Nt))

        # 5) φ = ψ + η
        phi = psi + eta

        # 6) solve A ξ = φ by CG
        xi = custom_cg(A_matvec, phi, tol=cg_tol, patience=cg_patience, maxiter=cg_maxiter)

        return xi

    # nof samples to estimate diag(A⁻¹)?
    keys = jax.random.split(jax.random.PRNGKey(0), nof_weight_samples)

    # draw all latent samples
    xis = jax.vmap(sample_latent, in_axes=(0,))(keys)
    x_n = jax.vmap(lambda xi: kron_mv(Ls, xi))(
        xis
    )  # for each sample, map xi through L to get sample in original space

    return x_map, x_n
    