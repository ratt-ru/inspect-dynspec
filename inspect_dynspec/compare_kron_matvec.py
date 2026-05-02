import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# --- Numpy version (from utils.py/gpr_smooth.py) ---
def kron_matvec_numpy(A, b):
    D = len(A)
    N = b.size
    x = b
    for d in range(D):
        Gd = A[d].shape[0]
        NGd = N // Gd
        X = np.reshape(x, (Gd, NGd))
        Z = A[d].dot(X).T
        x = Z.ravel()
    return x.reshape(b.shape)

# --- JAX version (from gpr_smooth_hetero.py) ---
def kron_mv_jax(Lt, Lv, z):
    # Emulate the numpy kron_matvec_numpy logic
    x = z
    for A in [Lv, Lt]:
        Gd = A.shape[0]
        NGd = x.size // Gd
        X = x.reshape(Gd, NGd)
        Z = A @ X
        x = Z.T.ravel()
    return x.reshape(z.shape)

# --- Test data ---
np.random.seed(42)
Nv, Nt = 5, 4
z_np = np.random.randn(Nv * Nt)
Lt_np = np.random.randn(Nt, Nt)
Lv_np = np.random.randn(Nv, Nv)

z_jax = jnp.array(z_np)
Lt_jax = jnp.array(Lt_np)
Lv_jax = jnp.array(Lv_np)

# --- Run both versions ---
res_numpy = kron_matvec_numpy([Lv_np, Lt_np], z_np)
res_jax = kron_mv_jax(Lt_jax, Lv_jax, z_jax)

# --- Compare outputs ---
print("Numpy result:", res_numpy)
print("JAX result:  ", np.array(res_jax))
print("Difference:  ", np.abs(res_numpy - np.array(res_jax)).max())

# --- Plot intermediate steps ---
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(z_np.reshape(Nv, Nt), aspect='auto')
axs[0, 0].set_title('Input z (Nv x Nt)')
axs[0, 1].imshow(res_numpy.reshape(Nv, Nt), aspect='auto')
axs[0, 1].set_title('Numpy kron_matvec result')
axs[1, 0].imshow(np.array(res_jax).reshape(Nv, Nt), aspect='auto')
axs[1, 0].set_title('JAX kron_mv result')
axs[1, 1].imshow(np.abs(res_numpy - np.array(res_jax)).reshape(Nv, Nt), aspect='auto')
axs[1, 1].set_title('Absolute difference')
plt.tight_layout()
plt.show()
