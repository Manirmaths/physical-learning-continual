# src/train_single.py
from __future__ import annotations

import jax
import jax.numpy as jnp

from network import solve_dirichlet, softplus_weights

print("RUNNING train_single.py from:", __file__, flush=True)

# -----------------------------
# Static (hashable) graph + IO
# -----------------------------
N_NODES = 8
EDGES = (
    (0, 4), (1, 4), (4, 5), (5, 6), (5, 7), (2, 6), (3, 7)
)
IN_NODES = (0, 1)     # Dirichlet inputs
OUT_NODES = (6, 7)    # readout nodes


def make_toy_data(key: jax.random.PRNGKey, n_samples: int = 256):
    X = jax.random.uniform(key, (n_samples, 2), minval=-1.0, maxval=1.0)
    Y = jnp.stack(
        [
            0.8 * X[:, 0] - 0.2 * X[:, 1],
            -0.1 * X[:, 0] + 0.9 * X[:, 1],
        ],
        axis=1,
    )
    return X, Y


def predict(w_raw: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    w = softplus_weights(w_raw)
    out_nodes = jnp.array(OUT_NODES, dtype=int)

    def one_example(x):
        v = solve_dirichlet(
            n_nodes=N_NODES,
            edges=EDGES,          # tuple is fine
            w=w,
            fixed_nodes=IN_NODES, # IMPORTANT: tuple, not jnp array
            fixed_values=x,
        )
        return v[out_nodes]

    return jax.vmap(one_example)(X)


def loss_fn(w_raw: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, l2: float) -> jnp.ndarray:
    Yhat = predict(w_raw, X)
    mse = jnp.mean((Yhat - Y) ** 2)
    w = softplus_weights(w_raw)
    reg = l2 * jnp.mean(w)
    return mse + reg


@jax.jit
def step(w_raw: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, lr: float, l2: float):
    loss, grad = jax.value_and_grad(loss_fn)(w_raw, X, Y, l2)
    return w_raw - lr * grad, loss


def main():
    print("Entered main()", flush=True)

    key = jax.random.PRNGKey(0)
    X, Y = make_toy_data(key, n_samples=256)

    m = len(EDGES)
    w_raw = 0.1 * jax.random.normal(key, (m,))

    lr = 5e-2
    l2 = 1e-4
    n_steps = 300

    # Force at least one visible print before JIT compilation
    print("Starting training loop...", flush=True)

    for t in range(n_steps):
        w_raw, loss = step(w_raw, X, Y, lr, l2)
        if t % 50 == 0:
            print(f"[Task A] iter {t:4d} | loss {loss:.4e}", flush=True)

    mse = jnp.mean((predict(w_raw, X) - Y) ** 2)
    print(f"[Task A] final MSE: {mse:.4e}", flush=True)

    jnp.save("w_taskA.npy", w_raw)
    print("Saved: w_taskA.npy", flush=True)


if __name__ == "__main__":
    main()