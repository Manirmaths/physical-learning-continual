# src/train_continual.py
from __future__ import annotations

import jax
import jax.numpy as jnp

from network import solve_dirichlet, softplus_weights

# -----------------------------
# Static (hashable) graph + IO
# -----------------------------
N_NODES = 8
EDGES = (
    (0, 4), (1, 4), (4, 5), (5, 6), (5, 7), (2, 6), (3, 7)
)
IN_NODES = (0, 1)     # Dirichlet inputs
OUT_NODES = (6, 7)    # readout nodes


def make_inputs(key, n_samples: int = 256):
    return jax.random.uniform(key, (n_samples, len(IN_NODES)), minval=-1.0, maxval=1.0)


def teacher_outputs(w_raw_star: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """Generate targets from the *same* physical network using teacher weights."""
    w_star = softplus_weights(w_raw_star)
    out_nodes = jnp.array(OUT_NODES, dtype=int)

    def one_example(x):
        v = solve_dirichlet(
            n_nodes=N_NODES,
            edges=EDGES,
            w=w_star,
            fixed_nodes=IN_NODES,
            fixed_values=x,
        )
        return v[out_nodes]

    return jax.vmap(one_example)(X)


def predict(w_raw: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    w = softplus_weights(w_raw)
    out_nodes = jnp.array(OUT_NODES, dtype=int)

    def one_example(x):
        v = solve_dirichlet(
            n_nodes=N_NODES,
            edges=EDGES,
            w=w,
            fixed_nodes=IN_NODES,
            fixed_values=x,
        )
        return v[out_nodes]

    return jax.vmap(one_example)(X)


def mse(w_raw: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    Yhat = predict(w_raw, X)
    return jnp.mean((Yhat - Y) ** 2)


def loss_fn(
    w_raw: jnp.ndarray,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    l2: float,
    stab_lambda: float,
    w_raw_anchor: jnp.ndarray,
) -> jnp.ndarray:
    """Main loss + mild size regularizer + optional stability penalty."""
    Yhat = predict(w_raw, X)
    data = jnp.mean((Yhat - Y) ** 2)

    w = softplus_weights(w_raw)
    reg = l2 * jnp.mean(w)

    # simple continual-learning stabilizer (L2 to anchor params)
    stab = stab_lambda * jnp.mean((w_raw - w_raw_anchor) ** 2)

    return data + reg + stab


@jax.jit
def step(
    w_raw: jnp.ndarray,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    lr: float,
    l2: float,
    stab_lambda: float,
    w_raw_anchor: jnp.ndarray,
):
    loss, grad = jax.value_and_grad(loss_fn)(
        w_raw, X, Y, l2, stab_lambda, w_raw_anchor
    )
    return w_raw - lr * grad, loss


def train_task(
    w_raw_init: jnp.ndarray,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    *,
    n_steps: int,
    lr: float,
    l2: float,
    stab_lambda: float = 0.0,
    w_raw_anchor: jnp.ndarray | None = None,
    label: str = "Task",
):
    if w_raw_anchor is None:
        w_raw_anchor = jnp.array(w_raw_init)

    w_raw = w_raw_init
    for t in range(n_steps):
        w_raw, loss = step(w_raw, X, Y, lr, l2, stab_lambda, w_raw_anchor)
        if t % 100 == 0:
            print(f"[{label}] iter {t:4d} | loss {loss:.4e}")
    return w_raw


def main():
    print("RUNNING train_continual.py", flush=True)
    key = jax.random.PRNGKey(0)

    # -----------------------------
    # Build two tasks (A then B)
    # -----------------------------
    key, kxA, kxB, kwA, kwB, kinit = jax.random.split(key, 6)

    X_A = make_inputs(kxA, n_samples=256)
    X_B = make_inputs(kxB, n_samples=256)

    m = len(EDGES)

    # Teacher parameters (different tasks)
    wA_star_raw = 0.7 * jax.random.normal(kwA, (m,))
    wB_star_raw = 0.7 * jax.random.normal(kwB, (m,))

    Y_A = teacher_outputs(wA_star_raw, X_A)
    Y_B = teacher_outputs(wB_star_raw, X_B)

    # Learner init
    w_raw0 = 0.1 * jax.random.normal(kinit, (m,))

    # -----------------------------
    # Train on Task A
    # -----------------------------
    lr = 1e-1
    l2 = 1e-4
    steps_A = 600

    print("\n=== Train Task A ===")
    wA = train_task(w_raw0, X_A, Y_A, n_steps=steps_A, lr=lr, l2=l2, label="Task A")
    mse_A_after_A = mse(wA, X_A, Y_A)
    print(f"[Task A] MSE after A training: {mse_A_after_A:.4e}")

    # -----------------------------
    # Train on Task B (no stabilizer) and measure forgetting on A
    # -----------------------------
    steps_B = 600
    print("\n=== Train Task B (no stabilizer) ===")
    wAB = train_task(wA, X_B, Y_B, n_steps=steps_B, lr=lr, l2=l2, label="Task B (free)")
    mse_B_after_B = mse(wAB, X_B, Y_B)
    mse_A_after_B = mse(wAB, X_A, Y_A)
    print(f"[Task B] MSE after B training: {mse_B_after_B:.4e}")
    print(f"[Task A] MSE after B training (forgetting): {mse_A_after_B:.4e}")
    print(f"Forgetting (A): {mse_A_after_B - mse_A_after_A:.4e}")

    # -----------------------------
    # Train on Task B WITH stability penalty to reduce forgetting
    # -----------------------------
    stab_lambda = 5e-2  # tune if needed
    print("\n=== Train Task B (with stability penalty) ===")
    wAB_stable = train_task(
        wA,
        X_B,
        Y_B,
        n_steps=steps_B,
        lr=lr,
        l2=l2,
        stab_lambda=stab_lambda,
        w_raw_anchor=wA,   # anchor = A solution
        label="Task B (stable)",
    )
    mse_B_after_Bs = mse(wAB_stable, X_B, Y_B)
    mse_A_after_Bs = mse(wAB_stable, X_A, Y_A)

    print(f"[Task B stable] MSE after B training: {mse_B_after_Bs:.4e}")
    print(f"[Task A stable] MSE after B training: {mse_A_after_Bs:.4e}")
    print(f"Forgetting (A, stable): {mse_A_after_Bs - mse_A_after_A:.4e}")

    # Save artifacts
    jnp.save("w_taskA.npy", wA)
    jnp.save("w_taskAB_free.npy", wAB)
    jnp.save("w_taskAB_stable.npy", wAB_stable)
    print("\nSaved: w_taskA.npy, w_taskAB_free.npy, w_taskAB_stable.npy")


if __name__ == "__main__":
    main()