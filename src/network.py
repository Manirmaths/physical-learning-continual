# src/network.py
from __future__ import annotations

import jax
import jax.numpy as jnp

print("USING network.py from:", __file__, flush=True)


def build_laplacian(n_nodes: int, edges, w: jnp.ndarray) -> jnp.ndarray:
    w = jnp.asarray(w)
    L = jnp.zeros((n_nodes, n_nodes), dtype=w.dtype)
    for k, (i, j) in enumerate(edges):
        wk = w[k]
        L = L.at[i, i].add(wk)
        L = L.at[j, j].add(wk)
        L = L.at[i, j].add(-wk)
        L = L.at[j, i].add(-wk)
    return L


def solve_dirichlet(
    n_nodes: int,
    edges,
    w: jnp.ndarray,
    fixed_nodes,          # python tuple/list of ints
    fixed_values: jnp.ndarray,
    ridge: float = 1e-8,
) -> jnp.ndarray:
    # fixed_nodes handled in Python (no setops calls; JIT-safe)
    fixed_nodes = tuple(int(i) for i in fixed_nodes)
    if len(fixed_nodes) == 0:
        raise ValueError("Need at least one fixed node")

    fixed_values = jnp.asarray(fixed_values, dtype=w.dtype).ravel()
    if fixed_values.shape[0] != len(fixed_nodes):
        raise ValueError("fixed_values must match fixed_nodes length")

    fixed_set = set(fixed_nodes)
    free_nodes = tuple(i for i in range(n_nodes) if i not in fixed_set)

    fixed_idx = jnp.array(fixed_nodes, dtype=int)
    free_idx = jnp.array(free_nodes, dtype=int)

    L = build_laplacian(n_nodes, edges, w)

    v = jnp.zeros((n_nodes,), dtype=w.dtype)

    if len(free_nodes) == 0:
        return v.at[fixed_idx].set(fixed_values)

    L_ff = L[jnp.ix_(free_idx, free_idx)]
    L_fc = L[jnp.ix_(free_idx, fixed_idx)]
    rhs = -L_fc @ fixed_values

    v_free = jnp.linalg.solve(L_ff + ridge * jnp.eye(L_ff.shape[0], dtype=w.dtype), rhs)

    v = v.at[fixed_idx].set(fixed_values)
    v = v.at[free_idx].set(v_free)
    return v


def softplus_weights(w_raw: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    return jax.nn.softplus(w_raw) + eps