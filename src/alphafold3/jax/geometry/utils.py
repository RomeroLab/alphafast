# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Utils for geometry library."""

from collections.abc import Iterable
import numbers

import jax
from jax import lax
import jax.numpy as jnp


def safe_select(condition, true_fn, false_fn):
  """Safe version of selection (i.e. `where`).

  This applies the double-where trick.
  Like jnp.where, this function will still execute both branches and is
  expected to be more lightweight than lax.cond.  Other than NaN-semantics,
  safe_select(condition, true_fn, false_fn) is equivalent to

    jax.tree.map(lambda x, y: jnp.where(condition, x, y),
                 true_fn(),
                 false_fn()),

  Compared to the naive implementation above, safe_select provides the
  following guarantee: in either the forward or backward pass, a NaN produced
  *during the execution of true_fn()* will not propagate to the rest of the
  computation and similarly for false_fn.  It is very important to note that
  while true_fn and false_fn will typically close over other tensors (i.e. they
  use values computed prior to the safe_select function), there is no NaN-safety
  for the backward pass of closed over values.  It is important than any NaN's
  are produced within the branch functions and not before them.  For example,

    safe_select(x < eps, lambda: 0., lambda: jnp.sqrt(x))

  will not produce NaN on the backward pass even if x == 0. since sqrt happens
  within the false_fn, but the very similar

    y = jnp.sqrt(x)
    safe_select(x < eps, lambda: 0., lambda: y)

  will produce a NaN on the backward pass if x == 0 because the sqrt happens
  prior to the false_fn.

  Args:
    condition: Boolean array to use in where
    true_fn: Zero-argument function to construct the values used in the True
      condition.  Tensors that this function closes over will be extracted
      automatically to implement the double-where trick to suppress spurious NaN
      propagation.
    false_fn: False branch equivalent of true_fn

  Returns:
    Resulting PyTree equivalent to tree_map line above.
  """
  true_fn, true_args = jax.closure_convert(true_fn)
  false_fn, false_args = jax.closure_convert(false_fn)

  true_args = jax.tree.map(
      lambda x: jnp.where(condition, x, lax.stop_gradient(x)), true_args
  )

  false_args = jax.tree.map(
      lambda x: jnp.where(condition, lax.stop_gradient(x), x), false_args
  )

  return jax.tree.map(
      lambda x, y: jnp.where(condition, x, y),
      true_fn(*true_args),
      false_fn(*false_args),
  )


def unstack(value: jnp.ndarray, axis: int = -1) -> list[jnp.ndarray]:
  return [
      jnp.squeeze(v, axis=axis)
      for v in jnp.split(value, value.shape[axis], axis=axis)
  ]


def angdiff(alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
  """Compute absolute difference between two angles."""
  d = alpha - beta
  d = (d + jnp.pi) % (2 * jnp.pi) - jnp.pi
  return d


def safe_arctan2(
    x1: jnp.ndarray, x2: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
  """Safe version of arctan2 that avoids NaN gradients when x1=x2=0."""

  return safe_select(
      jnp.abs(x1) + jnp.abs(x2) < eps,
      lambda: jnp.zeros_like(jnp.arctan2(x1, x2)),
      lambda: jnp.arctan2(x1, x2),
  )


def weighted_mean(
    *,
    weights: jnp.ndarray,
    value: jnp.ndarray,
    axis: int | Iterable[int] | None = None,
    eps: float = 1e-10,
) -> jnp.ndarray:
  """Computes weighted mean in a safe way that avoids NaNs.

  This is equivalent to jnp.average for the case eps=0.0, but adds a small
  constant to the denominator of the weighted average to avoid NaNs.
  'weights' should be broadcastable to the shape of value.

  Args:
    weights: Weights to weight value by.
    value: Values to average
    axis: Axes to average over.
    eps: Epsilon to add to the denominator.

  Returns:
    Weighted average.
  """

  weights = jnp.asarray(weights, dtype=value.dtype)
  weights = jnp.broadcast_to(weights, value.shape)

  weights_shape = weights.shape

  if isinstance(axis, numbers.Integral):
    axis = [axis]
  elif axis is None:
    axis = list(range(len(weights_shape)))

  return jnp.sum(weights * value, axis=axis) / (
      jnp.sum(weights, axis=axis) + eps
  )
