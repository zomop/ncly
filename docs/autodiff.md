# Automatic differentiation in NCLY

NCLY implements both classical modes of automatic differentiation. They answer
different questions and the library exposes both, because the cost profile of one
is the weakness of the other.

| Mode | Computes | Cost | Best when |
| --- | --- | --- | --- |
| Forward | `J·v` for one direction `v` | one evaluation per direction | few inputs, many outputs |
| Reverse | `vᵀ·J` for one seed `v` | one forward + one backward pass | many inputs, one output |

## Forward mode: dual numbers

A dual number is an element of the ring `ℝ[ε]/(ε²)`:

$$
x = a + b\varepsilon, \qquad \varepsilon^2 = 0 .
$$

Because `ε² = 0`, every analytic function propagates a derivative for free:

$$
(a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\varepsilon
$$

so the product rule is *arithmetic*, not a special case:

```rust
impl Mul for Dual {
    fn mul(self, rhs: Self) -> Self {
        Self {
            value:   self.value * rhs.value,
            tangent: self.tangent * rhs.value + self.value * rhs.tangent,
        }
    }
}
```

Transcendental functions follow from `f(a + bε) = f(a) + b f'(a) ε`:

| Function | Tangent |
| --- | --- |
| `exp` | `exp(a) · b` |
| `ln` | `b / a` |
| `sqrt` | `b / (2√a)` |
| `sin` | `b · cos(a)` |
| `cos` | `−b · sin(a)` |
| `tanh` | `b · (1 − tanh²a)` |
| `xⁿ` | `b · n · a^{n−1}` |

### Jacobian-vector products

Seeding one variable with tangent `1` and all others with tangent `0` yields one
column of the Jacobian. Seeding with an arbitrary vector `v` yields `J·v`
directly — this is the forward-mode JVP:

```rust
use ncly_autodiff::Dual;

// f(x, y) = [x·y, sin(x) + y²]  at (0.8, 1.5)
let x = Dual::with_tangent(0.8, 1.0);   // direction e₁
let y = Dual::with_tangent(1.5, 0.0);
let jvp = [(x * y).tangent, (x.sin() + y * y).tangent];
// = [y, cos(x)] = [1.5, 0.6967…]
```

Dual numbers allocate nothing and compose with the rest of the crate, which makes
them the right tool for validating reverse-mode rules: they share no code with
the tape.

## Reverse mode: the tape

Reverse mode records a **Wengert list** — a linear sequence of primitives. Node
`i` is appended after all of its parents, so the tape index is a valid
topological order and the backward sweep is a single reverse iteration:

```text
node 0:  leaf x = 2                  parents ∅
node 1:  leaf y = 3                  parents ∅
node 2:  mul(0, 1) = 6               parents [(0, ∂/∂x = y), (1, ∂/∂y = x)]
node 3:  powi(2, 2) = 36             parents [(2, ∂/∂ = 12)]
node 4:  sum(3) = 36                 parents [(3, ∂/∂ = 1)]
```

The sweep sets `adjoint[4] = 1` and walks down:

$$
\bar{x}_j \mathrel{+}= \bar{x}_i \cdot \frac{\partial x_i}{\partial x_j}
$$

Evaluation order, with the values from above:

| step | node | adjoint | contributes to |
| --- | --- | --- | --- |
| 1 | 4 (`sum`) | 1 | node 3 += 1·1 = 1 |
| 2 | 3 (`powi`) | 1 | node 2 += 1·12 = 12 |
| 3 | 2 (`mul`) | 12 | node 0 += 12·3 = 36; node 1 += 12·2 = 24 |

### Why local gradients are an enum

Each edge stores its local partial derivative **computed during the forward
pass**:

```rust
pub enum LocalGrad {
    Elementwise(Tensor),   // dy/dx as a tensor of the same shape
    Permute(Vec<usize>),   // undo a transpose
    Reshape(Vec<usize>),   // undo a reshape
    MatmulLeft(Tensor),    // G·Bᵀ   (needs B, not a per-element partial)
    MatmulRight(Tensor),   // Aᵀ·G   (needs A)
}
```

`mul` is elementwise, but `matmul` is *not*: its adjoint with respect to `A` is
`G·Bᵀ`, a matrix product. Encoding that distinction in the enum makes the
mistake impossible rather than merely unlikely.

### Broadcasting

If `y = x + b` broadcast `b` from shape `(1, n)` to `(m, n)`, the adjoint of `b`
is the *sum* over the axis that was replicated:

$$
\frac{\partial L}{\partial b_j} = \sum_{i=0}^{m-1} \bar{y}_{ij}
$$

`sum_to_shape` performs exactly this reduction, using the target shape to decide
which axes to collapse. This is the transpose (in the linear-algebra sense) of
broadcasting, which is why broadcasting needs no extra machinery in the compile
step: it is a linear map with a known transpose.

### Semantics

* **Accumulation.** `backward()` adds to leaf adjoints, so calling it twice
  doubles them, exactly like PyTorch. Intermediates are reset each sweep so that
  a shared subexpression (`y = f(x) + f(x)`) is not double-counted.
* **Zero-skipping.** A node with an all-zero adjoint contributes nothing, so the
  sweep skips its subtree. For a loss that depends on a small part of a large
  tape this is what keeps the backward pass cheap.
* **Shared tapes.** `Variable::with_tape` installs a thread-local tape so that
  several leaves can take part in one gradient. Without it, two independently
  created leaves would live on separate tapes and their gradients would not meet.

## Validation

Every rule is checked three ways:

1. **Finite differences** — central differences with
   `h = tol^{1/3} · max(1, |x|)`, which balances truncation (`O(h²)`) against
   rounding (`O(ε/h)`).
2. **Forward vs reverse** — the two engines share no code, so agreement is strong
   evidence the rule is right.
3. **Structural checks** — e.g. `sum` of a gradient equals `n·∂L/∂mean`, and
   `reshape`/`transpose` preserve gradient mass.

The suite lives in `tests/integration/tests/gradient_check.rs` and the unit tests
in `crates/ncly-autodiff/src/*.rs`.

## Known limitations

* The tape is `Vec`-backed and append-only; there is no checkpointing, so memory
  grows with the number of operations.
* Complex numbers, higher-order derivatives and forward-over-reverse (Hessian
  vectors) are not implemented.
* `logsumexp` is stable, but the general `exp`/`ln` primitive pair will overflow
  or underflow where the mathematical result would not.
* `abs` uses the subgradient `sign(0) = 0`; other libraries pick `+1` or `−1`.
