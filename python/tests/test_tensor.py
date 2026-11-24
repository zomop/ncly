"""Tests for the Python tensor API.

Run with ``pytest`` after building the extension:

    cd python && maturin develop --release && pytest -q
"""

import math

import pytest

import ncly


def test_construction_from_nested_lists_infers_shape():
    t = ncly.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert t.shape == [2, 3]
    assert t.ndim == 2
    assert t.size == 6
    assert t.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert t.to_nested() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_construction_with_explicit_shape():
    t = ncly.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    assert t.shape == [2, 2]


def test_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        ncly.Tensor([1.0, 2.0, 3.0], shape=[2, 2])


def test_ragged_sequences_are_rejected():
    with pytest.raises(ValueError):
        ncly.Tensor([[1.0, 2.0], [3.0]])


def test_elementwise_arithmetic_and_broadcasting():
    a = ncly.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = ncly.Tensor([10.0, 20.0, 30.0])
    assert (a + b).tolist() == [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    assert (a * a).tolist() == [1.0, 4.0, 9.0, 16.0, 25.0, 36.0]
    assert (-a).tolist() == [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]


def test_unary_functions_match_math_module():
    values = [-1.0, 0.0, 0.5, 2.0]
    t = ncly.Tensor(values)
    for got, expected in zip(t.exp().tolist(), [math.exp(v) for v in values]):
        assert math.isclose(got, expected, rel_tol=1e-12)
    for got, expected in zip(t.sin().tolist(), [math.sin(v) for v in values]):
        assert math.isclose(got, expected, abs_tol=1e-12)


def test_reductions():
    t = ncly.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert t.sum() == 21.0
    assert t.mean() == 3.5
    assert t.max() == 6.0
    assert t.min() == 1.0
    assert t.sum_axis(0).tolist() == [5.0, 7.0, 9.0]
    assert t.mean_axis(1).tolist() == [2.0, 5.0]


def test_matmul_and_transpose():
    a = ncly.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = ncly.Tensor([[7.0, 8.0], [9.0, 10.0]])
    product = a.matmul(b)
    assert product.shape == [3, 2]
    assert product.tolist() == [25.0, 28.0, 57.0, 64.0, 89.0, 100.0]
    assert a.transpose().shape == [2, 3]
    assert a.transpose().tolist() == [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]


def test_reshape_and_indexing():
    t = ncly.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    matrix = t.reshape([2, 3])
    assert matrix.shape == [2, 3]
    assert matrix[0, 1] == 2.0
    assert matrix[1].tolist() == [4.0, 5.0, 6.0]
    with pytest.raises(IndexError):
        _ = matrix[5, 0]


def test_norm_and_nan_detection():
    assert math.isclose(ncly.Tensor([3.0, 4.0]).norm(), 5.0)
    assert not ncly.Tensor([1.0, 2.0]).has_nan()


def test_linear_algebra_helpers():
    a = [[4.0, 3.0], [6.0, 3.0]]
    solution = ncly.solve(a, [10.0, 12.0])
    assert math.isclose(solution[0], 1.0, abs_tol=1e-10)
    assert math.isclose(solution[1], 2.0, abs_tol=1e-10)

    inverted = ncly.inverse(a)
    assert math.isclose(inverted[0][0], -0.5, abs_tol=1e-12)

    # Least squares fit of y = 2x + 1.
    design = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    coefficients = ncly.lstsq(design, [1.0, 3.0, 5.0])
    assert math.isclose(coefficients[0], 1.0, abs_tol=1e-10)
    assert math.isclose(coefficients[1], 2.0, abs_tol=1e-10)

    eigenvalues = ncly.eigvalsh([[2.0, 1.0], [1.0, 2.0]])
    assert math.isclose(eigenvalues[0], 1.0, abs_tol=1e-10)
    assert math.isclose(eigenvalues[1], 3.0, abs_tol=1e-10)


def test_singular_matrix_raises():
    # Singular systems map to ZeroDivisionError, matching the Rust
    # `NclyError::Singular` variant.
    with pytest.raises(ZeroDivisionError):
        ncly.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_sparse_matrix_roundtrip():
    matrix = ncly.CsrMatrix(2, 3, [0, 0, 1], [0, 2, 1], [2.0, 1.0, 3.0])
    assert matrix.nnz() == 3
    assert matrix.shape() == (2, 3)
    assert matrix.spmv([1.0, 1.0, 1.0]) == [3.0, 3.0]
    assert matrix.to_dense() == [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]]
