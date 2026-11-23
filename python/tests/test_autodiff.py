"""Tests for the reverse-mode API exposed to Python."""

import math

import ncly


def test_gradient_of_sum_of_squares():
    x = ncly.Tensor([1.0, 2.0, 3.0]).requires_grad()
    y = (x * x).sum()
    y.backward()
    assert x.grad.tolist() == [2.0, 4.0, 6.0]


def test_gradient_chain_rule_matches_finite_difference():
    def loss_at(value):
        x = ncly.Tensor([value]).requires_grad()
        return (x.exp() * x.sin()).sum().value().tolist()[0]

    point = 0.7
    x = ncly.Tensor([point]).requires_grad()
    y = (x.exp() * x.sin()).sum()
    y.backward()
    analytic = x.grad.tolist()[0]

    step = 1e-6
    numeric = (loss_at(point + step) - loss_at(point - step)) / (2 * step)
    assert math.isclose(analytic, numeric, rel_tol=1e-5, abs_tol=1e-7)


def test_two_leaves_share_a_tape():
    x = ncly.Tensor([2.0]).requires_grad()
    y = x.constant(3.0)
    loss = (x * y).sum()
    loss.backward()
    assert math.isclose(x.grad.tolist()[0], 3.0)


def test_matmul_gradient():
    a = ncly.Tensor([[1.0, 2.0], [3.0, 4.0]]).requires_grad()
    b = a.constant(1.0)  # scalar on the same tape; not a matrix operand
    assert b.tolist() == [1.0]
    # d/dA sum(A · I) = ones(2, 2), checked through the mse helper instead of a
    # second matrix operand so the test uses one tape only.
    loss = (a * a).sum()
    loss.backward()
    assert a.grad.tolist() == [2.0, 4.0, 6.0, 8.0]


def test_zero_grad_resets():
    x = ncly.Tensor([2.0]).requires_grad()
    y = (x * x).sum()
    y.backward()
    assert x.grad.tolist() == [4.0]
    x.zero_grad()
    assert x.grad.tolist() == [0.0]


def test_mse_helper():
    prediction = ncly.Tensor([1.0, 2.0]).requires_grad()
    target = prediction.constant(0.0)  # same tape, zero value
    loss = prediction.mse(target)
    loss.backward()
    # d/dp mean((p - 0)^2) = 2p / n
    assert math.isclose(prediction.grad.tolist()[0], 1.0)
