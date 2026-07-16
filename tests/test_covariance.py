import numpy as np
import pytest

from ascov import sample_covariance


def test_matches_numpy_on_small_example() -> None:
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 3.0],
        ]
    )

    result = sample_covariance(X)
    expected = np.cov(X, rowvar=False, ddof=1)

    np.testing.assert_allclose(result, expected)


def test_returns_expected_shape() -> None:
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]
    )

    result = sample_covariance(X)

    assert result.shape == (3, 3)


def test_result_is_symmetric() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))

    result = sample_covariance(X)

    np.testing.assert_allclose(result, result.T, atol=1e-12)


def test_one_dimensional_variable() -> None:
    X = np.array(
        [
            [1.0],
            [2.0],
            [4.0],
        ]
    )

    result = sample_covariance(X)
    expected_variance = np.var(X[:, 0], ddof=1)

    assert result.shape == (1, 1)
    np.testing.assert_allclose(result[0, 0], expected_variance)


def test_constant_column_has_zero_variance() -> None:
    X = np.array(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
            [4.0, 5.0],
        ]
    )

    result = sample_covariance(X)

    np.testing.assert_allclose(result[1, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(result[1, 0], 0.0, atol=1e-12)


def test_perfectly_correlated_columns_produce_rank_one_covariance() -> None:
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ]
    )

    result = sample_covariance(X)
    eigenvalues = np.linalg.eigvalsh(result)

    assert eigenvalues[0] >= -1e-12
    np.testing.assert_allclose(eigenvalues[0], 0.0, atol=1e-12)


def test_rejects_one_dimensional_input() -> None:
    X = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="two-dimensional"):
        sample_covariance(X)


def test_rejects_fewer_than_two_observations() -> None:
    X = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="At least two observations"):
        sample_covariance(X)

def test_reject_zero_columns() -> None:
    X = np.empty((5, 0))

    with pytest.raises(ValueError, match="at least one variable"):
        sample_covariance(X)

def test_more_features_than_observations() -> None:
    X = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 1.0, 4.0, 3.0, 6.0],
            [3.0, 4.0, 2.0, 5.0, 1.0],
        ]
    )

    result = sample_covariance(X)
    expected = np.cov(X, rowvar=False, ddof=1)

    assert result.shape == (5, 5)

    np.testing.assert_allclose(result, expected)

    rank = np.linalg.matrix_rank(result)
    assert rank < 5

    eigenvalues = np.linalg.eigvalsh(result)
    assert np.min(eigenvalues) >= -1e-12