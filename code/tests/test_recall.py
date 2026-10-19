import numpy as np
import pytest

from recall import recall_at_k

# 1-D points: id -> position. Query sits at 0.
# Distances from the query: id0=0, id1=1, id2=1 (tie with id1), id3=2, id4=3, id5=4
BASE = np.array([[0.0], [1.0], [-1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
QUERY = np.array([[0.0]], dtype=np.float32)


def kth(k):
    """True squared distance of the k-th neighbour of QUERY."""
    d = np.sort(((BASE - QUERY) ** 2).ravel())
    return np.array([d[k - 1]])


def test_perfect():
    assert recall_at_k([[0, 1, 2]], kth(3), BASE, QUERY, k=3) == 1.0


def test_all_wrong():
    assert recall_at_k([[5, 4, 3]], kth(3), BASE, QUERY, k=3) == 0.0


def test_partial():
    # k=4: true set distances {0,1,1,4}; ids 0 and 1 correct, 4 and 5 too far
    assert recall_at_k([[0, 1, 4, 5]], kth(4), BASE, QUERY, k=4) == 0.5


def test_tie_at_kth_neighbour_not_a_mistake():
    # k=2: exact answer is id0 plus either id1 or id2 (equidistant)
    assert recall_at_k([[0, 1]], kth(2), BASE, QUERY, k=2) == 1.0
    assert recall_at_k([[0, 2]], kth(2), BASE, QUERY, k=2) == 1.0


def test_beyond_tie_is_a_mistake():
    assert recall_at_k([[0, 3]], kth(2), BASE, QUERY, k=2) == 0.5


def test_missing_results_count_as_misses():
    assert recall_at_k([[0, -1, -1]], kth(3), BASE, QUERY, k=3) == pytest.approx(1 / 3)


def test_duplicate_ids_not_double_counted():
    assert recall_at_k([[0, 0, 0]], kth(3), BASE, QUERY, k=3) == pytest.approx(1 / 3)


def test_uses_only_first_k_columns():
    assert recall_at_k([[0, 1, 5, 5]], kth(2), BASE, QUERY, k=2) == 1.0


def test_averages_over_queries():
    queries = np.array([[0.0], [0.0]], dtype=np.float32)
    found = [[0, 1], [5, 4]]
    kk = np.array([kth(2)[0]] * 2)
    assert recall_at_k(found, kk, BASE, queries, k=2) == pytest.approx(0.5)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        recall_at_k([[0, 1]], kth(3), BASE, QUERY, k=3)
    with pytest.raises(ValueError):
        recall_at_k([[0, 1, 3]], np.array([0.0, 0.0]), BASE, QUERY, k=3)
