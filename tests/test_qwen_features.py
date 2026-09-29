import numpy as np
import pytest

from src.event_delta.qwen_features import ProbeError, TokenOrderError, tokens_to_grid


CONFIRMED_ORDER = {"status": "confirmed_from_installed_source"}


def test_merge_blocked_pre_merger_order_is_inverted_to_spatial_grid():
    spatial = np.arange(4 * 6).reshape(4, 6, 1)
    blocked = spatial.reshape(2, 2, 3, 2, 1).transpose(0, 2, 1, 3, 4).reshape(-1, 1)
    recovered = tokens_to_grid(blocked, np.array([[1, 4, 6]]), 2, "pre_merger", CONFIRMED_ORDER)
    np.testing.assert_array_equal(recovered, spatial)


def test_post_merger_grid_shape_and_count_are_checked():
    tokens = np.arange(6 * 3).reshape(6, 3)
    grid = tokens_to_grid(tokens, np.array([[1, 4, 6]]), 2, "post_merger", CONFIRMED_ORDER)
    assert grid.shape == (2, 3, 3)
    with pytest.raises(ProbeError, match="token/grid mismatch"):
        tokens_to_grid(tokens[:-1], np.array([[1, 4, 6]]), 2, "post_merger", CONFIRMED_ORDER)


def test_unresolved_token_order_stops_before_reshape():
    with pytest.raises(TokenOrderError, match="unresolved"):
        tokens_to_grid(np.zeros((6, 3)), np.array([[1, 4, 6]]), 2, "post_merger",
                       {"status": "unresolved", "warning": "unresolved order"})
