import numpy as np
import pytest

from src.event_delta.flow_warp import (
    SEVD_BACKGROUND,
    forward_splat_numpy,
    forward_splat_torch,
    pixel_to_feature_displacement,
    recurrent_forward_splat,
    sevd_flow_to_feature_displacement,
    sevd_flow_to_pixel_displacement,
)


def coordinate_features(height=4, width=6):
    yy, xx = np.indices((height, width), dtype=np.float32)
    return np.stack((xx, yy, xx + yy), axis=-1)


def test_forward_splat_uses_source_to_target_direction():
    features = coordinate_features()
    displacement = np.zeros((*features.shape[:2], 2), dtype=np.float32)
    displacement[..., 0] = 1
    propagated, valid, _confidence = forward_splat_numpy(features, displacement)
    np.testing.assert_allclose(propagated[:, 1:], features[:, :-1])
    assert not valid[:, 0].any()
    assert valid[:, 1:].all()


def test_pixel_to_feature_displacement_scaling():
    pixel = np.zeros((3, 4, 2), dtype=np.float32)
    pixel[..., 0], pixel[..., 1] = 20, 10
    grid = pixel_to_feature_displacement(pixel, (100, 200), (10, 20))
    np.testing.assert_allclose(grid[..., 0], 2)
    np.testing.assert_allclose(grid[..., 1], 1)


def test_sevd_background_subtraction_and_half_image_scale():
    height, width = 6, 8
    flow = np.full((height, width, 2), SEVD_BACKGROUND, dtype=np.float32)
    flow[..., 0] += 4 / width  # half-image convention -> two image pixels.
    pixel = sevd_flow_to_pixel_displacement(flow, (height, width))
    np.testing.assert_allclose(pixel[..., 0], 2)
    np.testing.assert_allclose(pixel[..., 1], 0)
    feature = sevd_flow_to_feature_displacement(flow, (3, 4), (height, width))
    np.testing.assert_allclose(feature[..., 0], 1)


def test_recurrent_two_step_translation_and_validity():
    features = coordinate_features(4, 7)
    displacement = np.zeros((4, 7, 2), dtype=np.float32)
    displacement[..., 0] = 1
    propagated, valid, coverage, confidence = recurrent_forward_splat(
        features, [displacement, displacement])
    np.testing.assert_allclose(propagated[:, 2:], features[:, :-2])
    assert not valid[:, :2].any() and valid[:, 2:].all()
    assert coverage == [6 / 7, 5 / 7]
    np.testing.assert_array_equal(confidence > 0, valid)


def test_confidence_masks_invalid_sources():
    features = coordinate_features(3, 4)
    confidence = np.ones((3, 4), dtype=np.float32)
    confidence[1, 1] = 0
    displacement = np.zeros((3, 4, 2), dtype=np.float32)
    propagated, valid, output_confidence = forward_splat_numpy(features, displacement, confidence)
    assert not valid[1, 1]
    assert output_confidence[1, 1] == 0
    np.testing.assert_allclose(propagated[valid], features[valid])


def test_shape_contract_rejects_old_30x1x40_broadcast_shape():
    features = np.zeros((30, 40, 3), dtype=np.float32)
    displacement = np.zeros((30, 40, 2), dtype=np.float32)
    bad_confidence = np.zeros((30, 1, 40), dtype=np.float32)
    with pytest.raises(ValueError, match=r"requires \(30, 40\), got \(30, 1, 40\)"):
        forward_splat_numpy(features, displacement, bad_confidence)


def test_torch_non_square_30x40_shape_contract_and_nine_recurrent_steps():
    """Regression for contribution[valid, None] producing [30,1,40]."""
    torch = pytest.importorskip("torch")
    height, width, channels = 30, 40, 3
    features = torch.arange(height * width * channels, dtype=torch.float32).reshape(height, width, channels)
    confidence = torch.ones((height, width), dtype=torch.float32)
    zero = torch.zeros((height, width, 2), dtype=torch.float32)

    identity, valid, output_confidence = forward_splat_torch(features, zero, confidence)
    assert identity.shape == (height, width, channels)
    assert valid.shape == output_confidence.shape == (height, width)
    assert torch.equal(identity, features)
    assert valid.all() and torch.isfinite(identity).all() and torch.isfinite(output_confidence).all()

    horizontal = zero.clone()
    horizontal[..., 0] = 1
    shifted_x, valid_x, confidence_x = forward_splat_torch(features, horizontal, confidence)
    assert shifted_x.shape == (height, width, channels)
    assert valid_x.shape == confidence_x.shape == (height, width)
    assert torch.equal(shifted_x[:, 1:], features[:, :-1])
    assert not valid_x[:, 0].any() and valid_x[:, 1:].all()

    vertical = zero.clone()
    vertical[..., 1] = 1
    shifted_y, valid_y, confidence_y = forward_splat_torch(features, vertical, confidence)
    assert shifted_y.shape == (height, width, channels)
    assert valid_y.shape == confidence_y.shape == (height, width)
    assert torch.equal(shifted_y[1:], features[:-1])
    assert not valid_y[0].any() and valid_y[1:].all()

    recurrent, recurrent_valid, coverage, recurrent_confidence = recurrent_forward_splat(
        features, [horizontal] * 9, confidence)
    assert recurrent.shape == (height, width, channels)
    assert recurrent_valid.shape == recurrent_confidence.shape == (height, width)
    assert len(coverage) == 9
    assert torch.isfinite(recurrent).all() and torch.isfinite(recurrent_confidence).all()
    assert torch.equal(recurrent[:, 9:], features[:, :-9])
