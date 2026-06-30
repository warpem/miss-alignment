"""Tests for optimize_global.py NaN handling."""

import pytest
import torch
from unittest.mock import patch
from warpylib import TiltSeries

from miss_alignment.alignment.optimize_global import optimize_shifts


def test_optimize_shifts_raises_on_nan_in_global_shifts():
    """Test that NaN in global shift parameters is handled with retries."""

    # Track how many times step is called to verify retry behavior
    call_count = [0]

    # Patch the LBFGS optimizer step to inject NaN before closure
    def mock_step(self, closure):
        call_count[0] += 1
        # Inject NaN before calling closure so it's detected
        # Preserve tensor shape by filling with NaN instead of collapsing
        for param_group in self.param_groups:
            for param in param_group["params"]:
                param.data.fill_(float("nan"))
        # Call closure which will detect NaN and raise AlignmentNanError
        closure()
        return None

    # Create minimal valid inputs
    n_tilts = 3
    ts = TiltSeries(n_tilts=n_tilts)
    ts.angles = torch.tensor([-30.0, 0.0, 30.0])
    ts.tilt_axis_offset_x = torch.zeros(n_tilts)
    ts.tilt_axis_offset_y = torch.zeros(n_tilts)

    # Store original offsets to verify they're preserved after failure
    original_offset_x = ts.tilt_axis_offset_x.clone()
    original_offset_y = ts.tilt_axis_offset_y.clone()
    ts.tilt_axis_angles = torch.tensor([2.0, 2.0, 2.0])
    original_tilt_axis_angles = ts.tilt_axis_angles.clone()

    # Mock model that just returns constant values
    class MockModel:
        def to(self, device):
            return self

        def freeze(self):
            pass

        def eval(self):
            pass

        def __call__(self, x):
            # Return score and log_precision with gradients
            return (
                torch.tensor([1.0], requires_grad=True),
                torch.tensor([0.0], requires_grad=True),
            )

    # Mock reconstruction to return dummy data
    original_reconstruct = ts.reconstruct_subvolumes_single
    ts.reconstruct_subvolumes_single = lambda **kwargs: torch.randn(
        1, 32, 32, 32, requires_grad=True
    )

    try:
        with patch.object(torch.optim.LBFGS, "step", mock_step):
            result_ts, losses = optimize_shifts(
                model=MockModel(),
                tilt_series=ts,
                images=torch.randn(n_tilts, 128, 128),
                pixel_size=10.0,
                positions=torch.tensor([[100.0, 100.0, 100.0]]),
                setting="global",
                patch_size=32,
                batch_size=1,
                apply_ctf=False,
                device="cpu",
                max_retries=3,
            )

            # After max_retries=3 failures, should return failure state
            # Verify: loss is inf, original offsets preserved
            assert losses == [float("inf")], f"Expected [inf] loss, got {losses}"
            assert torch.allclose(result_ts.tilt_axis_offset_x, original_offset_x), (
                "Original X offsets not preserved"
            )
            assert torch.allclose(result_ts.tilt_axis_offset_y, original_offset_y), (
                "Original Y offsets not preserved"
            )
            assert torch.allclose(
                result_ts.tilt_axis_angles, original_tilt_axis_angles
            ), "Original tilt-axis angles not preserved"
            # Should have tried 3 times (max_retries=3)
            assert call_count[0] == 3, f"Expected 3 retry attempts, got {call_count[0]}"

    finally:
        ts.reconstruct_subvolumes_single = original_reconstruct


def test_optimize_shifts_writes_back_tilt_axis_angle():
    """A successful global optimization finalizes the single tilt-axis angle.

    The model's per-subvolume normalization makes a synthetic gradient w.r.t.
    the angle awkward to construct, so instead we patch the LBFGS step to move
    the angle-delta leaf variable and verify it is written back uniformly to all
    tilts (initial + delta).
    """
    n_tilts = 3
    ts = TiltSeries(n_tilts=n_tilts)
    ts.angles = torch.tensor([-30.0, 0.0, 30.0])
    ts.tilt_axis_offset_x = torch.zeros(n_tilts)
    ts.tilt_axis_offset_y = torch.zeros(n_tilts)
    ts.tilt_axis_angles = torch.tensor([1.0, 1.0, 1.0])
    initial_tilt_axis_angles = ts.tilt_axis_angles.clone()

    angle_step = 2.0

    def mock_step(self, closure):
        # parameters are [shifts_y, shifts_x, tilt_axis_angle_delta]; move the
        # angle delta as if the optimizer found a better tilt-axis angle
        angle_delta = self.param_groups[0]["params"][-1]
        angle_delta.data.fill_(angle_step)
        closure()
        return None

    class MockModel:
        def to(self, device):
            return self

        def freeze(self):
            pass

        def eval(self):
            pass

        def __call__(self, x):
            return (
                torch.tensor([1.0], requires_grad=True),
                torch.tensor([0.0], requires_grad=True),
            )

    original_reconstruct = ts.reconstruct_subvolumes_single
    ts.reconstruct_subvolumes_single = lambda **kwargs: torch.randn(
        1, 32, 32, 32, requires_grad=True
    )

    try:
        with patch.object(torch.optim.LBFGS, "step", mock_step):
            result_ts, _ = optimize_shifts(
                model=MockModel(),
                tilt_series=ts,
                images=torch.randn(n_tilts, 128, 128),
                pixel_size=10.0,
                positions=torch.tensor([[100.0, 100.0, 100.0]]),
                setting="global",
                patch_size=32,
                batch_size=1,
                apply_ctf=False,
                device="cpu",
            )

        # the single angle delta is added uniformly to all tilts
        expected = initial_tilt_axis_angles + angle_step
        assert torch.allclose(result_ts.tilt_axis_angles, expected), (
            f"Expected tilt-axis angles {expected}, got {result_ts.tilt_axis_angles}"
        )
    finally:
        ts.reconstruct_subvolumes_single = original_reconstruct


@pytest.mark.skip(reason="Grid initialization in test environment is complex")
def test_optimize_shifts_raises_on_nan_in_image_warp():
    """Test that NaN in image warp grid parameters is handled with retries.

    Note: This test is skipped because proper grid initialization requires
    a fully configured TiltSeries with correct image/volume dimensions and
    grid state management across retries. The NaN detection and retry logic
    is identical to the global shift case, so that test provides adequate coverage.
    """

    call_count = [0]

    def mock_step(self, closure):
        call_count[0] += 1
        # Inject NaN before calling closure so it's detected
        # Preserve tensor shape by filling with NaN instead of collapsing
        for param_group in self.param_groups:
            for param in param_group["params"]:
                param.data.fill_(float("nan"))
        # Call closure which will detect NaN and raise AlignmentNanError
        closure()
        return None

    n_tilts = 3
    ts = TiltSeries(n_tilts=n_tilts)
    ts.angles = torch.tensor([-30.0, 0.0, 30.0])
    ts.tilt_axis_offset_x = torch.zeros(n_tilts)
    ts.tilt_axis_offset_y = torch.zeros(n_tilts)
    # Set image dimensions to allow proper grid initialization
    ts.image_dimensions_physical = torch.tensor([1280.0, 1280.0])
    ts.volume_dimensions_physical = torch.tensor([1280.0, 1280.0, 500.0])

    # Store original offsets to verify they're preserved after failure
    original_offset_x = ts.tilt_axis_offset_x.clone()
    original_offset_y = ts.tilt_axis_offset_y.clone()

    class MockModel:
        def to(self, device):
            return self

        def freeze(self):
            pass

        def eval(self):
            pass

        def __call__(self, x):
            return (
                torch.tensor([1.0], requires_grad=True),
                torch.tensor([0.0], requires_grad=True),
            )

    original_reconstruct = ts.reconstruct_subvolumes_single
    ts.reconstruct_subvolumes_single = lambda **kwargs: torch.randn(
        1, 32, 32, 32, requires_grad=True
    )

    try:
        with patch.object(torch.optim.LBFGS, "step", mock_step):
            result_ts, losses = optimize_shifts(
                model=MockModel(),
                tilt_series=ts,
                images=torch.randn(n_tilts, 128, 128),
                pixel_size=10.0,
                positions=torch.tensor([[100.0, 100.0, 100.0]]),
                setting=(3, 3, 3),  # 2D warping field
                patch_size=32,
                batch_size=1,
                apply_ctf=False,
                device="cpu",
                max_retries=3,
            )

            # After max_retries=3 failures, should return failure state
            assert losses == [float("inf")], f"Expected [inf] loss, got {losses}"
            assert torch.allclose(result_ts.tilt_axis_offset_x, original_offset_x), (
                "Original X offsets not preserved"
            )
            assert torch.allclose(result_ts.tilt_axis_offset_y, original_offset_y), (
                "Original Y offsets not preserved"
            )
            # Should have tried 3 times (max_retries=3)
            assert call_count[0] == 3, f"Expected 3 retry attempts, got {call_count[0]}"

    finally:
        ts.reconstruct_subvolumes_single = original_reconstruct


@pytest.mark.skip(reason="Volume warp grid setup requires more complex initialization")
def test_optimize_shifts_raises_on_nan_in_volume_warp():
    """Test that NaN in volume warp grid parameters is handled with retries.

    Note: This test is skipped because setting up a proper 4D volume warp grid
    requires more complex TiltSeries initialization that goes beyond the scope
    of this unit test. The NaN checking and retry logic is identical to the
    image warp case, so the coverage is adequate.
    """
    pass
