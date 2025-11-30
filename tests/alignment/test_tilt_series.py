import torch
from warpylib import TiltSeries

from miss_alignment.alignment.tilt_series import (
    generate_position_grid,
    run_iterative_anchoring,
)


def test_generate_position_grid():
    grid = generate_position_grid(
        volume_dimensions_physical=(5120, 5120, 1800),
        pixel_size=10,
        patch_size=96,
        patch_overlap=0.1,
    )
    assert grid.shape == (50, 3)

    grid = generate_position_grid(
        volume_dimensions_physical=(5120, 5120, 1800),
        pixel_size=20,
        patch_size=512,
        patch_overlap=0.1,
    )
    assert grid.shape == (1, 3), "grid should always give at least 1 position"


def test_run_iterative_anchoring_preserves_relative_offsets():
    """Test that iterative anchoring preserves relative offsets of unreliable tilts."""
    # Create a tilt series with 11 tilts: angles from -50 to +50
    n_tilts = 11
    ts = TiltSeries(n_tilts=n_tilts)
    ts.angles = torch.linspace(-50, 50, n_tilts)

    # Set up offsets with a known pattern (easy to verify)
    # Offsets increase linearly with tilt index when sorted by angle
    ts.tilt_axis_offset_x = torch.arange(n_tilts, dtype=torch.float32)
    ts.tilt_axis_offset_y = torch.arange(n_tilts, dtype=torch.float32) * 2

    # Store original relative offsets for verification
    sorted_indices = ts.indices_sorted_angle()
    original_rel_x = []
    original_rel_y = []
    for i in range(n_tilts - 1):
        curr_idx = sorted_indices[i]
        next_idx = sorted_indices[i + 1]
        original_rel_x.append(
            (ts.tilt_axis_offset_x[curr_idx] - ts.tilt_axis_offset_x[next_idx]).item()
        )
        original_rel_y.append(
            (ts.tilt_axis_offset_y[curr_idx] - ts.tilt_axis_offset_y[next_idx]).item()
        )

    # Track how many times the optimizer is called
    call_count = [0]

    def mock_optimizer(tilt_series: TiltSeries):
        """Mock optimizer that shifts all tilts by a fixed amount."""
        call_count[0] += 1
        # Shift all tilts by 10.0 in x and 5.0 in y
        tilt_series.tilt_axis_offset_x = tilt_series.tilt_axis_offset_x + 10.0
        tilt_series.tilt_axis_offset_y = tilt_series.tilt_axis_offset_y + 5.0
        # Use decreasing loss so each iteration improves (required for new behavior)
        return tilt_series, [1.0 / call_count[0]]

    # Run with initial_reliable_fraction=0.5 (roughly half reliable)
    # With 11 tilts and 0.5 fraction: n_reliable=5, n_unreliable_total=6, n_unreliable_per_side=3
    result_ts, losses = run_iterative_anchoring(
        tilt_series=ts,
        optimize_fn=mock_optimizer,
        initial_reliable_fraction=0.5,
    )

    # Verify optimizer was called multiple times:
    # 3 iterations (for n_unreliable_per_side = 3, 2, 1) + 1 final = 4 calls
    assert call_count[0] == 4, f"Expected 4 optimizer calls, got {call_count[0]}"

    # Verify we got loss values from each call
    assert len(losses) == 4

    # After all iterations, all tilts should have shifted by the cumulative optimizer shifts
    # But relative offsets between consecutive tilts should be preserved
    final_rel_x = []
    final_rel_y = []
    for i in range(n_tilts - 1):
        curr_idx = sorted_indices[i]
        next_idx = sorted_indices[i + 1]
        final_rel_x.append(
            (
                result_ts.tilt_axis_offset_x[curr_idx]
                - result_ts.tilt_axis_offset_x[next_idx]
            ).item()
        )
        final_rel_y.append(
            (
                result_ts.tilt_axis_offset_y[curr_idx]
                - result_ts.tilt_axis_offset_y[next_idx]
            ).item()
        )

    # The relative offsets should be preserved (within floating point tolerance)
    for i in range(n_tilts - 1):
        assert abs(final_rel_x[i] - original_rel_x[i]) < 1e-5, (
            f"Relative X offset at position {i} not preserved: "
            f"original={original_rel_x[i]}, final={final_rel_x[i]}"
        )
        assert abs(final_rel_y[i] - original_rel_y[i]) < 1e-5, (
            f"Relative Y offset at position {i} not preserved: "
            f"original={original_rel_y[i]}, final={final_rel_y[i]}"
        )


def test_run_iterative_anchoring_all_reliable():
    """Test behavior when all tilts are initially reliable (no iterations needed)."""
    n_tilts = 5
    ts = TiltSeries(n_tilts=n_tilts)
    ts.angles = torch.linspace(-20, 20, n_tilts)
    ts.tilt_axis_offset_x = torch.zeros(n_tilts)
    ts.tilt_axis_offset_y = torch.zeros(n_tilts)

    call_count = [0]

    def mock_optimizer(tilt_series: TiltSeries):
        call_count[0] += 1
        tilt_series.tilt_axis_offset_x = tilt_series.tilt_axis_offset_x + 1.0
        return tilt_series, [0.5]

    # With fraction=1.0, all tilts are reliable from the start
    # n_unreliable_per_side = 0, so we skip the while loop and just do final optimization
    result_ts, losses = run_iterative_anchoring(
        tilt_series=ts,
        optimize_fn=mock_optimizer,
        initial_reliable_fraction=1.0,
    )

    # Only 1 call (the final optimization)
    assert call_count[0] == 1
    assert len(losses) == 1


def test_run_iterative_anchoring_boundary_movement_propagates():
    """Test that boundary tilt movement propagates to unreliable tilts."""
    n_tilts = 7
    ts = TiltSeries(n_tilts=n_tilts)
    ts.angles = torch.linspace(-30, 30, n_tilts)

    # Start with all offsets at 0
    ts.tilt_axis_offset_x = torch.zeros(n_tilts)
    ts.tilt_axis_offset_y = torch.zeros(n_tilts)

    sorted_indices = ts.indices_sorted_angle()
    call_count = [0]

    def mock_optimizer(tilt_series: TiltSeries):
        """Only shift the reliable (central) tilts."""
        call_count[0] += 1
        # This simulates what happens when only central tilts can optimize well
        # For simplicity, shift all tilts - the restoration will fix unreliable ones
        tilt_series.tilt_axis_offset_x = tilt_series.tilt_axis_offset_x + 5.0
        # Use decreasing loss so iterations proceed
        return tilt_series, [1.0 / call_count[0]]

    # With 7 tilts and 0.5 fraction: n_reliable=3, n_unreliable_per_side=2
    result_ts, losses = run_iterative_anchoring(
        tilt_series=ts,
        optimize_fn=mock_optimizer,
        initial_reliable_fraction=0.5,
    )

    # After iterative anchoring, all tilts should have moved
    # (boundary movements propagate to unreliable tilts)
    for i in range(n_tilts):
        idx = sorted_indices[i]
        # All tilts should have non-zero offset after optimization
        assert result_ts.tilt_axis_offset_x[idx].item() != 0.0


def test_run_iterative_anchoring_reverts_on_worse_loss():
    """Test that worse iterations are immediately reverted to best."""
    n_tilts = 7
    ts = TiltSeries(n_tilts=n_tilts)
    ts.angles = torch.linspace(-30, 30, n_tilts)
    ts.tilt_axis_offset_x = torch.zeros(n_tilts)
    ts.tilt_axis_offset_y = torch.zeros(n_tilts)

    call_count = [0]
    # With 7 tilts and 0.5 fraction: n_reliable=3, n_unreliable_per_side=2
    # So we get 2 iterations in while loop + 1 final = 3 optimizer calls
    # First iteration improves, second gets worse (reverts), third improves again
    losses_sequence = [0.5, 0.8, 0.4]

    def mock_optimizer(tilt_series: TiltSeries):
        idx = call_count[0]
        loss = losses_sequence[idx]
        call_count[0] += 1

        # Each call shifts by different amount
        tilt_series.tilt_axis_offset_x = tilt_series.tilt_axis_offset_x + (idx + 1) * 10.0
        return tilt_series, [loss]

    result_ts, losses = run_iterative_anchoring(
        tilt_series=ts,
        optimize_fn=mock_optimizer,
        initial_reliable_fraction=0.5,
    )

    # Check that we got the expected loss values
    assert losses == [0.5, 0.8, 0.4]
    assert call_count[0] == 3

    # Final loss 0.4 is the best, so that solution should be kept
    # The final call added +30 to whatever state it started from
    # After call 1: best_offsets = 10.0 (loss=0.5)
    # After call 2: loss=0.8 >= 0.5, so reverts to 10.0
    # After call 3: starts from 10.0, adds +30 = 40.0 (loss=0.4 is new best)
    sorted_indices = ts.indices_sorted_angle()
    central_idx = sorted_indices[3]
    assert result_ts.tilt_axis_offset_x[central_idx].item() == 40.0


def test_run_iterative_anchoring_handles_nan():
    """Test that NaN values cause immediate revert."""
    n_tilts = 7
    ts = TiltSeries(n_tilts=n_tilts)
    ts.angles = torch.linspace(-30, 30, n_tilts)
    ts.tilt_axis_offset_x = torch.zeros(n_tilts)
    ts.tilt_axis_offset_y = torch.zeros(n_tilts)

    call_count = [0]
    # First iteration works, second produces NaN, third works
    losses_sequence = [0.5, float("nan"), 0.4]

    def mock_optimizer(tilt_series: TiltSeries):
        idx = call_count[0]
        loss = losses_sequence[idx]
        call_count[0] += 1

        tilt_series.tilt_axis_offset_x = tilt_series.tilt_axis_offset_x + (idx + 1) * 10.0
        return tilt_series, [loss]

    result_ts, losses = run_iterative_anchoring(
        tilt_series=ts,
        optimize_fn=mock_optimizer,
        initial_reliable_fraction=0.5,
    )

    assert call_count[0] == 3
    # After NaN, should have reverted to best (10.0), then final adds +30 = 40.0
    sorted_indices = ts.indices_sorted_angle()
    central_idx = sorted_indices[3]
    assert result_ts.tilt_axis_offset_x[central_idx].item() == 40.0
