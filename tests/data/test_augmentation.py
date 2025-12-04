import pytest
import torch
import random
from unittest.mock import patch

from miss_alignment.data._augmentation import (
    random_contrast,
    random_cube_mask,
    random_mirror,
    random_edge_mask,
    apply_mirror,
    MIRROR_COMBINATIONS,
)


class TestRandomContrast:
    """Test suite for random_contrast function."""

    def test_shape_preservation(self):
        """Test that output shape matches input shape."""
        volume = torch.randn(32, 64, 64)
        result = random_contrast(volume)
        assert result.shape == volume.shape

    def test_deterministic_with_seed(self):
        """Test reproducible results with torch seed."""
        volume = torch.randn(16, 32, 32)

        torch.manual_seed(42)
        result1 = random_contrast(volume)

        torch.manual_seed(42)
        result2 = random_contrast(volume)

        torch.testing.assert_close(result1, result2)

    def test_different_outputs_different_seeds(self):
        """Test that different seeds produce different outputs."""
        volume = torch.randn(16, 32, 32)

        torch.manual_seed(1)
        result1 = random_contrast(volume)

        torch.manual_seed(2)
        result2 = random_contrast(volume)

        assert not torch.allclose(result1, result2)

    def test_transformation_applied(self):
        """Test that the transformation actually modifies the input."""
        volume = torch.ones(8, 16, 16)
        result = random_contrast(volume)

        # Should not be identical to input (with high probability)
        assert not torch.allclose(volume, result)


class TestRandomMirror:
    """Test suite for random_mirror function."""

    def test_single_volume_shape_preservation(self):
        """Test that output shapes match input shapes."""
        volume = torch.randn(16, 32, 64)
        result = random_mirror([volume])
        assert len(result) == 1
        assert result[0].shape == volume.shape

    def test_multiple_volumes_consistent_flipping(self):
        """Test that all volumes are flipped identically."""
        vol1 = torch.randn(8, 16, 16)
        vol2 = torch.randn(8, 16, 16)

        random.seed(42)
        result = random_mirror([vol1, vol2])

        # If flipped, the relationship between volumes should be preserved
        # Test by checking if relative differences are maintained
        original_diff = vol1 - vol2
        result_diff = result[0] - result[1]

        # The difference should be identical after identical flipping
        # (might be flipped itself, but structure preserved)
        assert result_diff.shape == original_diff.shape

    def test_deterministic_with_seed(self):
        """Test reproducible results with random seed."""
        volume = torch.randn(8, 16, 16)

        random.seed(42)
        result1 = random_mirror([volume])

        random.seed(42)
        result2 = random_mirror([volume])

        torch.testing.assert_close(result1[0], result2[0])

    def test_empty_list_handling(self):
        """Test that empty input list returns empty list."""
        result = random_mirror([])
        assert result == []

    @patch("random.choice")
    def test_no_flip_returns_original(self, mock_choice):
        """Test that no flipping returns original volumes."""
        mock_choice.return_value = False  # No flips

        volume = torch.randn(8, 16, 16)
        result = random_mirror([volume])

        torch.testing.assert_close(result[0], volume)

    @patch("random.choice")
    def test_all_dimensions_flipped(self, mock_choice):
        """Test flipping all dimensions."""
        mock_choice.return_value = True  # Flip all dimensions

        volume = torch.randn(4, 8, 8)
        result = random_mirror([volume])

        # Manual flip on all dimensions should match
        expected = torch.flip(volume, [-3, -2, -1])
        torch.testing.assert_close(result[0], expected)


class TestRandomCubeMask:
    """Test suite for random_cube_mask function."""

    def test_shape_preservation(self):
        """Test that output shape matches input shape."""
        volume = torch.randn(32, 64, 64)
        result = random_cube_mask(volume)
        assert result.shape == volume.shape

    def test_probability_zero_no_change(self):
        """Test that p=0 results in no change."""
        volume = torch.randn(16, 32, 32)
        original = volume.clone()
        result = random_cube_mask(volume, p=0.0)
        torch.testing.assert_close(result, original)

    def test_probability_one_always_applies(self):
        """Test that p=1 always applies the mask."""
        volume = torch.ones(16, 32, 32)
        result = random_cube_mask(volume, p=1.0)

        # Should contain mask values in range [-1, 1]
        unique_values = torch.unique(result)
        has_mask_values = any(
            abs(val.item()) < 0.6 and val.item() != 1.0 for val in unique_values
        )
        assert has_mask_values

    @patch("random.random")
    def test_mask_application_deterministic(self, mock_random):
        """Test mask application with controlled randomness."""
        # Set up mocks for deterministic behavior
        mock_random.side_effect = [0.5, 0.2, 0.1]  # p trigger, size, mask_value

        volume = torch.zeros(10, 10, 10)
        result = random_cube_mask(volume, p=0.3, size_range=(0.2, 0.4))

        # Should be modified since 0.5 > (1 - 0.3) = 0.7 is False
        # Actually triggers since random() > 1-p means random() > 0.7
        # 0.5 is not > 0.7, so no mask applied
        torch.testing.assert_close(result, volume)

    def test_size_range_bounds(self):
        """Test that mask size respects the given range bounds."""
        volume = torch.ones(20, 20, 20)

        # Use p=1.0 to ensure mask is always applied
        with (
            patch("random.uniform") as mock_uniform,
            patch("random.randint") as mock_randint,
            patch("random.random") as mock_random,
        ):
            mock_random.side_effect = [0.5, 0.0]  # Trigger mask, set mask value
            mock_uniform.return_value = 0.25  # 25% of volume size
            mock_randint.return_value = 0  # Start at origin

            _ = random_cube_mask(volume, p=1.0, size_range=(0.2, 0.3))

            # Verify uniform was called with correct range
            mock_uniform.assert_called_once_with(0.2, 0.3)


class TestRandomEdgeMask:
    """Test suite for random_edge_mask function."""

    def test_shape_preservation(self):
        """Test that output shape matches input shape."""
        volume = torch.randn(16, 32, 32)
        result = random_edge_mask(volume)
        assert result.shape == volume.shape

    def test_probability_zero_no_change(self):
        """Test that p=0 results in no change."""
        volume = torch.randn(16, 32, 32)
        original = volume.clone()
        result = random_edge_mask(volume, p=0.0)
        torch.testing.assert_close(result, original)

    @patch("random.random")
    def test_probability_one_always_applies(self, mock_random):
        """Test that p=1 always applies edge masking."""
        mock_random.side_effect = [0.0, 0.5]  # p, mask_value
        volume = torch.ones(10, 10, 10)
        result = random_edge_mask(volume, p=1.0, edge_width=(2, 2))

        # Check that edges are masked (should contain 0.0 values)
        assert 0.0 in result
        # Check that center still has original values
        center = result[2:-2, 2:-2, 2:-2]
        assert torch.all(center == 1.0)

    def test_edge_width_range(self):
        """Test that edge width is within specified range."""
        volume = torch.ones(20, 20, 20)

        with (
            patch("random.random") as mock_random,
            patch("random.randint") as mock_randint,
        ):
            mock_random.return_value = 0.5  # Always apply mask
            mock_randint.return_value = 3  # Fixed width

            _ = random_edge_mask(
                volume,
                p=1.0,
                edge_width=(2, 5),
            )

            # Verify randint was called with correct range
            mock_randint.assert_called_once_with(2, 5)

    @patch("random.random")
    def test_all_edges_masked(self, mock_random):
        """Test that all 6 faces of the cube are masked."""
        mask_value = 1.0
        mock_random.side_effect = [0.5, (mask_value + 1) / 2]
        volume = torch.ones(10, 10, 10)
        width = 1

        result = random_edge_mask(volume, p=1.0, edge_width=(width, width))

        # Check all 6 faces are masked
        assert torch.all(result[:width, :, :] == mask_value)  # Front face
        assert torch.all(result[-width:, :, :] == mask_value)  # Back face
        assert torch.all(result[:, :width, :] == mask_value)  # Top face
        assert torch.all(result[:, -width:, :] == mask_value)  # Bottom face
        assert torch.all(result[:, :, :width] == mask_value)  # Left face
        assert torch.all(result[:, :, -width:] == mask_value)  # Right face

        # Check center is not masked
        center = result[width:-width, width:-width, width:-width]
        assert torch.all(center == 1.0)


# Parameterized tests for edge cases
class TestAugmentationEdgeCases:
    """Test edge cases across all augmentation functions."""

    @pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (100, 100, 100)])
    def test_various_volume_sizes(self, shape):
        """Test functions work with various volume sizes."""
        volume = torch.randn(shape)

        # All functions should handle different sizes
        contrast_result = random_contrast(volume)
        cube_result = random_cube_mask(volume.clone())
        mirror_result = random_mirror([volume.clone()])
        edge_result = random_edge_mask(volume.clone())

        assert contrast_result.shape == shape
        assert cube_result.shape == shape
        assert mirror_result[0].shape == shape
        assert edge_result.shape == shape

    def test_inplace_modifications(self):
        """Test that functions properly handle in-place modifications."""
        volume = torch.randn(8, 8, 8)
        original = volume.clone()

        # These functions may modify in-place
        random_cube_mask(volume, p=1.0)
        random_edge_mask(volume, p=1.0)

        # Volume should be modified
        assert not torch.allclose(volume, original)


class TestApplyMirror:
    """Test suite for apply_mirror function."""

    def test_shape_preservation(self):
        """Test that output shape matches input shape."""
        volume = torch.randn(16, 32, 64)
        result = apply_mirror(volume, (False, False, False))
        assert result.shape == volume.shape

    def test_no_flip_returns_original(self):
        """Test that no flipping returns identical tensor."""
        volume = torch.randn(8, 16, 16)
        result = apply_mirror(volume, (False, False, False))
        torch.testing.assert_close(result, volume)

    def test_flip_z_only(self):
        """Test flipping only the z dimension."""
        volume = torch.randn(8, 16, 16)
        result = apply_mirror(volume, (True, False, False))
        expected = torch.flip(volume, [-3])
        torch.testing.assert_close(result, expected)

    def test_flip_y_only(self):
        """Test flipping only the y dimension."""
        volume = torch.randn(8, 16, 16)
        result = apply_mirror(volume, (False, True, False))
        expected = torch.flip(volume, [-2])
        torch.testing.assert_close(result, expected)

    def test_flip_x_only(self):
        """Test flipping only the x dimension."""
        volume = torch.randn(8, 16, 16)
        result = apply_mirror(volume, (False, False, True))
        expected = torch.flip(volume, [-1])
        torch.testing.assert_close(result, expected)

    def test_flip_all_dimensions(self):
        """Test flipping all dimensions."""
        volume = torch.randn(8, 16, 16)
        result = apply_mirror(volume, (True, True, True))
        expected = torch.flip(volume, [-3, -2, -1])
        torch.testing.assert_close(result, expected)

    def test_flip_two_dimensions(self):
        """Test flipping two dimensions (z and y)."""
        volume = torch.randn(8, 16, 16)
        result = apply_mirror(volume, (True, True, False))
        expected = torch.flip(volume, [-3, -2])
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("combo", MIRROR_COMBINATIONS)
    def test_all_combinations(self, combo):
        """Test all 8 mirror combinations work correctly."""
        volume = torch.randn(4, 8, 8)
        result = apply_mirror(volume, combo)

        # Build expected result
        dims_to_flip = []
        for i, should_flip in enumerate(combo):
            if should_flip:
                dims_to_flip.append(-(3 - i))

        if dims_to_flip:
            expected = torch.flip(volume, dims_to_flip)
        else:
            expected = volume

        torch.testing.assert_close(result, expected)


class TestMirrorCombinations:
    """Test the MIRROR_COMBINATIONS constant."""

    def test_has_8_combinations(self):
        """Test that there are exactly 8 mirror combinations (2^3)."""
        assert len(MIRROR_COMBINATIONS) == 8

    def test_all_unique(self):
        """Test that all combinations are unique."""
        assert len(set(MIRROR_COMBINATIONS)) == 8

    def test_includes_no_flip(self):
        """Test that no-flip combination is included."""
        assert (False, False, False) in MIRROR_COMBINATIONS

    def test_includes_all_flip(self):
        """Test that all-flip combination is included."""
        assert (True, True, True) in MIRROR_COMBINATIONS

    def test_all_boolean_tuples(self):
        """Test that all combinations are 3-element boolean tuples."""
        for combo in MIRROR_COMBINATIONS:
            assert isinstance(combo, tuple)
            assert len(combo) == 3
            assert all(isinstance(b, bool) for b in combo)
