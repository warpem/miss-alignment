import pytest
import torch
import random
from unittest.mock import patch, MagicMock

# Import your modules (adjust import path as needed)
from miss_alignment.data.shift_generation import (
    ShiftConfig, ShiftGenerator, select_random_indices,
    generate_smooth_trajectory, generate_shift_trajectory,
    TrajectoryGenerator, JitterGenerator,
    OutlierGenerator, create_default_generator,
    project_shifts_3d_to_2d
)


class TestShiftConfig:
    """Test the ShiftConfig dataclass."""

    def test_shift_config_creation(self):
        """Test basic ShiftConfig creation."""

        def dummy_generator(n):
            return torch.zeros((n, 3))

        config = ShiftConfig("test", 0.5, dummy_generator)
        assert config.name == "test"
        assert config.probability == 0.5
        assert config.generator == dummy_generator

    def test_should_apply_zero_probability(self):
        """Test that zero probability never applies."""

        def dummy_generator(n):
            return torch.zeros((n, 3))

        config = ShiftConfig("test", 0.0, dummy_generator)

        # Test multiple times to ensure it never applies
        for _ in range(100):
            assert not config.should_apply()

    def test_should_apply_full_probability(self):
        """Test that probability 1.0 always applies."""

        def dummy_generator(n):
            return torch.zeros((n, 3))

        config = ShiftConfig("test", 1.0, dummy_generator)

        # Test multiple times to ensure it always applies
        for _ in range(100):
            assert config.should_apply()

    @patch('random.random')
    def test_should_apply_probability_logic(self, mock_random):
        """Test probability logic with mocked random."""

        def dummy_generator(n):
            return torch.zeros((n, 3))

        config = ShiftConfig("test", 0.7, dummy_generator)

        # Should apply when random < probability
        mock_random.return_value = 0.5
        assert config.should_apply()

        # Should not apply when random >= probability
        mock_random.return_value = 0.8
        assert not config.should_apply()


class TestShiftGenerator:
    """Test the ShiftGenerator class."""

    def test_init(self):
        """Test ShiftGenerator initialization."""
        configs = []
        generator = ShiftGenerator(configs, ensure_at_least_one=False)
        assert generator.shift_configs == configs
        assert generator.ensure_at_least_one == False

    def test_call_method(self):
        """Test that __call__ method works."""

        def dummy_generator(n):
            return torch.ones((n, 3))

        config = ShiftConfig("test", 1.0, dummy_generator)
        generator = ShiftGenerator([config])

        result = generator(10)
        assert result.shape == (10, 3)
        assert torch.allclose(result, torch.ones((10, 3)))

    def test_no_shifts_applied_no_fallback(self):
        """Test when no shifts are applied and ensure_at_least_one=False."""

        def dummy_generator(n):
            return torch.ones((n, 3))

        config = ShiftConfig("test", 0.0, dummy_generator)
        generator = ShiftGenerator([config], ensure_at_least_one=False)

        result = generator(10)
        assert result.shape == (10, 3)
        assert torch.allclose(result, torch.zeros((10, 3)))

    @patch('random.choice')
    def test_fallback_mechanism(self, mock_choice):
        """Test fallback when no shifts are selected but ensure_at_least_one=True."""

        def generator1(n):
            return torch.ones((n, 3))

        def generator2(n):
            return torch.ones((n, 3)) * 2

        config1 = ShiftConfig("test1", 0.0, generator1)  # Zero probability
        config2 = ShiftConfig("test2", 0.5, generator2)  # Non-zero probability

        mock_choice.return_value = config2

        generator = ShiftGenerator([config1, config2],
                                   ensure_at_least_one=True)

        with patch.object(config2, 'should_apply', return_value=False):
            result = generator(10)
            assert result.shape == (10, 3)
            # Should use config2 as fallback
            assert torch.allclose(result, torch.ones((10, 3)) * 2)
            mock_choice.assert_called_once()

    def test_multiple_shifts_combined(self):
        """Test that multiple shifts are properly combined."""

        def generator1(n):
            return torch.ones((n, 3))

        def generator2(n):
            return torch.ones((n, 3)) * 2

        config1 = ShiftConfig("test1", 1.0, generator1)
        config2 = ShiftConfig("test2", 1.0, generator2)

        generator = ShiftGenerator([config1, config2])
        result = generator(10)

        assert result.shape == (10, 3)
        # Should be sum of both generators
        assert torch.allclose(result, torch.ones((10, 3)) * 3)


class TestSelectRandomIndices:
    """Test the select_random_indices function."""

    def test_basic_functionality(self):
        """Test basic index selection."""
        sequence = torch.arange(10)
        indices = select_random_indices(sequence, max_sequence_length=3)

        assert isinstance(indices, torch.Tensor)
        assert indices.dtype == torch.long
        assert len(indices) <= 3
        assert len(indices) >= 1
        assert torch.all(indices >= 0)
        assert torch.all(indices < 10)

    def test_empty_sequence(self):
        """Test with empty sequence."""
        sequence = torch.tensor([])
        indices = select_random_indices(sequence)

        assert len(indices) == 0
        assert indices.dtype == torch.long

    def test_single_element_sequence(self):
        """Test with single element sequence."""
        sequence = torch.tensor([5])
        indices = select_random_indices(sequence)

        assert len(indices) == 1
        assert indices[0] == 0  # Index, not value

    def test_indices_are_consecutive(self):
        """Test that selected indices are consecutive."""
        sequence = torch.arange(20)
        indices = select_random_indices(sequence, max_sequence_length=5)

        if len(indices) > 1:
            diffs = indices[1:] - indices[:-1]
            assert torch.all(diffs == 1)

    def test_edge_only_option(self):
        """Test edge_only parameter."""
        sequence = torch.arange(20)

        # Test multiple times to check edge behavior
        for _ in range(50):
            indices = select_random_indices(sequence, max_sequence_length=3,
                                            edge_only=True)

            # Should start at either beginning or end
            start_idx = indices[0].item()
            assert start_idx == 0 or start_idx >= 17  # Near the end

    def test_invalid_input(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            select_random_indices("not a tensor")

        with pytest.raises(ValueError):
            select_random_indices(torch.randn(3, 3))  # 2D tensor


class TestTrajectoryGeneration:
    """Test trajectory generation functions."""

    def test_generate_smooth_trajectory(self):
        """Test smooth trajectory generation."""
        grid_x, grid_y, grid_z = generate_smooth_trajectory(grid_resolution=3)

        assert grid_x.data.shape == torch.Size([1, 3])
        assert grid_y.data.shape == torch.Size([1, 3])
        assert grid_z.data.shape == torch.Size([1, 3])

        # Check that values are in expected range [-1, 1]
        for grid in [grid_x, grid_y, grid_z]:
            assert torch.all(grid.data >= -1)
            assert torch.all(grid.data <= 1)

    def test_generate_shift_trajectory(self):
        """Test shift trajectory generation."""
        num_points = 50
        timesteps, x_pos, y_pos, z_pos = generate_shift_trajectory(num_points)

        assert len(timesteps) == num_points
        assert len(x_pos) == num_points
        assert len(y_pos) == num_points
        assert len(z_pos) == num_points

        # Check timesteps are properly spaced
        assert torch.allclose(timesteps, torch.linspace(0, 1, num_points))

        # Check that positions are detached (no gradients)
        assert not x_pos.requires_grad
        assert not y_pos.requires_grad
        assert not z_pos.requires_grad

    def test_generate_shift_trajectory_input_validation(self):
        """Test that generate_shift_trajectory validates input properly."""

        # Test valid input
        timesteps, x_pos, y_pos, z_pos = generate_shift_trajectory(4)
        assert len(timesteps) == 4
        assert len(x_pos) == 4
        assert len(y_pos) == 4
        assert len(z_pos) == 4

        # Test invalid inputs
        with pytest.raises(ValueError,
                           match="num_points must be at least 4, got 1"):
            generate_shift_trajectory(1)

        with pytest.raises(ValueError,
                           match="num_points must be at least 4, got 3"):
            generate_shift_trajectory(3)

        with pytest.raises(ValueError,
                           match="num_points must be at least 4, got 0"):
            generate_shift_trajectory(0)

        with pytest.raises(ValueError,
                           match="num_points must be at least 4, got -5"):
            generate_shift_trajectory(-5)


class TestGeneratorCreators:
    """Test the generator creator functions."""

    def test_create_trajectory_generator(self):
        """Test trajectory generator creation."""
        max_shift = 10.0
        generator = TrajectoryGenerator(max_shift)

        num_points = 20
        shifts = generator(num_points)

        assert shifts.shape == (num_points, 3)
        assert isinstance(shifts, torch.Tensor)

        # Check that mean is approximately zero (centered)
        mean_shift = torch.mean(shifts, dim=0)
        assert torch.allclose(mean_shift, torch.zeros(3), atol=1e-6)

    def test_create_jitter_generator(self):
        """Test jitter generator creation."""
        max_std = 5.0
        generator = JitterGenerator(max_std)

        num_points = 100
        shifts = generator(num_points)

        assert shifts.shape == (num_points, 3)
        assert isinstance(shifts, torch.Tensor)

        # Check that jitter has reasonable variance
        # (this is probabilistic, so we use loose bounds)
        std_dev = torch.std(shifts)
        assert std_dev > 0  # Should have some variance
        assert std_dev <= max_std * 2  # Shouldn't be too large

    def test_create_outlier_generator(self):
        """Test outlier generator creation."""
        max_shift = 20.0
        generator = OutlierGenerator(max_shift, max_sequence_length=3,
                                             edge_only=False)

        num_points = 50
        shifts = generator(num_points)

        assert shifts.shape == (num_points, 3)

        # Most points should be zero (only outliers are non-zero)
        non_zero_mask = torch.any(shifts != 0, dim=1)
        num_outliers = torch.sum(non_zero_mask).item()

        assert num_outliers >= 1
        assert num_outliers <= 3  # max_sequence_length

        # Check outlier magnitudes are within expected range
        outlier_shifts = shifts[non_zero_mask]
        if len(outlier_shifts) > 0:
            assert torch.all(torch.abs(outlier_shifts) <= max_shift)


class TestDefaultGenerator:
    """Test the default generator creation."""

    def test_create_default_generator(self):
        """Test default generator creation and basic functionality."""
        generator = create_default_generator()

        assert callable(generator)

        num_points = 30
        shifts = generator(num_points)

        assert shifts.shape == (num_points, 3)
        assert isinstance(shifts, torch.Tensor)

    def test_zero_probabilities_respected(self):
        """Test that zero probabilities are properly respected."""
        # Set all probabilities to zero except jitter
        generator = create_default_generator(
            trajectory_probability=0.0,
            jitter_probability=1.0,
            outlier_probability=0.0,
            high_tilt_outlier_probability=0.0,
            jitter_max_std=1.0
        )

        num_points = 50
        shifts = generator(num_points)

        # Should only have jitter (small random values everywhere)
        assert shifts.shape == (num_points, 3)
        # All points should have some shift (jitter)
        assert torch.all(torch.any(shifts != 0, dim=1))

    def test_all_zero_probabilities(self):
        """Test behavior when all probabilities are zero."""
        generator = create_default_generator(
            trajectory_probability=0.0,
            jitter_probability=0.0,
            outlier_probability=0.0,
            high_tilt_outlier_probability=0.0
        )

        num_points = 20
        shifts = generator(num_points)

        # Should return zeros since no shifts can be applied
        assert shifts.shape == (num_points, 3)
        assert torch.allclose(shifts, torch.zeros((num_points, 3)))


class TestProjectShifts3DTo2D:
    """Test 3D to 2D projection function."""
    def test_raise_error_with_wrong_matrix_shape(self):
        shifts_3d = torch.randn((10, 3))
        matrices = torch.randn((10, 3, 3))
        with pytest.raises(ValueError):
            project_shifts_3d_to_2d(shifts_3d, matrices)

    def test_raise_error_with_mismatch_of_points(self):
        shifts_3d = torch.randn((5, 3))
        matrices = torch.randn((2, 2, 3))
        with pytest.raises(RuntimeError):
            project_shifts_3d_to_2d(shifts_3d, matrices)

    def test_basic_projection(self):
        """Test basic 3D to 2D projection."""
        num_points = 10
        shifts_3d = torch.randn((num_points, 3))  # Random 3D shifts
        projection_matrices = torch.randn(
            (num_points, 2, 3))  # Random projection matrices

        shifts_2d = project_shifts_3d_to_2d(shifts_3d, projection_matrices)

        assert shifts_2d.shape == (num_points, 2)
        assert isinstance(shifts_2d, torch.Tensor)

    def test_identity_projection(self):
        """Test projection with identity-like matrices."""
        num_points = 5
        shifts_3d = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
             [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])

        # Create projection matrices that just take first two dimensions
        projection_matrices = torch.zeros((num_points, 2, 3))
        projection_matrices[:, 0, 0] = 1.0  # Take first dimension
        projection_matrices[:, 1, 1] = 1.0  # Take second dimension

        shifts_2d = project_shifts_3d_to_2d(shifts_3d, projection_matrices)

        expected = shifts_3d[:, :2]  # Should be first two dimensions
        assert torch.allclose(shifts_2d, expected)

    def test_zero_shifts(self):
        """Test projection with zero shifts."""
        num_points = 3
        shifts_3d = torch.zeros((num_points, 3))
        projection_matrices = torch.randn((num_points, 2, 3))

        shifts_2d = project_shifts_3d_to_2d(shifts_3d, projection_matrices)

        assert shifts_2d.shape == (num_points, 2)
        assert torch.allclose(shifts_2d, torch.zeros((num_points, 2)))


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from generator creation to 2D projection."""
        # Create a generator
        generator = create_default_generator(
            trajectory_probability=1.0,  # Ensure trajectory is applied
            jitter_probability=0.0,
            outlier_probability=0.0,
            high_tilt_outlier_probability=0.0
        )

        # Generate 3D shifts
        num_points = 20
        shifts_3d = generator(num_points)

        assert shifts_3d.shape == (num_points, 3)

        # Create projection matrices
        projection_matrices = torch.eye(2, 3).unsqueeze(0).repeat(num_points,
                                                                  1, 1)

        # Project to 2D
        shifts_2d = project_shifts_3d_to_2d(shifts_3d, projection_matrices)

        assert shifts_2d.shape == (num_points, 2)
        # Should be first two dimensions of 3D shifts
        assert torch.allclose(shifts_2d, shifts_3d[:, :2])

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with random seed."""
        # Set seeds
        torch.manual_seed(42)
        random.seed(42)

        generator = create_default_generator()
        shifts1 = generator(30)

        # Reset seeds
        torch.manual_seed(42)
        random.seed(42)

        shifts2 = generator(30)

        # Results should be identical
        assert torch.allclose(shifts1, shifts2)


# Fixtures for common test data
@pytest.fixture
def sample_shift_config():
    """Fixture providing a sample ShiftConfig."""

    def dummy_generator(n):
        return torch.ones((n, 3))

    return ShiftConfig("test_shift", 0.5, dummy_generator)


@pytest.fixture
def sample_generator():
    """Fixture providing a sample ShiftGenerator."""

    def dummy_generator(n):
        return torch.ones((n, 3))

    config = ShiftConfig("test_shift", 1.0, dummy_generator)
    return ShiftGenerator([config])


# Parametrized tests
@pytest.mark.parametrize("num_points", [4, 10, 50, 100])
def test_generator_output_shapes(num_points):
    """Test that generators produce correct output shapes for various input sizes."""
    generator = create_default_generator()
    shifts = generator(num_points)
    assert shifts.shape == (num_points, 3)


@pytest.mark.parametrize("max_shift", [0.1, 1.0, 10.0, 100.0])
def test_trajectory_generator_scaling(max_shift):
    """Test that trajectory generator respects max_shift parameter."""
    generator = TrajectoryGenerator(max_shift)
    shifts = generator(50)

    # The actual maximum might be less due to centering, but should be reasonable
    max_observed = torch.max(torch.abs(shifts)).item()
    assert max_observed <= max_shift * 2  # Allow some tolerance for centering
