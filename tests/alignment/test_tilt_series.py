from miss_alignment.alignment.tilt_series import generate_position_grid


def test_generate_position_grid():
    grid = generate_position_grid(
        volume_dimensions_physical=(5120, 5120, 1800),
        pixel_size=10,
        patch_size=96,
        patch_overlap=0.1
    )
    assert grid.shape == (50, 3)

    grid = generate_position_grid(
        volume_dimensions_physical=(5120, 5120, 1800),
        pixel_size=20,
        patch_size=512,
        patch_overlap=0.1
    )
    assert grid.shape == (1, 3), "grid should always give at least 1 position"