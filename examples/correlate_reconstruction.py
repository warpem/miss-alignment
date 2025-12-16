from pathlib import Path
from miss_alignment.alignment.correlation import (
    calculate_cross_correlation,
    get_shift_from_correlation_image,
)
from miss_alignment.alignment.utils import project_volume_shift_to_image_alignment
# update file reading from warpylib
from miss_alignment.data.io import read_tomogram_from_pickle, save_tomogram_to_pickle


dir_start = sorted(Path('run1/iter0').glob('*.pickle'))
dir_final = sorted(Path('run1/iter10').glob('*.pickle'))
out_dir = Path('run1/iter10_centered')
out_dir.mkdir(exist_ok=True)

for start_file, final_file in zip(dir_start, dir_final):
    print(start_file.name)
    out_file = out_dir / start_file.name
    
    tilt_series_start = read_tomogram_from_pickle(start_file)
    tilt_series_start.to('cuda')
    tilt_series_final = read_tomogram_from_pickle(final_file)
    tilt_series_final.to('cuda')

    # reconstruct
    vol_start = tilt_series_start.reconstruct_tomogram((200, 500, 500), 256)
    vol_final = tilt_series_final.reconstruct_tomogram((200, 500, 500), 256)
    
    # get the 3d shift
    n_tilts, _, _ = tilt_series_start.images.shape
    correlation = calculate_cross_correlation(
        vol_start, vol_final,
    )
    shift_3d = get_shift_from_correlation_image(correlation)
    shift_3d = -1 * shift_3d  # get the forward shift for the imaging model
    shift_3d = shift_3d.repeat(n_tilts, 1)
    
    # project 3d shift to 2d and correct the final alignments
    shifts_2d = project_shifts_3d_to_2d(
        shift_3d, tilt_series_final.projection_matrices[..., 1:3, :3]
    )
    tilt_series_final.sample_translations += shifts_2d

    # write the tilt_series to output_folder
    tilt_series_final.to('cpu')
    save_tomogram_to_pickle(tilt_series_final, out_file)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(mean_diff_initial[:, 0], label="initial y")
    ax[0].plot(mean_diff_final[:, 0], label="final y")
    ax[1].plot(mean_diff_initial[:, 1], label="initial x")
    ax[1].plot(mean_diff_final[:, 1], label="final x")
    ax[0].legend()
    ax[1].legend()

    plt.savefig(output_directory / f"{tilt_series_name}_yx_diff_per_tilt.png")
