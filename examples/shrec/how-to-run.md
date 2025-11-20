# Instructions for running on SHREC

## Download and prepare input

To download and prepare training input, run preproc.py and specify a `--download-dir` via the script arguments. This does the following:

* download the data from Zenodo
* use a helper function from miss_alignment.data.io to convert the
downloaded pickles for both the ground-truth set and the tiltxcorr-aligned
set to .json files for miss-alignment (each .json file will point to a generated .xml and .st file)

## Setup project

Create a project folder with the following layout. .json files should be directly copied to the iter0 folder without the .xml and .st files, as they point to their location on disk:

```
shrec_benchmark/              # this can have your desired name
├── iter0/                    # this name is strict
│   ├── model_0.json
│   ├── model_1.json
│   └── ...
├── ground_truth/
│   ├── model_0.json
│   ├── model_1.json
│   └── ...
├── config.yaml
└── init_weights.ckpt
```

Fill config.yaml with the following text, while updating the 'training_directory' and 'model_checkpoint':

```yaml
general:
  # MissAlignment iteratively trains models and realigns the tilt-series
  training_directory: /path/to/project/  # path to directory for writing iteration output
  apply_ctf: False                # set to True if CTF estimates are available in .xml
  iteration_settings:             # defines all iterations to run (length determines number of iterations)
    - { downsample: 2, alignment: global }
    - { downsample: 2, alignment: global }
    - { downsample: 2, alignment: global }
    - { downsample: 2, alignment: global }
    - { downsample: 1, alignment: global }
    - { downsample: 1, alignment: global }
    - { downsample: 1, alignment: global }
    - { downsample: 1, alignment: global }
  seed: 45132

model_training:
  # used to initialize model weights
  model_architecture: 'default'
  model_checkpoint: /path/to/project/init_weights.ckpt  # starting weights
  loss_margin: 0.5
  learning_rate: 1.0e-3
  # Set to zero to disable weight decay (i.e. the AdamW optimizer)
  weight_decay: 1.0e-4
  max_epochs_per_iteration: 30  # absolute maximum of stopping
  warmup_steps: 500
  multistep_lr_scheduler:
    milestones: [5, 15]
    gamma: 0.5                  # multiply learning rate by this value at the milestone epochs

data_loading:
  batch_size: 32                  # these values have been used as defaults:
  patch_size: 96                  # batch size 32 | reconstruction size 128 ^ 3
  steps_per_epoch: 1000           # this defines the size of an epoch

# this configures the shift generation for the contrastive training of MissAlignment
# all parameters are in units of 10A (nm) as we assume to work with tilt-series sampled to 10A
shift_generation:
  trajectory_probability: .5      # these make bananas and birds
  trajectory_max_shift: 10.0      # (10 A) trajectories fall somewhere between +/- f
  jitter_probability: .5
  jitter_max_std: 2.0             # (10 A) maximum standard deviation of a normal distribution
  outlier_probability: .5
  outlier_max_shift: 20.0         # (10 A) outlier can maximally go to this value
  fracture_probability: .5
  fracture_max_shift: 30.0       # (10 A) specific strong shifts for high tilt images up to this value

tilt_series_alignment:
  patch_size: 96      # same as training patch size
  patch_overlap: 0.0  # tolerated overlap between patches used for optimizing the alignment
  batch_size: 16      # amount of patches simultaneously reconstructed in memory -> the more the merrier 
```

## Starting the program

To run the model use the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MISS_ALIGNMENT_RECON_POOL_SIZE=200 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 miss-alignment train --config-file /path/to/conf.yaml --reconstruction-workers 3 --dataloader-workers 3 --n-devices 4
```

If you modify the computing resources it can be handy to run with the option `--monitor-production-and-consumption` to track the consumption/production ratio. The value should be around 2. The option does not (yet) robustly work throughout iterations. So, you should only run it for a few epochs to get the gist of the ratio, cancel the program, and restart without the option.

## Evaluate alignment performance against ground truth

After running you can evaluate the results against the ground truth alignment with the program 'compare_to_ground_truth.py'. The arguments should be self-explanatory.

