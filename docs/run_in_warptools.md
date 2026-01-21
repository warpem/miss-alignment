To run miss-alignment, it's best to have some initial etomo patch tracking alignments to start off with. I put a gist on my github that shows the WarpTools commands I used for this: https://gist.github.com/McHaillet/74596b3bea760001fd253de933baafe6
Patch tracking gave me some pretty solid results, and the autolevel command afterwards also works nicely to level the sample in the tomogram. You might need to adjust the patch size for the etomo alignment and autolevel command. If you collected at 1.7 A, I guess a value of 1000 would work. If you collected 1 A, the default of 500 should work.

After that you need to update two attributes of the Warp XML, a step that can hopefully be skipped with some new releases from Warp. I also made a gist for that. You can just copy this python script (https://gist.github.com/McHaillet/117b321f504ac54d2f082bbe9bb01f16) into your `warp_tiltseries/` folder in the WarpTools project. You do need to update the tomogram shape, image shape, and pixel size at the top of the script to match your dataset. The tomogram shape ideally tightly fits your sample to prevent training the model on empty regions, but this is of course not always possible: choose something that fits the thickest samples tightly (similar to AreTomo).
* activate the miss-alginment environment (or module)
* cd into the `warp_tiltseries/` folder
* execute the script with: `python update_warp_xml.py`

Now you should be set to run miss-alignment. You need to put a miss-alignment config file to the `warp_tiltseries` directory. Update the following things in the config:
* Set the training directory to: `/path/to/your/warp/project/warp_tiltseries/` 
* The `batch_size` on the last line of the file is important for GPU occupation, ideally it maximizes GPU usage. Using cards with 24 GB RAM, a batch size of 32 works well. For smaller cards you'll need to reduce. (The batch size in the dataloading section is way less intensive so leave it at 32)

Then start the program with this, I would advise running with 4 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 miss-alignment --config-file /path/to/warp_tiltseries/config.yaml --n-workers 3 --n-devices 4 --start-at-iteration 0 --prepare-stacks 10.0
```
If the program crashed after fully finishing iterations, you can continue at later iterations with the `—start-at-iteration` where the start is indicated counting from 0.

After finishing, you need to make reconstructions with WarpTools ts_reconstruct to see the results.
