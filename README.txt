0) Requirements
Please note that we provide a docker image to run the code.
https://hub.docker.com/r/cliecy/brx/tags
If you run the code in the container,please mount the code folder to the volume of the container and run the code with python environment /opt/newbrx/bin/python

1) Introduction
This code is a binary-version for Cascade-VAE (https://github.com/tsudalab/rxngenerator) which means the latent space is all binary tensor which is either 0 or 1.

You could read the original version for details.We have created the dataset and filtered it(./data/data.txt).

2) Training
To train the model, type the following command:
python trainvae.py -w 200 -l 100 -d 2 -v "./weights/data.txt_fragmentvocab.txt" -t "./data/data.txt"
Then the model will saved in weights folder.

check the result with sample.py

python sample.py -w 200 -l 100 -d 2 -v "./weights/data.txt_fragmentvocab.txt" -t "./data/data.txt" --w_save_path "weights/latent100NoEarlyStopBeta3_20Epoch.npy"


3) Ising Machine
The Ising Machine part is based on https://github.com/tsudalab/bVAE-IM

This project now uses the Gurobi solver only.
Place a valid Gurobi license file `gurobi.lic` in the project root
or set the environment variable `GRB_LICENSE_FILE` to its path.
The runtime will automatically pick it up from `gurobi.lic` if present.

Plese type the following command to run Ising Machine optimization
(If you use SLURM,then it's ok to use this,otherwise you need to change the code in bvae.py.Also you can use other random seed you need.Despite whether using diffrent random seed,you should check if you are using SLURM to run the code)
python bvae_im.py -w 200 -l 100 -d 2 -r $SLURM_ARRAY_TASK_ID  -v "fragmentvocab_path" -t "data_path" -s "saved_model_path" -m "qed"
Example:
python bvae_im.py -w 300 -l 100 -d 2 -r 1 -t "./data/data.txt" -s "/home/gzou/fitcheck/newnnn/brxngenerator-master/weights/hidden_size_300_latent_size_100_depth_2_beta_1.0_lr_0.001/bvae_iter-30-with.npy" -m "qed"
Then the optimization result will be saved in Results folder.

4) Error-Correcting Codes (ECC) - NEW FEATURE

This implementation now supports Error-Correcting Codes to improve binary VAE generation quality and robustness.

ECC Quickstart (subset training for testing):

a) Training with ECC (few epochs, small subset):
python trainvae.py -n 0 --subset 2000 --ecc-type repetition --ecc-R 3

b) Sampling with ECC:
python sample.py -w 200 -l 120 -d 2 -t "./data/data.txt" -v "./weights/data.txt_fragmentvocab.txt" --w_save_path "path/to/weights.npy" --ecc-type repetition --ecc-R 3 --subset 500

c) Evaluation metrics:
python eval_ecc_simple.py --samples 1000 --latent-size 12 --smoke-qubo

d) Compare ECC vs no ECC:
python eval_ecc_simple.py --samples 2000

ECC Parameters:
--ecc-type: 'none' (default) or 'repetition'
--ecc-R: Repetition factor (2 or 3, default 3)
--subset: Limit dataset size for faster testing

Expected improvements with ECC:
- Bit Error Rate (BER): ~80-90% reduction
- Word Error Rate (WER): ~90-95% reduction
- Better confidence calibration (lower entropy)

Note: When using ECC, latent_size must be divisible by ecc_R.
Example: latent_size=12 with ecc_R=3 uses 4 information bits encoded as 12 code bits.


