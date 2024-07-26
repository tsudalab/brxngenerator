0) Requirements
- Linux or MacOS (We run our experiments on Linux CPU server and MacOs)
- RDKit (version='2020.09.5')
- Python (version=3.8.7)
- Pytorch (version=2.3.0)
- amplify (version=0.9.1)

1) Introduction
This code is a binary-version for Cascade-VAE (https://github.com/tsudalab/rxngenerator). 
You could read the original version for details.
For dataset, you could use this file '/data/synthetic_routes.txt' directly.

2) Training
To train the model, type the following command:
python trainvae.py -w 200 -l 50 -d 2 -v "weights/data.txt_fragmentvocab.txt" -t "data/data.txt"

3) Ising Machine
The Ising Machine part is based on https://github.com/tsudalab/bVAE-IM/im

To run it, you need to register one token from Amplify.
The token can be registered freely at https://amplify.fixstars.com/en/.

Plese type the following command to run Bayesian optimization:
python bvae_im.py -w 200 -l 50 -d 2 -r 1 -v "fragmentvocab_path" -t "data_path" -s "saved_model_path" -m "qed"
Example:
python bvae_im.py -w 200 -l 50 -d 2 -r 1 -v "weights/data.txt_fragmentvocab.txt" -t "data/data.txt" -s "data/data_vae_iter-1.npy" -m "qed"



