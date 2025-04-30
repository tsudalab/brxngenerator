0) Requirements
Please use the environment.yaml for we use the conda as package manager.

1) Introduction
This code is a binary-version for Cascade-VAE (https://github.com/tsudalab/rxngenerator). 
You could read the original version for details.We have created the dataset and filtered it(./data/data.txt).

2) Training
To train the model, type the following command:
(If you use SLURM,then it's ok to use this,otherwise you need to change the code in trainvae.py)
python trainvae.py -w 200 -l 100 -d 2 -v "./weights/data.txt_fragmentvocab.txt" -t "./data/data.txt"
Then the model will saved in weights folder.

check the result with sample.py

python sample.py -w 200 -l 100 -d 2 -v "./weights/data.txt_fragmentvocab.txt" -t "./data/data.txt" --w_save_path "weights/latent100NoEarlyStopBeta3_20Epoch.npy"


3) Ising Machine
The Ising Machine part is based on https://github.com/tsudalab/bVAE-IM/im

To run it, you need to register one token from Amplify.
The token can be registered freely at https://amplify.fixstars.com/en/.
Then change the config.yaml in config folder.

Plese type the following command to run Ising Machine optimization
(If you use SLURM,then it's ok to use this,otherwise you need to change the code in bvae.py.Also you can use other random seed you need.Despite whether using diffrent random seed,you should check if you are using SLURM to run the code)
python bvae_im.py -w 200 -l 100 -d 2 -r $SLURM_ARRAY_TASK_ID  -v "fragmentvocab_path" -t "data_path" -s "saved_model_path" -m "qed"
Example:
python bvae_im.py -w 300 -l 100 -d 2 -r 1 -t "./data/data.txt" -s "/home/gzou/fitcheck/newnnn/brxngenerator-master/weights/hidden_size_300_latent_size_100_depth_2_beta_1.0_lr_0.001/bvae_iter-30-with.npy" -m "qed"
Then the optimization result will be saved in Results folder.



