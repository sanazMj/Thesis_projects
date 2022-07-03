# Common used files between the baseline GANs and AlphaWGANs.
* Load_data.py loads the data from .h5 file
* helper.py and utils_pre contain useful functions used in all the codes.
* constants.py contains a dictionary of training data attributes used for evaluation.
* Evaluation.py contains a set of functions used for evaluating the saved dictionaries (logged files).
* Results.py is used to evaluate a set of files.

Both experiments are evaulated at each 100 epochs over 10000 generated samples. Generator creates 10000 samples from noise and the samples are evaluated and a dictionary of evaluation metrics is saved under /logs/resultsdict_epoch(epoch).pkl. After the model is trained, run the Results.py over this .pkl files to see the results. 
## Vanilla GAN experiments
Run the main_sphere_ellipsoid by defining the dataset kind: {'sphere', 'Matlab'} and number of target points: {1, 7}. 


## AlphaWGAN experiments
Run the run_multiple.py by defining the main function (depending on the model and dataset). 


