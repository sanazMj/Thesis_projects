# Dataset:
The EES dataset used in this project is private and provided by Communication Research Centre Canada. The details on the dataset's structure is provided in [paper](http://dx.doi.org/10.1007/s00521-020-05656-2). One can use the following directions to create a synthetic version of the dataset:
* 9x9 dataset generation process:
  * Generate every binary pattern in a 5x5 cube. 
  * Use half of the cube, rotate it and combine the rotations to have eight fold symmetry.
# Training:
You can train the model by calling [run_multiple_experiments.py](https://github.com/sanazMj/Thesis_projects/blob/main/Chapter_1/Codes/EES%20experiments/run_multiple_experiments.py). Set the configuration details as you wish. 

Configuations include:
* Model_structure: { 'Convolutional' ,'FF','Convolutional', 'ConvOriginal'}
* Model_type :{'Conditional', 'Vanilla'}
* categorization : {2, 2.3, 8}
* full_image: {False, True} 
* Pixel_Full:{9, 19}
* num_epochs
* zdim: dimension of the noise vector
* batch_size
* channels
*  ndf = 2048 # Num Discriminator Features
*  ngf = 512 # Num Generator Features
