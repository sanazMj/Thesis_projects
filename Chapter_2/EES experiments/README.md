The EES dataset used in this project is a synthetic version of the private dataset provided by Communication Research Centre Canada. 
The details on the dataset's structure is provided in [paper](http://dx.doi.org/10.1007/s00521-020-05656-2). 
One can use the following directions to create a synthetic version of the dataset:
* 9x9 dataset generation process:
  * Generate every binary pattern in a 5x5 cube. 
  * Use half of the cube, rotate it and combine the rotations to have eight fold symmetry.
# Training:
You can train the model by calling [run_multiple_experiments.py](https://github.com/RyersonU-DataScienceLab/Sanaz_VARGAN/new/main/EES%20experiments). Set the configuration details as you wish. 
