# Project 2
This project is a collaboration between Amund J. K. Rekkavik and Kaja S. Sandnes. 
## Short Description
This project concerns itself with the use of neural networks in regression and classification analysis.
In the neural_network.py-file, we have written code for our own flexible neural network that has been used throughout this project. The results in the project come from the ipynb-files corresponding to each task of the complete description of the project. The mathematics of part a can be found the the theory part of the report.

Notes on the code can be found alongside it in the jupyter notebooks. Note that some of the code (especially the cross validation of the classification part) takes a long time to run. All figures are avalible in the /figures folder.

The random seed: "2026" has been used throughout to make results reproducable.

## Code Description
mnist.npz: file contains the data used in the classification analysis. It is imported directly into the part_e notebook.

neural_network.py: file contains the code for the feed forward neural network written for this project. It also
contains a subclass: EstimatorClassifier that can be used as an sklearn BaseEstimator class. Lastly, the file contains
some useful functions such as cost and activation functions, and their derivatives.

part_b_2.ipynb: notebook with plots and results connected to part b, c and d.

part_e.ipynb: notebook with plots and results connected to part e.

runge_data.py: file containing useful functions connected to dataset and scaling.

utils.py: Useful functions for plotting in addition to all activation functions and cost functions related to the NeuralNetwork class.

Code testing:
The test_gradient.py file asserts that the backpropagation algorithm of the NeuralNetwork class returns the same gradients
as jax automatic differentiation does. The classification part also trains an sklearn MLPClassifier with identical parameters
and random seed to verify that results obtained are comparable.
