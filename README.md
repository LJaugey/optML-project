# Optimization for Machine Learning - Miniproject
## Performance of Adaptive Stochastic Gradient Descent (AdSGD) on select datasets, for a simple neural network model 

### Description
This project tests the Adaptative Stochastic Gradient Descent method described in the paper ["Adaptive Gradient Descent without Descent"](https://arxiv.org/pdf/1910.09529.pdf), by Y. Malitsky and K. Mishchenko. A portion of the code is a modified version of what is available at the repository https://github.com/ymalitsky/adaptive_GD. 

### Datasets

- MNIST: A dataset of handwritten  digits from 0 to 9 in grayscale. Images of size $1\times 28 \times 28$. Training set: 60'000 samples. Test set: 10'000 samples. 
- FMNIST: A dataset of 10 classes of pictures of clothing in grayscale. Images of size $1\times 28 \times 28$. Training set: 60'000 samples. Test set: 10'000 samples. 
- EMNIST: A dataset of handwritten letters from a to z in grayscale. There are 26 classes. Images of size $1\times 28 \times 28$. Training set: 124'800 samples. Test set: 20'800 samples. 
- CIFAR10: A dataset of 10 different classes in color. Images of size $3\times 32 \times 32$. Training set: 50'000 samples. Test set: 10'000 samples. 

### Prerequisites
- A working version of Python 3.7
- A working version of MPI installed
- Python Packages : numpy, torch, torchvision, torchaudio, mpi4py, Pathlib

### Code
- ``train.py``: Run script.
- ``lenet5.py``: NN model based on [LeNet5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).
- ``optimizer.py``: Implementation of AdSGD.
- ``helpers.py``: Useful functions for training model and saving results.
- ``plotter.py``: Produces the graphs from the results of the experiments.
- ``launch.sh``: Script to run all the training experiments on the Izar cluster (EPFL). 


### How to run the code on your computer
- ``python train.py optimizer dataset lr_amplifier lr_damping``

where optimizer can be chosen from 
- ``sgdm`` : SGDm with learning rate $\lambda=0.01$ and momentum term $\beta = 0.9$.
- ``adam`` : Adam with learning rate $\lambda = 0.001$.
- ``adgd`` : AdSGD with learning rate amplifier $\gamma = $``lr_amplifier`` and learning rate damping $\delta=\frac{1}{\alpha}=$``lr_damping``.

dataset can be ``mnist``, ``fmnist``, ``emnist`` or ``cifar10`` and ``lr_amplifier`` and ``lr_damping`` are doubles. These last two values have no impact if an optimizer other than ``adgd`` is chosen.

Example:
- ``python train.py adgd mnist 0.1 1.0``

It should be noted that this code requires MPI to run. One can easily modify the code in ``train.py``, so the code runs without using MPI. Simply remove the lines
```
14 comm = MPI.COMM_WORLD
15 size = comm.Get_size()
16 rank = comm.Get_rank()
```
and remove any reference to ``rank`` in the code.

### How to run the code on the Izar cluster
- Load the modules : ``module load gcc mvapich2 python``
- Create a virtual environment, install python and all required packages. 
- Run the launch script: ``sbatch launch.sh``