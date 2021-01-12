## A Factor graph/MRF parameter learning/estimation package via stochastic maximum likelihood
The current implementation only supports binary variables, with values in e.g. {-1, +1} and uses some Numba to speed
 up the Numpy-based algorithms (by a little bit) -- there are certainly more potential speed-ups :) 

## Dependencies
Currently only tested with Python3.7: 

    conda create -n my-fg python=3.7
    conda activate my-fg
    pip install -r requirements.txt

## Expected Factor Graph Definition
The package assumes the MRF to be in exponential form, that is the distribution over the random variables 
<img src="https://latex.codecogs.com/svg.latex?x"> can be written as:  
<img src="https://latex.codecogs.com/svg.latex?p_\theta(x)=\frac{1}{Z_\theta}\exp(\theta^T\phi(x)),">

where <img src="https://latex.codecogs.com/svg.latex?Z_\theta"> is the partition function (a normalization constant), 
<img src="https://latex.codecogs.com/svg.latex?\theta"> are the model's parameter, and  <img src="https://latex.codecogs.com/svg.latex?\phi(x)"> are the sufficient statistics/factors/potentials.
 
## Example
If you want to see/run the example: click [here](examples/bb/supervised_vs_latent.ipynb). And, if you want to
compare to Snorkel you need to install it as follows: ``pip install snorkel``