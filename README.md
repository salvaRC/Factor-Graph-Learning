## A Factor graph/MRF parameter learning package via stochastic maximum likelihood 

Currently only tested with Python3.7: 

    conda create -n my-fg python=3.7
    conda activate my-fg
    pip install -r requirements.txt

The package assumes the MRF to be in exponential form, that is the distribution over the random variables 
<img src="https://render.githubusercontent.com/render/math?math=x"> can be written as: 

<img src="https://render.githubusercontent.com/render/math?math=p_\theta(x) = \frac{1}{Z_\theta}\exp(\theta^T \phi(x)),">

where <img src="https://render.githubusercontent.com/render/math?math=Z_\theta"> is the partition function (a normalization constant), 
<img src="https://render.githubusercontent.com/render/math?math= \theta"> are the model's parameter, and  <img src="https://render.githubusercontent.com/render/math?math= \phi(x)"> are the sufficient statistics/factors/potentials.
 

If you want to see/run the example: click [here](examples/bb/supervised_vs_latent_no_numba.ipynb). And, if you want to
compare to Snorkel you need to install it as follows: ``pip install snorkel``