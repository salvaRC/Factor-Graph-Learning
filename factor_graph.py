import numpy as np


# from numba import jit, jitclass, int32, float32, byte, boolean


# @jit(nopython=True)
def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


class SufficientStats:
    def __init__(self, funcs):
        """
        :param funcs: list of callable functions
        """
        self.funcs = funcs

    def __call__(self, variable_values):
        return np.array([
            func(variable_values) for func in self.funcs
        ]).reshape(-1)


class FactorGraph:
    r"""
    Binary Random Variables have IDs \in {0, 1, 2, .., #Vars - 1} and possible values in { +1, polarities[i] }
    """

    def __init__(self, n_vars, potentials, polarities):
        """
        :param n_vars: #Random Variables
        :param potentials: list of tuples of (callable function, Image cardinality)
                                Image cardinality just means the number of values the function returns (>= 1)
        :param polarities: a list of integers, where the i-th entry corresponds to the possible value (besides +1)
                            that variable i can take.
        """
        # assert all([pol in [-1, 0, 1] for pol in polarities]), "Each polarity must be either positive (+1) or negative (-1), or abstain (0)"
        assert len(polarities) == n_vars
        self.n_variables = n_vars
        self.sufficient_statistics = SufficientStats([func for func, _ in potentials])
        self.n_params = sum([img_card for _, img_card in potentials])
        self.parameter = np.zeros(self.n_params, dtype=np.float32)
        self.polarities = polarities
        self.chain_sampler_tmp = None
        # self.parameter[:] = 0.7

    def conditional(self, target_varID, values):
        r"""
        :param values: of shape (#Vars,).
                        Note that the values corresponding to the target variable is not used, i.e.
                        values[target_varID] can be set to any arbitrary value.
        :param target_varID: ID \in {0, .., #Vars-1} of the variable to be inferred
        :return:The conditional P(Var_ID = 1 | Vars_-ID) of the given variable given all the others.
        """
        vals_copy = values.copy()
        vals_copy[target_varID] = 1
        pos = self.sufficient_statistics(vals_copy)

        vals_copy[target_varID] = self.polarities[target_varID]  # e.g. -1
        neg = self.sufficient_statistics(vals_copy)
        return sigmoid(
            self.parameter @ (pos - neg)
        )

    def predict(self, target_varID, values):
        r"""
        :param values: of shape (#Vars,).
                        Note that the values corresponding to the target variable is not used, i.e.
                        values[target_varID] can be set to any arbitrary value.
        :param target_varID: ID \in {0, .., #Vars-1} of the variable to be inferred
        :return: 1 if P(target_var = 1| Vars_{-target_var}) > 0.5, and the opposite polarity otherwise
        """
        return +1 if self.conditional(target_varID, values) > 0.5 else self.polarities[target_varID]

    def predict_proba(self, data, target_varID=0):
        r"""
        :param data: of shape (#samples, #Vars).
                    Note that the values corresponding to the target variable are not used, i.e.
                    data[:, target_varID] can be set to any arbitrary value.
        :param target_varID: ID \in {0, .., #Vars-1} of the variable to be inferred
        :return: the soft labels computed from the learned posterior of shape (#samples,)
        """
        Y_soft = [
            self.conditional(target_varID, observed_vals)
            for observed_vals in data
        ]
        return np.array(Y_soft)

    def fit(self, observations, lr=0.01, burn_ins=10, n_epochs=25, gibbs_samples=20, batch_size=32, n_gibbs_samples=50,
            verbose=True):
        """
         This method will fit the Factor Graph/MRF parameter to the given data/observation using:
          Stochastic maximum likelihood for fitting an Markov Random Field (MRF) algorithm from
                Machine Learning: A Probabilistic Perspective, K. Murphy (page 680)
        :param observations: The data to learn from, a (#samples, #variables) array
        :param lr: learning rate
        :param burn_ins: #TODO
        :param n_epochs: Epochs of SGD
        :param batch_size: Batch size to compute the gradient on
        :param n_gibbs_samples: How many Gibbs chain samples to use for approximating the expectation of the sufficient statistics
        :return:
        """
        assert set(np.unique(observations)) == {-1, 0, 1}, "Unsupported labels!"
        assert observations.shape[1] == self.n_variables, f"Observations should have {self.n_variables} columns!"
        n_samples = observations.shape[0]
        observations = np.array(observations)
        self.chain_sampler_tmp = observations[0, :]  # init to some arbitrary value

        for epoch in range(n_epochs):
            if verbose and epoch % 25 == 0:
                print(f"Epoch {epoch}...")
            permutation = np.random.permutation(n_samples)  # shuffle the training set/observations
            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i + batch_size]
                batch = observations[indices, :]
                """ Get estimate of the expectation of the sufficient statistics by repeated MCMC sampling"""
                approx_EsuffStats, gradient = np.zeros(self.n_params), np.zeros(self.n_params)
                for _ in range(n_gibbs_samples):
                    approx_EsuffStats += self.sufficient_statistics(
                        self.gibbs_sample(n_steps=gibbs_samples)  # single sample of the random variables
                    )
                approx_EsuffStats /= n_gibbs_samples
                """ Compute the gradient over the mini-batch """
                for observation in batch:
                    gradient += self.sufficient_statistics(observation) - approx_EsuffStats
                gradient /= len(batch)
                """ Stochastic gradient ascent -- minimization is for Losers xD """
                self.parameter += lr * gradient

    def gibbs_sample(self, n_steps=10):
        if n_steps < 1:
            return self.chain_sampler_tmp.copy()
        k = 0
        while True:
            for varID in range(self.n_variables):  # chain
                r""" 
                Set x_i = argmax_{x_i \in values(X_i)} P_\theta (x_i | x_{-i})
                I.e. set the new x_i to the most probable state, given all the other variables.
                 """
                self.chain_sampler_tmp[varID] = self.predict(varID, self.chain_sampler_tmp)
                k += 1
                if k == n_steps:
                    return self.chain_sampler_tmp.copy()
