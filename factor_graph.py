import numpy as np
import torch
from torch import sigmoid
from sampler import ChainGibbsSampler


class Variable:
    r"""
    Models a binary random variable RV \in {0, polarity}, where polarity can either be -1 or +1.
    """
    def __init__(self, ID, polarity, neighbors, paramIDs):
        """

        :param ID: the index of this RV within the graph's RVs, i.e. the entry ID of any full observation will be associated with this RV
        :param polarity: can either be -1 or +1, depending on the value the RV can take
        :param neighbors: all IDs of RVs to which this variable has an edge to in the graph
        :param paramIDs: the associated indices of the parameter vector/sufficient statistics that depend on this RV
        """
        self.ID = ID
        self.polarity = polarity  # \in {-1, 1}
        self.neighbors = torch.tensor(neighbors).long()
        self.paramIDs = torch.tensor(paramIDs).long()

    def sample_posterior(self, parameter, values):
        """
        :param parameter: parameter vector of shape (#Variables,)
        :param values: manifestations of the random variables
        :return: polarity if P(LF_i = polarity| LF_{-i}, Y) > 0.5
                    0     if P(LF_i = polarity| LF_{-i}, Y) < 0.5

        """
        return torch.round(self.polarity * self.conditional(parameter, values))

    def conditional(self, parameter, values):
        """
        :param parameter: parameter vector of shape (#Variables,)
        :param values: manifestations of the random variables
        :return: P(Var_i = polarity| Var_{-i}) = P(LF_i = polarity| LF_{-i}, Y)
        """
        return sigmoid(
            (-1) ** (1 + values[self.ID]) * self.polarity *  # scalar
            parameter[self.paramIDs] @ values[self.neighbors]  # dot product
        )


class Label(Variable):
    r"""
    Models a binary label Y \in {-1, +1}, with (accuracy) edges to all other variables (LFs)
    """
    def __init__(self, n_LFs, ID=0):
        super().__init__(ID=ID, polarity=[-1, 1], neighbors=np.arange(1, n_LFs + 1), paramIDs=np.arange(0, n_LFs))

    def sample_posterior(self, parameter, values):
        """

        :param parameter: parameter vector of shape (#Variables,)
        :param values: manifestations of the random variables
        :return: 1 if P( label = 1 | LFs}) > 0.5
                -1 if P( label = 1 | LFs}) < 0.5
        """
        return torch.sign(self.conditional(parameter, values) - 0.5)

    def conditional(self, parameter, values):
        """

        :param parameter: parameter vector of shape (#Variables,)
        :param values: manifestations of the random variables
        :return: P( label = 1 | Variables_{-label}) = P( label = 1 | LFs})
        """
        return sigmoid(
            2 * parameter[self.paramIDs] @ values[1:]
        )  # or values[self.neighbors]


class FactorGraph:
    r"""
    Y has ID = 0
    LFs have IDs \in {1, 2, .., #LFs}

    parameter = \theta = [LF1_acc, LF2_acc, ..., LFm_acc, corr_1, corr_2, ..., corr_c]
    """

    def __init__(self, n_LFs, LF_polarities, deps=None):
        assert n_LFs == len(LF_polarities), "LF_polarities should denote the possible non-abstain value of each LF!"
        assert all([pol in [-1, 1] for pol in LF_polarities]), "Each polarity must be either positive (+1) or negative (-1)"
        if deps is None:
            deps = []

        self.n_variables = n_LFs + 1
        self.n_LFs = n_LFs
        self.n_deps = len(deps)
        self.deps = deps
        assert all(
            [1 <= i <= n_LFs and 1 <= j <= n_LFs for i, j in deps]
        ), r"Each dep should have form (i, j) \in {1,.., #LFs}!"

        self.n_params = self.n_LFs + self.n_deps
        self.parameter = torch.zeros(self.n_params)
        self.parameter[:self.n_LFs] = 0.7

        """ Set LF vars """
        self._compile_vars(deps, LF_polarities)

    def _compile_vars(self, deps, LF_polarities):
        LFs_neighbors = [
            [0] for _ in range(self.n_LFs)  # edge to Y
        ]
        LFs_params = [
            [i] for i in range(self.n_LFs)  # Accuracy param for LF_i is the i-th - 1 param
        ]

        for corr_ID, (i, j) in enumerate(deps, self.n_LFs):  # first dep_ID starts at parameter index #LFs
            assert i != j, "LF cannot have a dependency to itself!"
            LFs_neighbors[i].append(j)
            LFs_neighbors[j].append(i)
            LFs_params[i].append(corr_ID)
            LFs_params[j].append(corr_ID)

        self.vars = [Label(self.n_LFs, ID=0)]
        for i, (p, LF_neighbors, LF_params) in enumerate(zip(LF_polarities, LFs_neighbors, LFs_params), 1):
            lf = Variable(ID=i, polarity=p, neighbors=LF_neighbors, paramIDs=LF_params)
            self.vars.append(lf)

    def conditional(self, variable, values):
        """
        The conditional of the given variable given all the others.
        See the Variable/Label definition for more details.
        """
        return variable.conditional(self.parameter, values)

    def predict_proba(self, label_matrix):
        """
        :param label_matrix: of shape (#samples, #LFs)
        :return: the soft labels computed from the learned posterior of shape (#samples,)
        """
        Y_soft = [
            self.conditional(self.vars[0], labels)
            for labels in label_matrix
        ]
        return np.array(Y_soft)

    def sufficient_statistics(self, values):
        """
        :param values: Manifestation of the variables of shape (#variables,)
        :return: The sufficient statistics/factor values for the given RV manifestation
        """
        assert len(values) == self.n_variables, f"values need to be of same size as |variables|, but was {len(values)}"
        suffStats = torch.zeros(self.n_params)
        """ Accuracy parameters """
        suffStats[:self.n_LFs] = values[0] * values[1:]
        """ Correlation dependencies """
        for corr_ID, (i, j) in enumerate(self.deps, self.n_LFs):
            suffStats[corr_ID] = values[i] * values[j]
        return suffStats

    def fit(self, observations, lr=0.01, burn_ins=10, n_epochs=25, batch_size=32, n_gibbs_samples=50):
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
        observations = torch.tensor(observations).float()
        sampler = ChainGibbsSampler(variables=self.vars, x_init=observations[0, :])

        for epoch in range(n_epochs):
            if epoch % 50 == 0:
                print(f"Epoch {epoch}...")
            permutation = torch.randperm(n_samples)  # shuffle the training set/observations
            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i + batch_size]
                batch = observations[indices, :]
                """ Get estimate of the expectation of the sufficient statistics by repeated MCMC sampling"""
                approx_EsuffStats, gradient = torch.zeros(self.n_params), torch.zeros(self.n_params)
                for _ in range(n_gibbs_samples):
                    approx_EsuffStats += self.sufficient_statistics(
                        sampler(self.parameter, n_steps=20)  # single sample of the random variables
                    )
                approx_EsuffStats /= n_gibbs_samples
                """ Compute the gradient over the mini-batch """
                for observation in batch:
                    gradient += self.sufficient_statistics(observation) - approx_EsuffStats
                gradient /= batch.size()[0]
                """ Stochastic gradient ascent -- minimization is for Losers xD """
                self.parameter += lr * gradient

