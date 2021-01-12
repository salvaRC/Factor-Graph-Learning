import numpy as np
from numba import njit


@njit
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

    def __init__(self, n_vars, potentials, polarities, priors=None, seed=77):
        """
        :param n_vars: #Random Variables
        :param potentials: list of tuples of (callable function, Image cardinality)
                                Image cardinality just means the number of values the function returns (>= 1)
        :param polarities: a list of n_vars tuples, where the i-th entry corresponds to the possible values
                            that variable i can take.
        :param priors: a list of n_vars tuples, where the i-th entry corresponds to the prior probability for variable i
                            taking the values in the order given in polarities.
        """
        assert all([len(pol) <= 2 for pol in polarities]), "Only binary variables supported"
        assert len(polarities) == n_vars
        self.n_variables = n_vars
        self.sufficient_statistics = SufficientStats([func for func, _ in potentials])
        self.n_params = sum([img_card for _, img_card in potentials])
        self.parameter = np.zeros(self.n_params, dtype=np.float32)
        self.parameter[:n_vars] = 0.7
        self.polarities = polarities
        self.priors = [[0.5, 0.5] for _ in range(n_vars)] if priors is None else priors
        self.sampler = None
        self.rng = np.random.default_rng(seed=seed)
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
        vals_copy[target_varID] = self.polarities[target_varID][0]  # e.g. +1
        pos = self.sufficient_statistics(vals_copy)

        vals_copy[target_varID] = self.polarities[target_varID][1]  # e.g. -1
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
        return self.polarities[target_varID][0] if self.conditional(target_varID, values) > 0.5 \
            else self.polarities[target_varID][1]

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
            for observed_vals in data.astype(np.int32)
        ]
        return np.array(Y_soft)

    def fit(self, observations, lr=0.01, decay=1.0, burn_ins=10, n_epochs=25, gibbs_steps_per_sample=20, batch_size=32,
            n_gibbs_samples=50, evaluate_func=None,
            verbose=True, persistive_sampling=True, eval_args=(), eval_kwargs=dict()):
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
        observations = np.array(observations, dtype=np.int32)
        self.sampler = GibbsSampler(self.polarities, self.priors, rng=self.rng)
        history = {"accuracy": [], "f1": [], "auc": [], "epochs": range(1, n_epochs + 1)}
        for epoch in range(n_epochs):
            if persistive_sampling:
                self.sampler.burn_in(self.predict, burn_ins=burn_ins)

            permutation = self.rng.permutation(n_samples)  # shuffle the training set/observations
            for i in range(0, n_samples, batch_size):
                if not persistive_sampling:
                    self.sampler = GibbsSampler(self.polarities, self.priors, rng=self.rng)
                    # print("---"*20, "\n", self.sampler.varID_to_sample)
                    self.sampler.burn_in(self.predict, burn_ins=burn_ins)
                indices = permutation[i:i + batch_size]
                batch = observations[indices, :]
                """ Get estimate of the expectation of the sufficient statistics by repeated MCMC sampling"""
                approx_Expectation, observed = np.zeros(self.n_params), np.zeros(self.n_params)
                for _ in range(n_gibbs_samples):
                    approx_Expectation += self.sufficient_statistics(
                        self.sampler.sample(self.predict)
                    )  # single sample of the random variables
                approx_Expectation /= n_gibbs_samples
                """ Observed value for the sufficient statistics """
                for observation in batch:
                    observed += self.sufficient_statistics(observation)
                observed /= len(batch)
                """ Compute the gradient over the mini-batch """
                gradient = observed - approx_Expectation
                """ Stochastic gradient descent step"""
                self.parameter -= lr * gradient
            lr *= decay  # decay stepsize
            if evaluate_func is not None:
                stats = evaluate_func(*eval_args, **eval_kwargs)
                [history[metric].append(stats[metric]) for metric in ["accuracy", "f1", "auc"]]
                print(f"Epoch {epoch}: Acc: {stats['accuracy']} | F1: {stats['f1']} | AUC: {stats['auc']}")
            elif verbose and epoch % (n_epochs / 5) == 0:
                print(f"Epoch {epoch}...")

        return history

    '''def gibbs_sample(self, n_steps=10):
        if n_steps < 1:
            return self.sampler.copy()
        k = 0
        while True:
            for varID in range(self.n_variables):  # chain
                r""" 
                Set x_i = argmax_{x_i \in values(X_i)} P_\theta (x_i | x_{-i})
                I.e. set the new x_i to the most probable state, given all the other variables.
                 """
                self.sampler[varID] = self.predict(varID, self.sampler)
                k += 1
                if k == n_steps:
                    return self.sampler.copy()'''


class GibbsSampler:
    def __init__(self, polarities, priors, seed=77, rng=None):
        self.rng = np.random.default_rng(seed=seed) if rng is None else rng
        self.n_variables = len(polarities)
        self.polarities = polarities
        self.priors = priors
        self.samples = np.array([
            self.rng.choice(pol, p=prior) for pol, prior in zip(self.polarities, self.priors)
        ], dtype=np.int32)  # observations[self.rng.choice(self.n_variables), :]  # OR: init to some arbitrary value
        self.varID_to_sample = self.rng.choice(self.n_variables)

    def burn_in(self, conditional_func, burn_ins=20):
        for _ in range(burn_ins):
            self.sample(conditional_func)

    def sample(self, conditional_func):
        r"""
            Set x_i = argmax_{x_i \in values(X_i)} P_\theta (x_i | x_{-i})
            I.e. set the new x_i to the most probable state, given all the other variables.
        """
        self.samples[self.varID_to_sample] = conditional_func(self.varID_to_sample, self.samples)
        self.varID_to_sample = (self.varID_to_sample + 1) % self.n_variables
        return self.samples.copy()

