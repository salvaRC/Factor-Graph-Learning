"""
def gibbs_sampler(x_init, variables, parameter, steps=20):
    x = x_init
    for step in range(steps):
        for i, var in enumerate(variables):
            x[i] = var(parameter, x)
    return x
"""

class ChainGibbsSampler:
    def __init__(self, x_init, variables, burn_in=10):
        self.x = x_init
        self.vars = variables
        # self.sample()

    def sample(self, parameter, n_steps=10):
        return self(parameter, n_steps)

    def __call__(self, parameter, n_steps=10):
        k = 0
        while True:
            for i, var in enumerate(self.vars):  # chain
                r""" 
                Set x_i = argmax_{x_i \in values(X_i)} P_\theta (x_i | x_{-i})
                I.e. set the new x_i to the most probable state, given all the other variables.
                 """
                self.x[i] = var.sample_posterior(parameter, self.x)
                # assert (self.x[i] == 0 and var.ID != 0) or self.x[i] == var.polarity or (var.ID == 0 and self.x[i] != 0)
                k += 1
                if k == n_steps:
                    return self.x.detach().clone()
