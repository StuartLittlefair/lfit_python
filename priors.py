"""Custom prior for the pocoMC library"""

import numpy as np

from model import Node


class CustomPrior:
    def __init__(self, model: Node):
        self.model = model
        self.dim = len(model.priors)

    def _propose_sample(self):
        """Propose a single sample from the prior distribution of the individual parameters"""

        sample = np.array([dist.rvs() for dist in self.model.priors])
        return sample

    def rvs(self, size=1):
        """Draw samples from the prior distribution of the individual parameters.

        This must respect not only the individual parameter priors, but also any constraints between
        parameters. For example, we cannot have brightspots that miss the disc, or a phase width
        that is too large for the mass ratio.
        """

        # draw samples with replacement until valid
        samples = []
        for i in range(size):
            ok = False
            sample = self._propose_sample()
            while not ok:
                self.model.dynasty_par_vals = sample
                ok = self.model.check_validity()
                sample = self._propose_sample()
            samples.append(sample)
        return np.array(samples)

    def logpdf(self, x):
        x = np.atleast_2d(x)
        logp = np.zeros(len(x))
        for i, dist in enumerate(self.model.priors):
            logp += dist.logpdf(x[:, i])
        return logp

    @property
    def bounds(self):
        bounds = []
        for prior in self.model.priors:
            bounds.append(prior.support())
        return np.array(bounds)
