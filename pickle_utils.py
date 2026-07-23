import numpy as np

from CVModel import construct_model


def ln_prior(param_vector, input_fname):
    model = construct_model(input_fname)
    model.dynasty_par_vals = param_vector
    return np.sum([prior.pdf(val) for prior, val in zip(model.priors, param_vector)])


def ln_prob(param_vector, input_fname):
    model = construct_model(input_fname)
    model.dynasty_par_vals = param_vector
    lnprior = np.sum([prior.pdf(val) for prior, val in zip(model.priors, param_vector)])
    ln_like = model.ln_like()
    val = lnprior + ln_like

    return val


def ln_like(param_vector, input_fname):
    model = construct_model(input_fname)
    model.dynasty_par_vals = param_vector
    val = model.ln_like()

    return val
