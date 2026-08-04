import emcee
import h5py
import numpy as np
import pandas as pd


def read_chain(chain_fname):
    """
    Reads an old style text chain, and emcee HDF5 chain or a new style pocoMC HDF5 chain

    Returns pandas DataFrame and list of variable names
    """
    try:
        df = pd.read_csv(chain_fname, sep=r"\s+")
        colKeys = list(df.columns.values)[1:-1]
    except UnicodeDecodeError:
        with h5py.File(chain_fname, "r") as f:
            pocoMC = "weights" in f

        if pocoMC:
            # pocoMC chain
            colKeys, df = read_pocoMC_chain(chain_fname)
        else:
            # emcee chain
            colKeys, df = read_emcee_chain(chain_fname)

    return colKeys, df


def read_pocoMC_chain(chain_fname):
    with h5py.File(chain_fname, "r") as f:
        colKeys = [n.decode() for n in f["var_names"]]
        samples = np.array(f["samples"])
        weights = np.array(f["weights"])
        ln_prob = np.array(f["logp"])

    # weight samples
    idx_resampled = np.random.choice(
        np.arange(len(weights)), size=len(samples), replace=True, p=weights
    )
    samples = samples[idx_resampled]
    ln_prob = ln_prob[idx_resampled]
    df = pd.DataFrame(samples, columns=colKeys)
    df["ln_prob"] = ln_prob
    return colKeys, df


def write_pocoMC_chain(chain_fname, model, sampler):
    samples, weights, logl, logp = sampler.posterior()
    # list of strings for parameter names
    names = model.dynasty_par_names
    with h5py.File(chain_fname, "w") as f:
        f.create_dataset("samples", data=samples)
        f.create_dataset("weights", data=weights)
        f.create_dataset("logp", data=logp)
        f.create_dataset("logl", data=logl)
        f.create_dataset("var_names", data=np.array(names, dtype="S"))


def read_emcee_chain(chain_fname):
    reader = emcee.backends.HDFBackend(chain_fname, read_only=True)
    samples = reader.get_chain(discard=0, flat=True, thin=1)
    nwalkers, npars = reader.shape
    nsamples = samples.size // npars // nwalkers
    with h5py.File(chain_fname, "r") as f:
        colKeys = list(f["mcmc"].attrs["var_names"])
        df = pd.DataFrame(samples, columns=colKeys)
        df["ln_prob"] = reader.get_log_prob(discard=0, flat=True, thin=1)
        nsamples = samples.size // npars // nwalkers
        df["walker_no"] = np.array(list(range(nwalkers)) * nsamples)
    return colKeys, df
