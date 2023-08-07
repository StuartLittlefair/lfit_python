import numpy as np
import pandas as pd
import emcee
import h5py


def read_chain(chain_fname):
    """
    Reads an old style text chain or a new style HDF5 chain

    Returns pandas DataFrame and list of variable names
    """
    try:
        df = pd.read_csv(chain_fname, delim_whitespace=True)
        colKeys = list(df.columns.values)[1:-1]
    except UnicodeDecodeError:
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
