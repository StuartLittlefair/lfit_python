import numpy as np
import scipy.stats as stats
import pandas as pd
import emcee
import dask.dataframe as dd
import seaborn
from os import stat
try:
    import triangle
    # This triangle should have a method corner
    # There are two python packages with conflicting names
    getattr(triangle, "corner")
except (AttributeError, ImportError):
    # We want the other package
    import corner as triangle

# lightweight progress bar
from tqdm import tqdm
import scipy.integrate as intg
import warnings
from matplotlib import pyplot as plt

TINY = -np.inf


class Prior(object):
    '''a class to represent a prior on a parameter, which makes calculating
    prior log-probability easier.

    Priors can be of five types: gauss, gaussPos, uniform, log_uniform and mod_jeff

    gauss is a Gaussian distribution, and is useful for parameters with
    existing constraints in the literature
    gaussPos is like gauss but enforces positivity
    Gaussian priors are initialised as Prior('gauss',mean,stdDev)

    uniform is a uniform prior, initialised like Prior('uniform',low_limit,high_limit)
    uniform priors are useful because they are 'uninformative'

    log_uniform priors have constant probability in log-space. They are the uninformative prior
    for 'scale-factors', such as error bars (look up Jeffreys prior for more info)

    mod_jeff is a modified jeffries prior - see Gregory et al 2007
    they are useful when you have a large uncertainty in the parameter value, so
    a jeffreys prior is appropriate, but the range of allowed values starts at 0

    they have two parameters, p0 and pmax.
    they act as a jeffrey's prior about p0, and uniform below p0. typically
    set p0=noise level
    '''
    def __init__(self, type, p1, p2):
        assert type in ['gauss', 'gaussPos', 'uniform', 'log_uniform', 'mod_jeff']
        self.type = type
        self.p1 = p1
        self.p2 = p2
        if type == 'log_uniform' and self.p1 < 1.0e-30:
            warnings.warn('lower limit on log_uniform prior rescaled from %f to 1.0e-30' % self.p1)
            self.p1 = 1.0e-30
        if type == 'log_uniform':
            self.normalise = 1.0
            self.normalise = np.fabs(intg.quad(self.ln_prob, self.p1, self.p2)[0])
        if type == 'mod_jeff':
            self.normalise = np.log((self.p1+self.p2)/self.p1)

    def ln_prob(self, val):
        if self.type == 'gauss':
            prob = stats.norm(scale=self.p2, loc=self.p1).pdf(val)
            if prob > 0:
                return np.log(prob)
            else:
                return TINY
        elif self.type == 'gaussPos':
            if val <= 0.0:
                return TINY
            else:
                prob = stats.norm(scale=self.p2, loc=self.p1).pdf(val)
                if prob > 0:
                    return np.log(prob)
                else:
                    return TINY
        elif self.type == 'uniform':
            if (val > self.p1) and (val < self.p2):
                return np.log(1.0/np.abs(self.p1-self.p2))
            else:
                return TINY
        elif self.type == 'log_uniform':
            if (val > self.p1) and (val < self.p2):
                return np.log(1.0 / self.normalise / val)
            else:
                return TINY
        elif self.type == 'mod_jeff':
            if (val > 0) and (val < self.p2):
                return np.log(1.0 / self.normalise / (val+self.p1))
            else:
                return TINY


class Param(object):
    '''A Param needs a starting value, a current value, and a prior
    and a flag to state whether is should vary'''
    def __init__(self, name, startVal, prior, isVar=True):
        self.name = name
        self.startVal = startVal
        self.prior = prior
        self.currVal = startVal
        self.isVar = isVar

    @classmethod
    def fromString(cls, name, parString):
        fields = parString.split()
        val = float(fields[0])
        priorType = fields[1].strip()
        priorP1 = float(fields[2])
        priorP2 = float(fields[3])
        if len(fields) == 5:
            isVar = bool(int(fields[4]))
        else:
            isVar = True
        return cls(name, val, Prior(priorType, priorP1, priorP2), isVar)

    @property
    def isValid(self):
        return np.isfinite(self.prior.ln_prob(self.currVal))


def fracWithin(pdf, val):
    return pdf[pdf >= val].sum()


def thumbPlot(chain, labels, **kwargs):
    seaborn.set(style='ticks')
    seaborn.set_style({"xtick.direction": "in","ytick.direction": "in"})
    fig = triangle.corner(chain, labels=labels, bins=50,
                          label_kwargs=dict(fontsize=18), **kwargs)
    return fig


def scatterWalkers(pos0, percentScatter):
    warnings.warn('scatterWalkers decprecated: use emcee.utils.sample_ball instead')
    nwalkers = pos0.shape[0]
    npars = pos0.shape[1]
    scatter = np.array([np.random.normal(size=npars) for i in range(nwalkers)])
    return pos0 + percentScatter*pos0*scatter/100.0


def initialise_walkers(p, scatter, nwalkers, ln_prior):
    # Create starting ball of walkers with a certain amount of scatter
    p0 = emcee.utils.sample_ball(p, scatter*p, size=nwalkers)
    # Make initial number of invalid walkers equal to total number of walkers
    numInvalid = nwalkers
    print('Initialising walkers...')
    print('Number of walkers currently invalid:')
    # All invalid params need to be resampled
    while numInvalid > 0:
        # Create a mask of invalid params
        isValid = np.array([np.isfinite(ln_prior(p)) for p in p0])
        bad = p0[~isValid]
        # Determine the number of good and bad walkers
        nbad = len(bad)
        print(nbad)
        ngood = len(p0[isValid])
        # Choose nbad random rows from ngood walker sample
        replacement_rows = np.random.randint(ngood, size=nbad)
        # Create replacement values from valid walkers
        replacements = p0[isValid][replacement_rows]
        # Add scatter to replacement values
        replacements += 0.5*replacements*scatter*np.random.normal(size=replacements.shape)
        # Replace invalid walkers with new values
        p0[~isValid] = replacements
        numInvalid = len(p0[~isValid])
    return p0


def initialise_walkers_pt(p, scatter, nwalkers, ntemps, ln_prior):
    # Create starting ball of walkers with a certain amount of scatter
    p0 = np.array([emcee.utils.sample_ball(p, scatter*p, size=nwalkers) for
                   i in range(ntemps)])
    orig_shape = p0.shape
    # Re-shape p0 array
    p0 = p0.reshape(nwalkers*ntemps, len(p))
    # Make initial number of invalid walkers equal to total number of walkers
    numInvalid = nwalkers*ntemps
    print('Initialising walkers...')
    print('Number of walkers currently invalid:')
    # All invalid params need to be resampled
    while numInvalid > 0:
        # Create a mask of invalid params
        isValid = np.array([np.isfinite(ln_prior(p)) for p in p0])
        bad = p0[~isValid]
        # Determine the number of good and bad walkers
        nbad = len(bad)
        print(nbad)
        ngood = len(p0[isValid])
        # Choose nbad random rows from ngood walker sample
        replacement_rows = np.random.randint(ngood, size=nbad)
        # Create replacement values from valid walkers
        replacements = p0[isValid][replacement_rows]
        # Add scatter to replacement values
        replacements += 0.5*replacements*scatter*np.random.normal(size=replacements.shape)
        # Replace invalid walkers with new values
        p0[~isValid] = replacements
        numInvalid = len(p0[~isValid])
    p0 = p0.reshape(orig_shape)
    return p0


def run_burnin(sampler, startPos, nSteps, storechain=False, progress=True):
    iStep = 0
    if progress:
        bar = tqdm(total=nSteps)
    for pos, prob, state in sampler.sample(startPos, iterations=nSteps, storechain=storechain):
        iStep += 1
        if progress:
            bar.update()
    if progress:
        bar.close()
    return pos, prob, state


def run_mcmc_save(sampler, startPos, nSteps, rState, file, progress=True, **kwargs):
    '''runs an MCMC chain with emcee, and saves steps to a file'''
    # open chain save file
    if file:
        f = open(file, "w")
        f.close()
    iStep = 0
    if progress:
        bar = tqdm(total=nSteps)
    for pos, prob, state in sampler.sample(startPos, iterations=nSteps, rstate0=rState,
                                           storechain=True, **kwargs):
        if file:
            f = open(file, "a")
        iStep += 1
        if progress:
            bar.update()
        for k in range(pos.shape[0]):
            # loop over all walkers and append to file
            thisPos = pos[k]
            thisProb = prob[k]
            if file:
                f.write("{0:4d} {1:s} {2:f}\n".format(k, " ".join(map(str, thisPos)), thisProb))
        if file:
            f.close()
    if progress:
        bar.close()
    return sampler


def run_ptmcmc_save(sampler, startPos, nSteps, file, progress=True, **kwargs):
    '''runs PT MCMC and saves zero temperature chain to a file'''
    if file:
        f = open(file, "w")
        f.close()
    iStep = 0
    if progress:
        bar = tqdm(total=nSteps)
    for pos, prob, like in sampler.sample(startPos, iterations=nSteps, storechain=True, **kwargs):
        f = open(file, "a")
        iStep += 1
        if progress:
            bar.update()
        # pos is shape (ntemps, nwalkers, npars)
        # prob is shape (ntemps, nwalkers)
        # loop over all walkers for first temp and append to file
        zpos = pos[0, ...]
        zprob = prob[0, ...]
        for k in range(zpos.shape[0]):
            thisPos = zpos[k]
            thisProb = zprob[k]
            f.write("{0:4d} {1:s} {2:f}\n".format(k, " ".join(map(str, thisPos)), thisProb))
        f.close()
    if progress:
        bar.close()
    return sampler


def flatchain(chain, npars, nskip=0, thin=1):
    '''flattens a chain (i.e collects results from all walkers),
    with options to skip the first nskip parameters, and thin the chain
    by only retrieving a point every thin steps - thinning can be useful when
    the steps of the chain are highly correlated'''
    return chain[:, nskip::thin, :].reshape((-1, npars))

def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order
    Taken from 
    https://stackoverflow.com/questions/2301789/read-a-file-in-reverse-order-using-python
    """
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # the first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # if the previous chunk starts right from the beginning of line
                # do not concact the segment to the last line of new chunk
                # instead, yield the segment first 
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment

def readchain(file, nskip=0, thin=1., memory=10):
    '''Memory is the amount of memory available to read the file into, in Gigabytes'''
    # Get filesize in Gb
    filesize = stat(file).st_size
    # The max file size we can read is set by the amount of memory we have. 
    if filesize > memory*1e9:
        ### TODO: Read in only the tail of the chainfile. 
        # Temporary solution, read in a sample of the chain.
        print("Warning! The supplied chain file ({:.1f}Gb) is larger than {:.1f}Gb.".format(filesize/1e9, memory))
        # Count the lines
        with open(file, 'r') as f:
            linesize = 0
            nsample = 50
            for i in range(nsample):
                line = f.readline()
                linesize += len(line.encode('utf-8'))
            linesize /= nsample
        nlines = int(filesize / linesize)

        print("A line is ~{} bytes, and the file contains {} lines".format(linesize, nlines))

        # only read every nth line
        n = 2*filesize / (memory*1e9)
        n = np.ceil(n)
        # Skips is a list of indeces to NOT read
        skips = [x for x in range(nlines) if x%n != 0]

        print("Sampling it by reading only every {}th line in the chain, to reduce to a {:.1f}Gb file...".format(n, filesize/(1e9*n)))
        data = pd.read_csv(file, header=None, compression=None, delim_whitespace=True,
                            skiprows=skips)
    else:
        # read in whole file
        data = pd.read_csv(file, header=None, compression=None, delim_whitespace=True)
    data = np.array(data)
    nwalkers = int(data[:, 0].max()+1)
    nprod = int(data.shape[0]/nwalkers)
    npars = data.shape[1] - 1  # first is walker ID, last is ln_prob
    chain = np.reshape(data[:, 1:], (nwalkers, nprod, npars))
    return chain


def readchain_dask(file, nskip=0, thin=1):
    data = dd.io.read_csv(file, engine='c', header=None, compression=None,
                          na_filter=False, delim_whitespace=True)
    data = data.compute()
    data = np.array(data)
    nwalkers = int(data[:, 0].max()+1)
    nprod = int(data.shape[0]/nwalkers)
    npars = data.shape[1] - 1  # first is walker ID, last is ln_prob
    chain = np.reshape(data[:, 1:], (nwalkers, nprod, npars))
    return chain


def readflatchain(file):
    data = pd.read_csv(file, header=None, compression=None, delim_whitespace=True)
    data = np.array(data)
    return data


def plotchains(chain, npar, alpha=0.2):
    nwalkers, nsteps, npars = chain.shape
    fig = plt.figure()
    for i in range(nwalkers):
        plt.plot(chain[i, :, npar], alpha=alpha, color='k')
    return fig


def GR_diagnostic(sampler_chain):
    '''Gelman & Rubin check for convergence.'''
    m, n, ndim = np.shape(sampler_chain)
    R_hats = np.zeros((ndim))
    samples = sampler_chain[:, :, :].reshape(-1, ndim)
    for i in range(ndim):  # iterate over parameters

        # Define variables
        chains = sampler_chain[:, :, i]

        flat_chain = samples[:, i]
        psi_dot_dot = np.mean(flat_chain)
        psi_j_dot = np.mean(chains, axis=1)
        psi_j_t = chains

        # Calculate between-chain variance
        between = sum((psi_j_dot - psi_dot_dot)**2) / (m - 1)

        # Calculate within-chain variance
        inner_sum = np.sum(np.array([(psi_j_t[j, :] - psi_j_dot[j])**2 for j in range(m)]), axis=1)
        outer_sum = np.sum(inner_sum)
        W = outer_sum / (m*(n-1))

        # Calculate sigma
        sigma2 = (n-1)/n * W + between

        # Calculate convergence criterion (potential scale reduction factor)
        R_hats[i] = (m + 1)*sigma2/(m*W) - (n-1)/(m*n)
    return R_hats


def ln_marginal_likelihood(params, lnp):
    '''given a flattened chain which consists of a series
    of samples from the parameter posterior distributions,
    and another array which is ln_prob (posterior) for these
    parameters, estimate the marginal likelihood of this model,
    allowing for model selection.

    Such a chain is created by reading in the output file of
    an MCMC run, and running flatchain on it.

    Uses the method of Chib & Jeliazkov (2001) as outlined
    by Haywood et al 2014

    '''
    raise Exception("""This routine is incorrect and should not be used until fixed.
    See the emcee docs for the Parallel Tempering sampler instead""")
    # maximum likelihood estimate
    loc_best = lnp.argmin()
    log_max_likelihood = lnp[loc_best]
    best = params[loc_best]
    # standard deviations
    sigmas = params.std(axis=0)

    # now for the magic
    # at each step in the chain, add up 0.5*((val-best)/sigma)**2 for all params
    term = 0.5*((params-best)/sigmas)**2
    term = term.sum(axis=1)

    # top term in posterior_ordinate
    numerator = np.sum(np.exp(term))
    denominator = np.sum(lnp/log_max_likelihood)
    posterior_ordinate = numerator/denominator

    log_marginal_likelihood = log_max_likelihood - np.log(posterior_ordinate)
    return log_marginal_likelihood


def rebin(xbins, x, y, e=None, weighted=True, errors_from_rms=False):
    digitized = np.digitize(x, xbins)
    xbin = []
    ybin = []
    ebin = []
    for i in range(0, len(xbins)):
            bin_y_vals = y[digitized == i]
            bin_x_vals = x[digitized == i]
            if e is not None:
                bin_e_vals = e[digitized == i]
            if weighted:
                if e is None:
                    raise Exception('Cannot compute weighted mean without errors')
                weights = 1.0/bin_e_vals**2
                xbin.append(np.sum(weights*bin_x_vals) / np.sum(weights))
                ybin.append(np.sum(weights*bin_y_vals) / np.sum(weights))
                if errors_from_rms:
                    ebin.append(np.std(bin_y_vals))
                else:
                    ebin.append(np.sqrt(1.0/np.sum(weights)))
            else:
                xbin.append(bin_x_vals.mean())
                ybin.append(bin_y_vals.mean())
                if errors_from_rms:
                    ebin.append(np.std(bin_y_vals))
                else:
                    ebin.append(np.sqrt(np.sum(bin_e_vals**2)) / len(bin_e_vals))
    xbin = np.array(xbin)
    ybin = np.array(ybin)
    ebin = np.array(ebin)
    return (xbin, ybin, ebin)
