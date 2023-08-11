import multiprocessing as mp
import os
from enum import Enum
import copy

import configobj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import emcee
import scipy.interpolate as interp
from astropy.stats import sigma_clipped_stats

from utils import read_chain
from mcmc_utils import (
    flatchain,
    initialise_walkers,
    run_burnin,
    run_mcmc_save,
    thumbPlot,
)
from model import Param

# import warnings


# Location of data tables
ROOT, _ = os.path.split(__file__)


def ln_prior(vect, model):
    # first we update the model to use the pars suggested by the MCMC chain
    for i in range(model.npars):
        model[i] = vect[i]

    lnp = 0.0

    # teff, (usually uniform between allowed range - 6 to 90,000)
    param = model.teff
    lnp += param.prior.ln_prob(param.currVal)

    # logg, uniform between allowed range (7.01 to 8.99), or Gaussian from constraints
    param = model.logg
    lnp += param.prior.ln_prob(param.currVal)

    # Parallax, gaussian prior of the gaia value.
    param = model.plax
    lnp += param.prior.ln_prob(param.currVal)

    # enforce +ve parallax
    if model.plax.currVal <= 0:
        return -np.inf

    # reddening, cannot exceed galactic value (should estimate from line of sight)
    # https://irsa.ipac.caltech.edu/applications/DUST/
    param = model.ebv
    lnp += param.prior.ln_prob(param.currVal)
    return lnp


def ln_likelihood(vect, model):
    # first we update the model to use the pars suggested by the MCMC chain
    for i in range(model.npars):
        model[i] = vect[i]
    return -0.5 * model.chisq()


def ln_prob(vect, model):
    lnp = ln_prior(vect, model)
    if np.isfinite(lnp):
        lnp += ln_likelihood(vect, model)
    if np.isnan(lnp):
        lnp = -np.inf
    return lnp


def sdss2kg5(g, r):
    KG5 = g - 0.2240 * (g - r) ** 2 - 0.3590 * (g - r) + 0.0460
    return KG5


sdss2kg5_vect = np.vectorize(sdss2kg5)


def sdssmag2flux(mag):
    return 3631e3 * np.power(10, -0.4 * mag)


class PhotometricSystem(Enum):
    SDSS = "SDSS"
    HCAM = "HCAM"
    UCAM_SDSS = "UCAM_SDSS"
    UCAM_SUPER = "UCAM_SUPER"
    USPEC = "USPEC"

    @property
    def bergeron_table(self):
        """
        Name of correct table for Bergeron mags
        """
        if self.name == "SDSS":
            return os.path.join(ROOT, "Bergeron/Table_DA_sdss")
        else:
            return os.path.join(ROOT, "Bergeron/Table_DA")

    def color_correction_data(self, band):
        """
        Information needed for correction from Bergeron table to this band

        Parameters
        ----------
        band: str
            e.g u, g, r, i, z

        Returns
        -------
        table: str
            Path to color correction tables
        column: str
            Name of column to read from table for correction
        """
        if self.name == "USPEC":
            table = os.path.join(
                ROOT,
                "Bergeron/color_correction_tables/color_corrections_HCAM-GTC-super_minus_tnt_uspec.csv",
            )
            column = band
        elif self.name == "SDSS" or self.name == "HCAM":
            table = None
            column = None
        else:
            table = os.path.join(
                ROOT,
                "Bergeron/color_correction_tables/color_corrections_HCAM-GTC-super_minus_ntt_ucam.csv",
            )
            column = f"{band}_s" if "SUPER" in self.name else band
        return table, column

    def central_wavelength(self, band):
        super_lambda_c = {
            "u": 352.6,
            "g": 473.2,
            "r": 619.9,
            "i": 771.1,
            "z": 915.6,
        }
        lambda_c = {
            "u": 355.7,
            "g": 482.5,
            "r": 626.1,
            "i": 767.2,
            "z": 909.7,
            "kg5": 507.5,
        }
        lambda_c_dict = (
            super_lambda_c if self.name in ["HCAM", "UCAM_SUPER"] else lambda_c
        )
        return lambda_c_dict[band]


class Flux(object):
    BANDS = ["u", "g", "r", "i", "z"]
    # Multiply E(B-V) by these numbers to get extinction in each band
    # from Schlafly & Finkbeiner (2011) @ R_V = 3.1
    EXTINCTION_COEFFS = {
        "u": 4.239,
        "g": 3.303,
        "r": 2.285,
        "i": 1.698,
        "z": 1.263,
        "kg5": 2.751,
    }

    def __init__(self, val, err, photometric_system, band, syserr=0.03):
        """
        Representation of an observed WD flux

        Parameters
        ----------
        val, err: float
            Observed value and error
        photometric_system: PhotometricSystem
            The system this flux is observed in e.g SDSS or HCAM
        band: str
            Name of filter, e.g 'u', 'g' etc.

            Do not use values like 'g_s'  here, that is taken
            care of in the photometric system
        syserr: float
            Additional systematic error added to account for calibration
            issues.
        """
        self.flux = val
        self.err = np.sqrt(err**2 + (val * syserr) ** 2)

        # This is the actual band observed with.
        self.photometric_system = photometric_system
        self.band = band
        self.mag = 2.5 * np.log10(3631e3 / self.flux)
        self.magerr = 2.5 * 0.434 * (self.err / self.flux)

        # Create an interpolater for the color corrections
        correction_table_name, column = photometric_system.color_correction_data(
            self.band
        )
        if correction_table_name is not None:
            correction_table = pd.read_csv(correction_table_name)
            self.correction_func = interp.LinearNDInterpolator(
                correction_table[["Teff", "logg"]], correction_table[column]
            )
        else:
            self.correction_func = None

        # Create an interpolator for the Bergeron table
        DA = pd.read_csv(
            photometric_system.bergeron_table,
            delim_whitespace=True,
            skiprows=0,
            header=1,
        )
        self.bergeron_func = interp.LinearNDInterpolator(
            DA[["Teff", "log_g"]], DA[band]
        )

    def __repr__(self):
        return "Flux(val={:.3f}, err={:.3f}, photometric_system={}, band={})".format(
            self.flux, self.err, self.photometric_system, self.band
        )

    @property
    def extinction_coefficient(self):
        return self.EXTINCTION_COEFFS[self.band]

    @property
    def central_wavelength(self):
        return self.photometric_system.central_wavelength(self.band)


class WDModel:
    """
    Model for calculating WD Fluxes

    Can be passed to MCMC routines for calculating model and chisq, and prior prob

    This class also behaves like a list, of the current values of all parameters
    this enables it to be seamlessly used with emcee

    Note that parallax should be provided in MILLIarcseconds.

    Parameters
    -----------
    teff, logg, plax, ebv: `mcmc_utils.Param`
        Fittable parameters.
    fluxes: list(Flux)
        A list of observed fluxes.
    """

    # arguments are Param objects (see mcmc_utils)
    def __init__(self, teff, logg, plax, ebv, fluxes):
        self.teff = teff
        self.logg = logg
        self.plax = plax
        self.ebv = ebv

        # initialise list bit of object with parameters
        self.variables = [self.teff, self.logg, self.plax, self.ebv]

        # Observed data
        self.obs_fluxes = copy.copy(fluxes)

    # these routines are needed so object will behave like a list
    def __getitem__(self, ind):
        return self.variables[ind].currVal

    def __setitem__(self, ind, val):
        self.variables[ind].currVal = val

    def __delitem__(self, ind):
        self.variables.remove(ind)

    def __len__(self):
        return len(self.variables)

    def insert(self, ind, val):
        self.variables.insert(ind, val)

    @property
    def npars(self):
        return len(self.variables)

    @property
    def dist(self):
        if self.plax.currVal <= 0.0:
            return np.inf
        else:
            return 1000.0 / self.plax.currVal

    def __repr__(self):
        return "WDModel(teff={:.1f}, logg={:.2f}, plax={:.3f}, ebv={:.1f})".format(
            self.teff.currVal,
            self.logg.currVal,
            self.plax.currVal,
            self.ebv.currVal,
        )

    @property
    def apparent_mags(self):
        """
        Calculate apparent magnitudes for each of my observed fluxes
        """
        mags = []
        t, g = self.teff.currVal, self.logg.currVal
        # Distance modulus
        dmod = -5.0 * np.log10(self.plax.currVal / 100)
        for flux in self.obs_fluxes:
            abs_mag = flux.bergeron_func(t, g)
            # correction from magnitude in bergeron table to observed system
            if flux.correction_func is None:
                correction = 0
            else:
                correction = flux.correction_func(t, g)
            # correction is Bergeron System - Observed System
            # correct to OBSERVED system
            abs_mag -= correction
            # apply distance modulus
            mag = abs_mag + dmod

            # apply exinction
            extinction = self.ebv.currVal * flux.extinction_coefficient
            mag += extinction
            mags.append(mag)

        return np.array(mags)

    def chisq(self):
        """Calculate Chisq"""

        mags = self.apparent_mags
        predicted_fluxes = sdssmag2flux(mags)
        observed_fluxes = np.array([f.flux for f in self.obs_fluxes])
        errors = np.array([f.err for f in self.obs_fluxes])
        # Chi-squared
        chisq = np.power(((predicted_fluxes - observed_fluxes) / errors), 2)
        chisq = np.sum(chisq)

        return chisq


def plotColors(model, fname="colorplot.pdf"):
    print("\n\n-----------------------------------------------")
    print("Creating color plots...")
    _, ax = plt.subplots(figsize=(6, 6))

    # OBSERVED DATA
    flux_u = [obs for obs in model.obs_fluxes if "u" in obs.band][0]
    flux_g = [obs for obs in model.obs_fluxes if "g" in obs.band][0]
    flux_r = [obs for obs in model.obs_fluxes if "r" in obs.band][0]
    print("Observations:\n    {}\n    {}\n    {}".format(flux_u, flux_g, flux_r))

    obs_ug_err = np.sqrt((flux_u.magerr**2) + (flux_g.magerr**2))
    obs_gr_err = np.sqrt((flux_g.magerr**2) + (flux_r.magerr**2))

    # Correct magnitudes to the Bergeron frame
    t, g = model.teff.currVal, model.logg.currVal
    u_mag = flux_u.bergeron_mag(t, g)
    g_mag = flux_g.bergeron_mag(t, g)
    r_mag = flux_r.bergeron_mag(t, g)

    if model.DEBUG:
        print("Observation, uncorrected for IS extinction:")
        print(
            "   Magnitudes:\n     u: {}\n     g: {}\n     r: {}".format(
                u_mag, g_mag, r_mag
            )
        )

    # subtract interstellar extinction
    ex = model.ebv
    u_mag -= model.extinction_coefficients["u_s"] * ex.currVal
    g_mag -= model.extinction_coefficients["g_s"] * ex.currVal
    r_mag -= model.extinction_coefficients["r_s"] * ex.currVal

    print("After correcting (if necessary), and removing IS extinction:")
    print(
        "   Magnitudes:\n     u: {}\n     g: {}\n     r: {}".format(u_mag, g_mag, r_mag)
    )

    ug_mag = u_mag - g_mag
    gr_mag = g_mag - r_mag

    print(
        "Observed Colors in the HCAM/GTC/super lightpath (corrected for IS extinction):"
    )
    print("u-g = {:> 5.3f}+/-{:< 5.3f}".format(ug_mag, obs_ug_err))
    print("g-r = {:> 5.3f}+/-{:< 5.3f}".format(gr_mag, obs_gr_err))

    # Generate the model's apparent magnitudes (no atmosphere, no IS extinction), and plot that color too
    # Get absolute magnitudes
    abs_mags = model.gen_absolute_mags()
    # Apply distance modulus
    dmod = 5.0 * np.log10(model.dist / 10.0)
    modelled_mags = abs_mags + dmod

    # Calculate the colours
    bands = [obs.orig_band for obs in model.obs_fluxes]
    u_index = bands.index(flux_u.orig_band)
    g_index = bands.index(flux_g.orig_band)
    r_index = bands.index(flux_r.orig_band)
    if model.DEBUG:
        print(
            "Bergeron model interpolations for T: {:.0f}, log(g): {:.3f}...".format(
                model.teff.currVal, model.logg.currVal
            )
        )
        print("Observed bands: {}".format(bands))
        print("Modelled mags: {}".format(modelled_mags))
        print("Indexes|| u: {} || g: {} || r: {}\n".format(u_index, g_index, r_index))

    model_ug = modelled_mags[u_index] - modelled_mags[g_index]
    model_gr = modelled_mags[g_index] - modelled_mags[r_index]

    # bergeron model magnitudes, will be plotted as tracks
    bergeron_umags = np.array(model.DA["u_s"])
    bergeron_gmags = np.array(model.DA["g_s"])
    bergeron_rmags = np.array(model.DA["r_s"])

    # calculate colours
    ug = bergeron_umags - bergeron_gmags
    gr = bergeron_gmags - bergeron_rmags

    # make grid of teff, logg from the bergeron table
    teffs = np.unique(model.DA["Teff"])
    loggs = np.unique(model.DA["log_g"])
    nteff = len(teffs)
    nlogg = len(loggs)
    # reshape colours onto 2D grid of (logg, teff)
    ug = ug.reshape((nlogg, nteff))
    gr = gr.reshape((nlogg, nteff))

    # Plotting
    # Bergeron cooling tracks and isogravity contours
    for a in range(nlogg):
        ax.plot(ug[a, :], gr[a, :], "k-")
    for a in range(0, nteff, 4):
        ax.plot(ug[:, a], gr[:, a], "r--")

    # Observed color
    ax.errorbar(
        x=ug_mag,
        y=gr_mag,
        xerr=obs_ug_err,
        yerr=obs_gr_err,
        fmt="o",
        ls="none",
        color="darkred",
        capsize=3,
        label="Observed",
    )

    # Modelled color
    ax.errorbar(
        x=model_ug,
        y=model_gr,
        fmt="o",
        ls="none",
        color="blue",
        capsize=3,
        label="Modelled - T: {:.0f} | logg: {:.2f}".format(t, g),
    )

    # annotate for teff
    xa = ug[0, 4] + 0.03
    ya = gr[0, 4]
    val = teffs[4]
    t = ax.annotate(
        "T = %d K" % val,
        xy=(xa, ya),
        color="r",
        horizontalalignment="left",
        verticalalignment="center",
        size="small",
    )
    t.set_rotation(0.0)

    xa = ug[0, 8] + 0.03
    ya = gr[0, 8]
    val = teffs[8]
    t = ax.annotate(
        "T = %d K" % val,
        xy=(xa, ya),
        color="r",
        horizontalalignment="left",
        verticalalignment="center",
        size="small",
    )
    t.set_rotation(0.0)

    xa = ug[0, 20] + 0.01
    ya = gr[0, 20] - 0.01
    val = teffs[20]
    t = ax.annotate(
        "T = %d K" % val,
        xy=(xa, ya),
        color="r",
        horizontalalignment="left",
        verticalalignment="top",
        size="small",
    )
    t.set_rotation(0.0)

    xa = ug[0, 24] + 0.01
    ya = gr[0, 24] - 0.01
    val = teffs[24]
    t = ax.annotate(
        "T = %d K" % val,
        xy=(xa, ya),
        color="r",
        horizontalalignment="left",
        verticalalignment="top",
        size="small",
    )
    t.set_rotation(0.0)

    ax.set_xlabel("{}-{}".format(flux_u.orig_band, flux_g.orig_band))
    ax.set_ylabel("{}-{}".format(flux_g.orig_band, flux_r.orig_band))
    ax.set_xlim([-0.5, 1])
    ax.set_ylim([-0.5, 0.5])
    ax.legend()

    plt.savefig(fname)
    plt.show()

    print("Done!")
    print("-----------------------------------------------\n")


def plotFluxes(model, fname="fluxplot.pdf"):
    """
    Plot observed fluxes vs model fluxes
    """
    model_mags = model.apparent_mags
    model_flx = sdssmag2flux(model_mags)
    # Central wavelengths for the bands
    lambdas = np.array([obs.central_wavelength for obs in model.obs_fluxes])
    obs_flx = [obs.flux for obs in model.obs_fluxes]
    obs_flx_err = [obs.err for obs in model.obs_fluxes]

    # Do the actual plotting
    _, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(
        lambdas,
        model_flx,
        xerr=None,
        yerr=None,
        fmt="P",
        ls="none",
        color="darkred",
        label="Modelled apparent flux",
        markersize=6,
        linewidth=1,
        capsize=None,
    )
    ax.errorbar(
        lambdas,
        obs_flx,
        xerr=None,
        yerr=obs_flx_err,
        fmt="o",
        ls="none",
        color="blue",
        label="Observed flux",
        markersize=6,
        linewidth=1,
        capsize=None,
    )
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Flux, mJy")
    ax.legend()

    plt.tight_layout()
    plt.savefig(fname)
    plt.close("all")


if __name__ == "__main__":
    LOGFILE = open("WDPARAMS.LOGS", "w")

    # Allows input file to be passed to code from argument line
    import argparse

    parser = argparse.ArgumentParser(description="Fit WD Fluxes")
    parser.add_argument("file", action="store", help="input file")
    args = parser.parse_args()

    # Use parseInput function to read data from input file
    input_dict = configobj.ConfigObj(args.file)

    # Read information about mcmc, priors, neclipses, sys err
    nburn = int(input_dict["nburn"])
    nprod = int(input_dict["nprod"])
    nthread = int(input_dict["nthread"])
    nwalkers = int(input_dict["nwalkers"])
    scatter = float(input_dict["scatter"])
    thin = int(input_dict["thin"])
    toFit = int(input_dict["fit"])

    # Grab the variables
    teff = Param.fromString("teff", input_dict["teff"])
    logg = Param.fromString("logg", input_dict["logg"])
    plax = Param.fromString("plax", input_dict["plax"])
    ebv = Param.fromString("ebv", input_dict["ebv"])
    syserr = float(input_dict["syserr"])
    chain_file = input_dict.get("chain", None)

    # Logging
    LOGFILE.write("Fitting White Dwarf fluxes to model cooling tracks...\n")
    LOGFILE.write("Running fit from the following input file:\n")
    LOGFILE.write("#################################\n\n")
    LOGFILE.write(open(args.file, "r").read())
    LOGFILE.write("#################################\n\n")
    LOGFILE.write("Setting up fluxes...\n\n")

    # # # # # # # # # # # #
    # Load in chain file  #
    # # # # # # # # # # # #
    if chain_file is None:
        colKeys = []
        fluxes = []
    else:
        print("Reading in the chain file,", chain_file)
        colKeys, chain = read_chain(chain_file)
        print("Done!")

        # Extract the fluxes from the chain file, and create a list of Fux objects from that
        chain_bands = [
            key.lower().replace("wdflux_", "")
            for key in colKeys
            if "wdflux" in key.lower()
        ]
        print("I found the following bands in the chain file:")
        systems = input_dict["photometric_systems"]
        for band in chain_bands:
            print("\t{} ({})".format(band, systems.get(band, None)))
        print("\n\n\n")

        fluxes = []
        for band in chain_bands:
            # TODO: Add KG5 fluxes.
            if band == "kg5":
                LOGFILE.write("KG5 BANDS ARE CURRENTLY UNUSED. SKIPPING")
                print("KG5 BANDS ARE CURRENTLY UNUSED. SKIPPING")
                continue
            else:
                index = colKeys.index(f"wdFlux_{band}")
                if band not in systems:
                    msg = f"No photometric system for {band}, skipping"
                    LOGFILE.write(msg + "\n")
                    print(msg)
                    continue

                system = PhotometricSystem(systems[band])
                mean, _, std = sigma_clipped_stats(chain[:, index])
                flx = Flux(mean, std, system, band, syserr=syserr)
                print(f"{band} = {flx}")
                LOGFILE.write(f"{band} = {flx}")
                fluxes.append(flx)

    while True:
        print("Would you like to add another flux?")
        cont = input("y/n: ")
        if cont.lower() == "y":
            print("Enter a band:")
            band = input("> ")
            print("Enter a photometric system:")
            system = input("> ")
            print("Enter a Flux, in mJy")
            flx = input("> ")
            print("Enter an error on flux, mJy")
            fle = input("> ")

            flx = float(flx)
            fle = float(fle)
            system = PhotometricSystem(system)

            flux = Flux(flx, fle, system, band, syserr=syserr)
            fluxes.append(flux)
        else:
            print("Done!")
            break

    # Create the model object
    myModel = WDModel(teff, logg, plax, ebv, fluxes)
    npars = myModel.npars

    if toFit:
        guessP = np.array(myModel)
        nameList = ["Teff", "log_g", "Parallax", "E(B-V)"]

        # mp.set_start_method("spawn")
        pool = mp.Pool(nthread)
        p0 = initialise_walkers(guessP, scatter, nwalkers, ln_prior, myModel)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            npars,
            ln_prob,
            args=(myModel,),
            pool=pool,
        )

        # burnIn
        pos, prob, state = run_burnin(sampler, p0, nburn)

        # production
        sampler.reset()
        col_names = "walker_no " + " ".join(nameList) + " ln_prob"
        sampler = run_mcmc_save(
            sampler, pos, nprod, state, "chain_wd.txt", col_names=col_names
        )
        # Collect results from all walkers
        fchain = flatchain(sampler.chain, npars, thin=thin)

        # Plot the likelihoods
        likes = sampler.chain[..., -1]

        # Plot the mean likelihood evolution
        like_mu = np.mean(likes, axis=0)
        like_std = np.std(likes, axis=0)
        steps = np.arange(likes.shape[1])
        std = np.std(likes)

        # Make the likelihood plot
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.fill_between(
            steps, like_mu - like_std, like_mu + like_std, color="red", alpha=0.4
        )
        ax.plot(steps, like_mu, color="green")

        ax.set_xlabel("Step")
        ax.set_ylabel("ln_prob")

        plt.tight_layout()
        plt.savefig("wdparams_likelihoods.pdf")
        plt.close("all")

        bestPars = []
        print(fchain.shape)
        for i in range(npars):
            par = fchain[:, i]
            lolim, best, uplim = np.percentile(par, [16, 50, 84])
            myModel[i] = best

            print("%s = %f +%f -%f" % (nameList[i], best, uplim - best, best - lolim))
            bestPars.append(best)
        print("Creating corner plots...")
        fig = thumbPlot(fchain, nameList)
        fig.savefig("wdparams_cornerPlot.pdf")
        plt.close("all")
    else:
        bestPars = [par for par in myModel]

    print("Done!")
    print("Chisq = {:.3f}".format(myModel.chisq()))
    # Plot measured and model colors and fluxes
    plotFluxes(myModel)
    print("Model: {}".format(myModel))
    LOGFILE.close()
