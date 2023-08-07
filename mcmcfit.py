"""
This script will run the actual fitting procedure.
Requires the input file, and data files defined in that.
Supplied at the command line, via:

    python3 mcmcfit.py mcmc_input.dat
"""

import argparse
import multiprocessing as mp
import os
from pprintpp import pprint
from shutil import rmtree
from sys import exit
import h5py

import configobj
import emcee
import numpy as np

import mcmc_utils as utils
import plot_lc_model as plotCV
from CVModel import construct_model, extract_par_and_key

try:
    import ptemcee

    noPT = False
except:
    print("Failed to import ptemcee! Disabling parallel tempering.")
    noPT = True


# I need to wrap the model's ln_like, ln_prior, and ln_prob functions
# in order to pickle them :(
def ln_prior(param_vector, model):
    model.dynasty_par_vals = param_vector
    val = model.ln_prior()

    return val


def ln_prob(param_vector, model):
    model.dynasty_par_vals = param_vector
    val = model.ln_prob()

    return val


def ln_like(param_vector, model):
    model.dynasty_par_vals = param_vector
    val = model.ln_like()

    return val


def run_pt():
    print(
        "MCMC using parallel tempering at {} levels, for {} total walkers.".format(
            ntemps, nwalkers * ntemps
        )
    )

    # Create the initial ball of walker positions
    p_0 = utils.initialise_walkers_pt(
        p_0, p0_scatter_1, nwalkers, ntemps, ln_prior, model
    )

    # Create the sampler
    sampler = ptemcee.sampler.Sampler(
        nwalkers,
        npars,
        ln_like,
        ln_prob,
        loglargs=(model,),
        logpargs=(model,),
        ntemps=ntemps,
        pool=pool,
    )

    # Run the burnin phase
    print("\n\nExecuting the burn-in phase...")
    pos, prob, state = utils.run_burnin(sampler, p_0, nburn)

    # Do we want to do that again?
    if double_burnin:
        # If we wanted to run a second burn-in phase, then do. Scatter the
        # position about the first burn
        print("Executing the second burn-in phase")
        p_0 = pos[np.unravel_index(prob.argmax(), prob.shape)]
        p_0 = utils.initialise_walkers_pt(
            p_0, p0_scatter_2, nwalkers, ntemps, ln_prior, model
        )

    # Now, reset the sampler. We'll use the result of the burn-in phase to
    # re-initialise it.
    sampler.reset()
    print("Starting the main MCMC chain. Probably going to take a while!")

    # Get the column keys. Otherwise, we can't parse the results!
    col_names = "walker_no " + " ".join(model.dynasty_par_names) + " ln_prob"

    # Run production stage of parallel tempered mcmc
    sampler = utils.run_ptmcmc_save(
        sampler, pos, nprod, "chain_prod.txt", col_names=col_names
    )


def run(nwalkers, npars, ln_prob, ln_prior, p_0, model, pool, extend=False):
    backend = emcee.backends.HDFBackend("chain_prod.h5")
    if not extend:
        # reset backend, overwriting existing chain if needs be
        backend.reset(nwalkers, npars)
        # use the initial guess as the starting point
        p_0 = utils.initialise_walkers(p_0, p0_scatter_1, nwalkers, ln_prior, model)
    else:
        # extend the exisisting chain, using it's state as starting point
        p0 = None

    # Create the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        npars,
        ln_prob,
        args=(model,),
        pool=pool,
        backend=backend,
        moves=[
            (emcee.moves.DEMove(), 0.7),
            (emcee.moves.DESnookerMove(), 0.3),
        ],
    )
    sampler.run_mcmc(p_0, nburn, progress=True)

    # Run the burnin phase
    print("\n\nExecuting the burn-in phase...")
    state = sampler.run_mcmc(p_0, nburn, store=False, progress=True)

    # Do we want to do that again?
    if double_burnin:
        # If we wanted to run a second burn-in phase, then do. Scatter the
        # position about the first burn
        print("Executing the second burn-in phase")
        p_0 = state.coords[np.argmax(state.log_prob)]
        p_0 = utils.initialise_walkers(p_0, p0_scatter_2, nwalkers, ln_prior, model)

        # Run that burn-in
        state = sampler.run_mcmc(p_0, nburn, store=False, progress=True)

    # Now, reset the sampler. We'll use the result of the burn-in phase to
    # re-initialise it.
    sampler.reset()
    print("Starting the main MCMC chain. Probably going to take a while!")
    sampler.run_mcmc(state, nprod, store=True, progress=True)
    return sampler


if __name__ in "__main__":
    # Set up the parser.
    parser = argparse.ArgumentParser(
        description="""Execute an MCMC fit to a dataset."""
    )

    parser.add_argument(
        "input",
        help="The filename for the MCMC parameters' input file.",
        type=str,
    )

    parser.add_argument(
        "--debug", help="Enable the debugging flag in the model", action="store_true"
    )

    parser.add_argument(
        "--quiet", help="Do not plot the initial conditions", action="store_true"
    )

    parser.add_argument(
        "-e", "--extend", help="extend a previous chain", action="store_true"
    )

    args = parser.parse_args()
    input_fname = args.input
    debug = args.debug
    quiet = args.quiet

    if debug:
        if os.path.isdir("DEBUGGING"):
            rmtree("DEBUGGING")

    # Build the model from the input file
    model = construct_model(input_fname, debug)

    print("\nStructure:")
    pprint(model.structure)

    input_dict = configobj.ConfigObj(input_fname)

    # Read in information about mcmc
    nburn = int(input_dict["nburn"])
    nprod = int(input_dict["nprod"])
    nthreads = int(input_dict["nthread"])
    nwalkers = int(input_dict["nwalkers"])
    ntemps = int(input_dict["ntemps"])
    scatter_1 = float(input_dict["first_scatter"])
    scatter_2 = float(input_dict["second_scatter"])
    to_fit = int(input_dict["fit"])
    use_pt = bool(int(input_dict["usePT"]))
    double_burnin = bool(int(input_dict["double_burnin"]))
    comp_scat = bool(int(input_dict["comp_scat"]))

    if use_pt and noPT:
        print("\n\n!!!! Can't use Parallel tempering !!!!\n\n")
        use_pt = False

    # neclipses no longer strictly necessary, but can be used to limit the
    # maximum number of fitted eclipses
    try:
        neclipses = int(input_dict["neclipses"])
    except KeyError:
        neclipses = len(model.search_node_type("Eclipse"))
        print("The model has {} eclipses.".format(neclipses))

    # Wok out how many degrees of freedom we have in the model
    # How many data points do we have?
    dof = np.sum([ecl.lc.n_data for ecl in model.search_node_type("Eclipse")])
    # Subtract a DoF for each variable
    dof -= len(model.dynasty_par_names)
    # Subtract one DoF for the fit
    dof -= 1
    dof = int(dof)

    print(
        "\n\nInitial guess has a chisq of {:.3f} ({:d} D.o.F.).".format(
            model.chisq(), dof
        )
    )
    print("\nFrom the wrapper functions with the above parameters, we get;")
    pars = model.dynasty_par_vals
    print("a ln_prior of {:.3f}".format(ln_prior(pars, model)))
    print("a ln_like of {:.3f}".format(ln_like(pars, model)))
    print("a ln_prob of {:.3f}".format(ln_prob(pars, model)))
    print()
    if np.isinf(model.ln_prior()):
        print("ERROR: Starting position violates priors!")
        print("Offending parameters are:")

        pars, names = model.__get_descendant_params__()
        for par, name in zip(pars, names):
            print("{:>15s}_{:<5s}: Valid?: {}".format(par.name, name, par.isValid))

            if not par.isValid:
                print("  -> {}_{}".format(par.name, name))

        # Calculate ln_prior verbosely, for the user's benefit
        model.ln_prior(verbose=True)
        exit()

    # If we're not running the fit, plot our stuff.
    if not quiet:
        plotCV.nxdraw(model)
        plotCV.plot_model(
            model, True, save=True, figsize=(11, 8), save_dir="Initial_figs/"
        )
    if not to_fit:
        exit()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  MCMC Chain sampler, handled by emcee.                      #
    #  The below plugs the above into emcee's relevant functions  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # How many parameters do I have to deal with?
    npars = len(model.dynasty_par_vals)

    print("\n\nThe MCMC has {:d} variables and {:d} walkers".format(npars, nwalkers))
    print("(It should have at least 2*npars, {:d} walkers)".format(2 * npars))
    if nwalkers < 2 * npars:
        exit()

    # p_0 is the initial position vector of the MCMC walker
    p_0 = model.dynasty_par_vals

    # We cant to scatter that, so create an array of our scatter values.
    # This will allow us to tweak the scatter value for each individual
    # parameter.
    p0_scatter_1 = np.array([scatter_1 for _ in p_0])

    # If comp_scat is asked for, each value wants to be scattered differently.
    # Some want more, some less.
    if comp_scat:
        # scatter factors. p0_scatter_1 will be multiplied by these:
        scat_fract = {
            "ln_ampin_gp": 5.0,
            "ln_ampout_gp": 5.0,
            "tau_gp": 5.0,
            "q": 1,
            "rwd": 1,
            "dphi": 0.2,
            "dFlux": 1,
            "sFlux": 1,
            "wdFlux": 1,
            "rsFlux": 1,
            "rdisc": 1,
            "ulimb": 1e-6,
            "scale": 1,
            "fis": 1,
            "dexp": 1,
            "phi0": 1,
            "az": 1,
            "exp1": 1,
            "exp2": 1,
            "yaw": 1,
            "tilt": 1,
        }

        for par_i, name in enumerate(model.dynasty_par_names):
            # Get the parameter of this parName, striping off the node encoding
            key, _ = extract_par_and_key(name)

            # Skip the GP params
            if key.startswith("ln"):
                continue

            # Multiply it by the relevant factor
            p0_scatter_1[par_i] *= scat_fract[key]

        # Create another array for second burn-in
        p0_scatter_2 = p0_scatter_1 * (scatter_2 / scatter_1)

    # Run MCMC
    with mp.get_context("spawn").Pool(nthreads) as pool:
        if use_pt:
            run_pt(nwalkers, npars, ln_prob, ln_prior, p_0, model, pool)
            plotCV.fit_summary("chain_prod.txt", input_fname, automated=True)
        else:
            sampler = run(
                nwalkers, npars, ln_prob, ln_prior, p_0, model, pool, args.extend
            )
            # add parnames to chain file
            with h5py.File("chain_prod.h5", "r+") as f:
                f["mcmc"].attrs["var_names"] = model.dynasty_par_names
            plotCV.fit_summary("chain_prod.h5", input_fname, automated=True)
