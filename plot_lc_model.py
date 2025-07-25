"""
Plotting routines to accompany mcmcfit.py
"""

import argparse
import os
import warnings
from random import choice

import configobj
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import mcmc_utils as utils
from CVModel import construct_model
from utils import read_chain


def nxdraw(model):
    """Draw a hierarchical node map of a model."""

    # Build the network
    G = model.create_tree()
    pos = hierarchy_pos(G)

    # Figure has two inches of width per node
    figsize = (2 * float(G.number_of_nodes()), 8.0)
    print("Figure will be {}".format(figsize))

    _, ax = plt.subplots(figsize=figsize)
    nx.draw(G, ax=ax, pos=pos, with_labels=True, node_color="grey", font_weight="heavy")

    plt.show()


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
    the root will be found and used
    - if the tree is directed and this is given, then
    the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
    then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with
    other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        fail_msg = "cannot use hierarchy_pos on a graph that is not a tree"
        raise TypeError(fail_msg)

    if root is None:
        if isinstance(G, nx.DiGraph):
            # Allows back compatibility with nx version 1.11
            root = next(iter(nx.topological_sort(G)))
        else:
            root = choice(list(G.nodes))

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def _hierarchy_pos(
    G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
):
    """
    see hierarchy_pos docstring for most arguments

    pos: a dict saying where all nodes go if they have been assigned
    parent: parent of this branch. - only affects it if non-directed

    """

    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    children = list(G.neighbors(root))

    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)

    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(
                G,
                child,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=nextx,
                pos=pos,
                parent=root,
            )

    return pos


def plot_eclipse(
    ecl_node, save=False, figsize=(11.0, 8.0), fname=None, save_dir=".", ext=".pdf"
):
    """Create a plot of the eclipse's data.

    If save is True, a copy of the figure is saved.

    If fname is defined, save the figure with that filename. Otherwise,
    infer one from the data filename
    """
    # Generate the lightcurve of the total, and the components.
    flx = ecl_node.cv.calcFlux(ecl_node.cv_parlist, ecl_node.lc.x, ecl_node.lc.w)
    wd_flx = ecl_node.cv.ywd
    sec_flx = ecl_node.cv.yrs
    BS_flx = ecl_node.cv.ys
    disc_flx = ecl_node.cv.yd

    # print("This model has a chisq of {:.3f}".format(ecl_node.chisq()))

    # Start the plotting area
    fig, axs = plt.subplots(2, sharex=True, figsize=figsize, height_ratios=(2, 1))

    # Plot the data first. Also do errors
    axs[0].errorbar(
        ecl_node.lc.x,
        ecl_node.lc.y,
        yerr=ecl_node.lc.ye,
        linestyle="none",
        ecolor="grey",
        zorder=1,
    )
    axs[0].step(ecl_node.lc.x, ecl_node.lc.y, where="mid", color="black")

    # Plot the model over the data
    axs[0].plot(ecl_node.lc.x, wd_flx, color="lightblue", label="WD")
    axs[0].plot(ecl_node.lc.x, sec_flx, color="magenta", label="Sec")
    axs[0].plot(ecl_node.lc.x, BS_flx, color="darkblue", label="BS")
    axs[0].plot(ecl_node.lc.x, disc_flx, color="brown", label="Disc")
    axs[0].plot(ecl_node.lc.x, flx, color="red")
    axs[0].legend()

    # Plot the errorbars
    axs[1].errorbar(
        ecl_node.lc.x,
        ecl_node.lc.y - flx,
        yerr=ecl_node.lc.ye,
        linestyle="none",
        ecolor="grey",
        zorder=1,
    )
    axs[1].step(ecl_node.lc.x, ecl_node.lc.y - flx, where="mid", color="black")

    # 0 residuals line, to guide the eye
    axs[1].axhline(0.0, linestyle="--", color="black", alpha=0.7, zorder=0)

    # Labelling. Top one gets title, bottom one gets x label
    title_text = "{} --- chisq: {:.1f} --- ln_prob: {:.1f}".format(
        ecl_node.lc.name, ecl_node.chisq(), ecl_node.ln_prob()
    )
    axs[0].set_title(title_text)
    axs[0].set_ylabel("Flux, mJy")

    axs[1].set_xlabel("Phase")
    axs[1].set_ylabel("Residual Flux, mJy")

    # Arrange the figure on the page, and show it
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    if save:
        # Check that save_dir exists
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # If we didnt get told to use a certain fname, use this node's name
        if fname is None:
            fname = ecl_node.lc.name.replace(".calib", "")
            fname = fname.replace(".txt", "")
            fname += ext

        # Make the filename
        fname = "/".join([save_dir, fname])

        # If the user specified a path like './figs/', then the above could
        # return './figs//Eclipse_N.pdf'; I want to be robust against that.
        while "//" in fname:
            fname = fname.replace("//", "/")

        plt.savefig(fname)

    return fig, axs


def plot_GP_eclipse(
    ecl_node, save=False, figsize=(11.0, 8.0), fname=None, save_dir=".", ext=".pdf"
):
    """Plot my data. Returns fig, ax

    If save is True, save the figures.
    Figsize is passed to matplotlib.
    """

    # Get the figure and axes from the eclipse
    fig, ax = plot_eclipse(ecl_node, False, figsize, fname, save_dir, ext)

    # Get the residuals of the model
    residuals = ecl_node.lc.y - ecl_node.calcFlux()

    # Create the GP of this eclipse
    gp = ecl_node.create_GP()
    # Compute the GP
    gp.compute(ecl_node.lc.x, ecl_node.lc.ye)

    # Draw samples from the GP
    samples = gp.sample_conditional(residuals, ecl_node.lc.x, size=300)

    # Get the mean, mu, standard deviation, and
    mu = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

    ax[1].fill_between(
        ecl_node.lc.x,
        mu + (1.0 * std),
        mu - (1.0 * std),
        color="r",
        alpha=0.4,
        zorder=20,
    )

    if save:
        # Check that save_dir exists
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # If we didnt get told to use a certain fname, use this node's name
        if fname is None:
            fname = ecl_node.lc.name.replace(".calib", "")
            fname = fname.replace(".txt", "")
            fname += ext

        # Make the filename
        fname = "/".join([save_dir, fname])

        # If the user specified a path like './figs/', then the above could
        # return './figs//Eclipse_N.pdf'; I want to be robust against that.
        while "//" in fname:
            fname = fname.replace("//", "/")

        plt.savefig(fname)

    return fig, ax


def plot_model(model, show, *args, **kwargs):
    """Calls the relevant plotter for each eclipse contained in the model.
    Passes *args and **kwargs to it.

    Inputs:
    -------
      model, Node:
        The model to be plotted. This will be queried for child Eclipse nodes,
        which will then be passed to the plot_eclipse and plot_GP_eclipse funcs
      show, Bool:
        If show is True, call plt.show(). Otherwise, just save the figures
      args, kwargs:
        Passed to the plot_eclipse and plot_GP_eclipse functions.

    Returns:
    --------
      None.
    """

    eclipses = model.search_node_type("Eclipse")
    for eclipse in eclipses:
        if str(eclipse.__class__.__name__) in ["SimpleEclipse", "ComplexEclipse"]:
            fig, ax = plot_eclipse(eclipse, *args, **kwargs)
        elif str(eclipse.__class__.__name__) in ["SimpleGPEclipse", "ComplexGPEclipse"]:
            fig, ax = plot_GP_eclipse(eclipse, *args, **kwargs)

        if show:
            plt.show()

        plt.close()
        del fig
        del ax


def fit_summary(
    chain_fname,
    input_fname,
    nskip=0,
    thin=1,
    automated=False,
    corners=True,
):
    """Takes the chain file made by mcmcfit.py and summarises the initial
    and final conditions. Uses the input filename normally supplied to
    mcmcfit.py to accurately reconstruct the model.

    Inputs:
    -------
      chain_fname: str
        The chain file to be read in. This must be the chain_prod.txt version,
        i.e. the one that tracks which walker is doing what.
      input_fname: str
        The input file normally supplied to mcmcfit.py.
      nskip: int
        This many steps will be stripped from the front of the chain
      thin: int
        Only every 'thin'-th step will be used.
      automated: bool
        If this is True, no figures will actually show, and will only be
        saved to file.
      corners: bool
        If this is True, create and save corner plots of each eclipse's params
    """

    # Retrieve some info from the input dict.
    input_dict = configobj.ConfigObj(input_fname)

    nsteps = int(input_dict["nprod"])
    nwalkers = int(input_dict["nwalkers"])

    # Read in the data
    print("Reading in the data...")
    colKeys, df = read_chain(chain_fname)

    print("Done!\nData shape: {}".format(df.shape))
    print(
        "Expected a shape (nwalkers, nprod, npars+2): ({}, {}, {})".format(
            nwalkers, nsteps, len(colKeys) + 2
        )
    )

    nwalkers = df["walker_no"].max() + 1
    df["step"] = df.index // nwalkers
    nsteps = df["step"].max() + 1

    print(
        "The chain file actually contains {} walkers, over {} steps.".format(
            nwalkers, nsteps
        )
    )

    # Create the Final_figs and Initial_figs directories.
    if not os.path.isdir("Final_figs"):
        os.mkdir("Final_figs")
    if not os.path.isdir("Initial_figs"):
        os.mkdir("Initial_figs")

    # Plot an image of the walker likelihoods over time.

    print("Reading in the chain file for likelihoods...")
    walker_likes = df["ln_prob"]

    # Get a numpy array of likelihoods, shaped (nwalkers, nprod).
    walker_likes = np.asarray(
        [walker_likes.loc[df["walker_no"] == i] for i in range(nwalkers)]
    )

    # Plot the mean likelihood evolution
    std = np.std(walker_likes, axis=0)
    likes = np.mean(walker_likes, axis=0)

    steps = np.asarray(df["step"].loc[df["walker_no"] == 1])

    likes = likes[nskip::thin]
    std = std[nskip::thin]
    steps = steps[nskip::thin]

    # Make the likelihood plot
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.fill_between(steps, likes - std, likes + std, color="red", alpha=0.4)
    ax.plot(steps, likes, color="green")

    ax.set_xlabel("Step")
    ax.set_ylabel("ln_prob")

    plt.tight_layout()

    oname = "Final_figs/ln_prob.pdf"

    plt.savefig(oname)
    print("Saved to {}".format(oname))
    if automated:
        plt.close()
    else:
        plt.show(block=False)

    # Check that the user is sure about not thinning or skipping, in light of the chain size.
    if not automated:
        if nskip == 0:
            nskip = input(
                "You opted not to skip any data, are you still sure? (default 0)\n-> nskip: "
            )
            try:
                nskip = int(nskip)
            except ValueError:
                nskip = 0
    data = df.loc[df["step"] >= nskip]
    print("First step is {}".format(steps.min()))
    ax.set_xlim(left=nskip + steps.min(), right=steps.max())
    fig.canvas.draw_idle()

    if not automated:
        if thin == 1:
            print("Thinning omits every Nth step of the chain.")
            thin = input(
                "You opted not to thin the data, are you sure? (defaults to no thinning)\n-> thin: "
            )
            try:
                thin = int(thin)
            except ValueError:
                thin = 1
    if thin > 1:
        data = data.loc[data.step % thin == 0]

    nwalkers = len(np.unique(data["walker_no"]))
    nsteps = len(np.unique(data["step"]))
    npars = len(colKeys)
    print(
        "After thinning and skipping: {} walkers, {} steps, {} params".format(
            nwalkers, nsteps, npars
        )
    )
    plt.close()

    # Analyse the chain. Take the mean as the result, and 2 sigma as the error
    result = pd.DataFrame()
    result["median"] = data.quantile(0.50)
    result["84th percentile"] = data.quantile(0.84)
    result["16th percentile"] = data.quantile(0.16)

    # report. While we're doing that, make a dict of the values we want to assign
    # to the new version of the model.
    print("Result of the chain:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(result)
    modparams = result.loc[colKeys, :]
    modparams.to_csv("modparams.csv", header=True)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Use the input file to reconstruct the model #
    # # # # # # # # # # # # # # # # # # # # # # # #

    model = construct_model(input_fname)
    parDict = {k: v for k, v in result["median"].to_dict().items() if k in colKeys}

    # We want to know where we started, so we can evaluate improvements.
    # Wok out how many degrees of freedom we have in the model
    eclipses = model.search_node_type("Eclipse")
    # How many data points do we have?
    dof = np.sum([ecl.lc.x.size for ecl in eclipses])
    # Subtract a DoF for each variable
    dof -= len(model.dynasty_par_vals)
    # Subtract one DoF for the fit
    dof -= 1

    model_preport = "The following is the result of the MCMC fit running in:\n"
    model_preport += "  Machine name: {}\n  Directory: {}\n".format(
        os.uname().nodename, os.getcwd()
    )
    model_preport += "\n\nInitial guess has a chisq of {:.3f} ({:d} D.o.F.).\n".format(
        model.chisq(), dof
    )
    model_preport += "\nEvaluating the model, we get;\n"
    model_preport += "a ln_prior of {:.3f}\n".format(model.ln_prior())
    model_preport += "a ln_like of {:.3f}\n".format(model.ln_like())
    model_preport += "a ln_prob of {:.3f}\n".format(model.ln_prob())
    print(model_preport)

    if not automated:
        print("Initial conditions being plotted now...")

    plot_model(
        model, not automated, save=True, figsize=(11, 8), save_dir="Initial_figs/"
    )

    # # # # # # # # # # # # # # # # #
    # The model is now fully built. #
    # # # # # # # # # # # # # # # # #

    model.dynasty_par_dict = parDict

    model_report = ""
    model_report += "\n\nMCMC result has a chisq of {:.3f} ({:d} D.o.F.).\n".format(
        model.chisq(), dof
    )
    model_report += "\nEvaluating the model, we get;\n"
    model_report += "a ln_prior of {:.3f}\n".format(model.ln_prior())
    model_report += "a ln_like of {:.3f}\n".format(model.ln_like())
    model_report += "a ln_prob of {:.3f}\n".format(model.ln_prob())
    model_report += "\n\nThe final fits of this chain are attached below.\n\n"

    if not automated:
        print("Final conditions being plotted now...")
        input("> ")

    plot_model(model, not automated, save=True, figsize=(11, 8), save_dir="Final_figs/")

    if corners:
        # Corner plots. Collect the eclipses.
        eclipses = model.search_node_type("Eclipse")
        for eclipse in eclipses:
            # Get the par names from the eclipse.
            print("Doing the corner plot for eclipse {}".format(eclipse))

            # Sometimes, the walkers can fall into a phi0 == 0.0. When this happens,
            # the thumbplot gets confused and dies, since there's no range.
            # This parameter it typically only important if something goes badly
            # wrong anyway, so if it gets stuck here, just filter it out.

            # Eclipse-level variables
            par_labels = [
                "{}_{}".format(par, eclipse.label) for par in eclipse.node_par_names
            ]

            # Get the par names from the band
            band = eclipse.parent
            par_labels += [
                "{}_{}".format(par, band.label) for par in band.node_par_names
            ]

            # get the par names from the core part of the model
            my_model = band.parent
            par_labels += [
                "{}_{}".format(par, my_model.label) for par in my_model.node_par_names
            ]

            # Only plot parameters that are in the chain file.
            par_labels = list(set(par_labels).intersection(colKeys))
            print("\nMy corner plot labels are:")
            print(par_labels)

            # Get the indexes in the chain file, and gather those columns
            chain_slice = np.asarray(data[par_labels])
            print("chain_slice has the shape:", chain_slice.shape)

            # If I've nothing to plot, continue to the next thing.
            if par_labels == []:
                continue

            # skip immobile pars (with warning)
            dev = np.std(chain_slice, axis=0)
            if any(dev == 0):
                stationary_mask = np.where(dev == 0)[0]
                bad_params = ", ".join([par_labels[i] for i in stationary_mask])
                warnings.warn(f"chains for {bad_params} are stationary")
                dev[stationary_mask] = np.nan
                chain_slice = chain_slice[:, ~np.isnan(dev)]
                par_labels = [p for p, d in zip(par_labels, dev) if not np.isnan(d)]

            print(
                "\nAfter checking for immobile variables, my corner plot labels are:~"
            )
            print(par_labels)
            print("chain_slice has the shape:", chain_slice.shape)
            print("Par_labels has the shape: ", len(par_labels))

            fig = utils.thumbPlot(chain_slice, par_labels)
            fname = os.path.split(eclipse.lc.fname)[1]
            fname = os.path.splitext(fname)[0]
            oname = "Final_figs/" + fname + "_corners.pdf"
            print("Saving to {}...".format(oname))
            plt.savefig(oname)
            plt.close("all")

            del chain_slice
            try:
                del fig
            except NameError:
                pass


if __name__ == "__main__":
    # CLI Arguments
    parser = argparse.ArgumentParser(
        description="""Takes a chain file and the input file that created it, and
        summarises the results.
        Plots the before and after lightcurves, and the corner plot of each eclipse.
        Plots the likelihood evolution."""
    )

    parser.add_argument(
        "chain_fname",
        help="The filename for the MCMC chain",
        type=str,
    )
    parser.add_argument(
        "input_fname",
        help="The input command file for the MCMC chain",
        type=str,
    )
    parser.add_argument(
        "--nskip",
        help="How many steps to cut off the front of the MCMC chain",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--thin",
        help="I will only take every Nth step of the chain",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        help="If I'm being quiet, no figure will be shown, only saved to file.",
    )
    parser.add_argument(
        "--no-corners",
        dest="corners",
        action="store_false",
        help="Do not create the corner plots",
    )

    args = parser.parse_args()

    chain_fname = args.chain_fname
    input_fname = args.input_fname

    nskip = args.nskip
    thin = args.thin

    automated = bool(args.quiet)
    corners = args.corners

    fit_summary(
        chain_fname,
        input_fname,
        nskip,
        thin,
        automated=automated,
        corners=corners,
    )
