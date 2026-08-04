"""
Subclasses from the `model` module, that actually comprise the tree
structure of the model fed to emcee. The `trunk` is an LCModel or GPLCModel
node, with child Bands, that have XEclipse leaves to evaluate the CV lightcurve
fit to the data. Data is stored in the Lightcurve class.
"""

import os

import configobj
import george
import lfit_rust as lfit
import numpy as np
import roche

from model import Node, Param, extract_par_and_key

BIG = 9e99


class Lightcurve:
    """This object keeps track of the observational data.
    Can be generated from a file, with Lightcurve.from_calib(file)."""

    def __init__(self, name, x, y, ye, w=None):
        """Here, hold this."""
        self.name = name

        self.fname = None

        if w is None:
            w = np.mean(np.diff(x)) * np.ones_like(x) / 2.0

        self.x = x
        self.y = y
        self.ye = ye
        self.w = w

    @property
    def n_data(self):
        return self.x.shape[0]

    @classmethod
    def from_calib(cls, fname, name=None):
        """Read in a calib file, of the format;
        phase flux error

        and treat lines with # as commented out.
        """
        delimiters = " ,|"
        for delimiter in delimiters:
            try:
                data = np.loadtxt(fname, delimiter=delimiter, comments="#")
                break
            except ValueError:
                print(
                    "Couldn't split the calib file with '{}', trying next delimiter...".format(
                        delimiter
                    )
                )
        try:
            phase, flux, error = data.T
        except ValueError:
            # we probably have a Tseries style file
            phase, _, flux, error, bmask = data.T

        # Filter out nans.
        mask = np.where(np.isnan(flux) == 0)

        phase = phase[mask]
        flux = flux[mask]
        error = error[mask]

        width = np.mean(np.diff(phase)) * np.ones_like(phase) / 2.0

        # Set the name of this eclipse as the filename of the data file.
        if name is None:
            _, name = os.path.split(fname)

        lc = cls(name, phase, flux, error, width)
        lc.fname = fname
        return lc

    def trim(self, lo, hi):
        """Trim the data, so that all points are in the x range lo > xi > hi"""
        xt = self.x

        mask = (xt > lo) & (xt < hi)

        self.x = self.x[mask]
        self.y = self.y[mask]
        self.ye = self.ye[mask]
        self.w = self.w[mask]


# Subclasses.
class SimpleEclipse(Node):
    """Subclass of Node, specifically for storing a single eclipse.
    Uses the simple BS model.
    Lightcurve data is stored on this level.

    Inputs:
    -------
      lightcurve; Lightcurve:
        A Lightcurve object, containing data
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Node, optional:
        The parent of this node.
      children; list(Node), or Node:
        The children of this node. Single Node is also accepted
    """

    # Define this subclasses parameters
    node_par_names = ("dFlux", "sFlux", "rdisc", "scale", "az", "fis", "dexp", "phi0")

    def __init__(self, lightcurve, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If the lightcurve is a Lightcurve object, save it. Otherwise,
        # read it in from the file.
        if isinstance(lightcurve, Lightcurve):
            self.lc = lightcurve
        elif isinstance(lightcurve, str):
            self.lc = Lightcurve.from_calib(lightcurve)
        else:
            msg = "Argument lightcurve is not a string or Lightcurve! "
            msg += "Got {}".format(lightcurve)
            raise TypeError(msg)

        # Create the CV object
        self.cv = lfit.CV(self.cv_parlist)

        self.log(
            "SimpleEclipse.__init__", "Successfully ran the SimpleEclipse initialiser."
        )

    def calcFlux(self):
        """Fetch the CV parameter vector, and generate it's model lightcurve"""
        # Get the model CV lightcurve across our data.
        try:
            flx = self.cv.calcFlux(self.cv_parlist, self.lc.x, self.lc.w)
        except Exception as e:
            # print(repr(e))
            # msg = "Error: {}; parlist: {}".format(str(e), repr(self.cv_parlist))
            # print(msg)
            flx = np.nan
        return flx

    def calcComponents(self):
        """Return a list of the component fluxes as well as the total
        returns:
          (tot_flx, wdFlux, sFlux, rsFlux, dFlux)
        """
        flx = self.cv.calcFlux(self.cv_parlist, self.lc.x, self.lc.w)
        return flx, self.cv.ywd, self.cv.ys, self.cv.yrs, self.cv.yd

    def chisq(self):
        """Return the chisq of this eclipse, given current params."""
        flx = self.calcFlux()

        # If the model gets any nans, return inf
        if np.any(np.isnan(flx)):
            return BIG

        # Calculate the chisq of this model.
        chisq = ((self.lc.y - flx) / self.lc.ye) ** 2
        chisq = np.sum(chisq)

        return chisq

    def ln_like(self):
        """Calculate the chisq of this eclipse, against the data stored in its
        lightcurve object."""

        chisq = self.chisq()
        return -0.5 * chisq

    def check_validity(self) -> bool:
        """
        Check for validity of the parameters at this level.

        Some parameter combinations are invalid (e.g: no eclipse for q and dphi). Here we
        check for any of these invalid combinations
        """
        # Before we start, I'm going to collect the necessary parameters. By
        # only calling this once, we save a little effort.
        ancestor_param_dict = self.ancestor_param_dict

        ##############################################
        # ~~ Is the disc large enough to precess? ~~ #
        ##############################################

        # Defined the maximum size of the disc before it starts precessing, as
        # a fraction of Roche Radius
        rdisc_max_a = 0.46

        # get the location of the L1 point from q
        q = ancestor_param_dict["q"].currVal
        try:
            xl1 = roche.xl1(q)
        except AssertionError:
            return False

        # Get the rdisc, scaled to the Roche Radius
        rdisc = ancestor_param_dict["rdisc"].currVal
        rdisc_a = rdisc * xl1
        ##############################################
        # ~~~~~ Does the stream miss the disc? ~~~~~ #
        ##############################################
        try:
            # If the stream does not intersect the disc, this throws an error
            r, _ = roche.bspot(q, rdisc_a)
        except Exception:
            return False

        """
        if rdisc_a > rdisc_max_a:
            return False

        ##############################################
        # ~~~~~ Is the BS scale physically OK? ~~~~~ #
        ##############################################
        # We're gonna check to see if the BS scle is #
        # in the range rwd/4 < (BS scale) < rwd*3.   #
        # If it isn't, then it's either too          #
        # concentrated to make sense, or so large    #
        # that our approximation of a smooth disc is #
        # no longer a good idea.                     #
        #
        # We'll also check the BS scale is less than #
        # 1/10th, since otherwise the bright spot    #
        # might extend more than one separation from #
        # the spot position.                         #
        ##############################################

        # Get the WD radius.
        rwd = ancestor_param_dict["rwd"].currVal

        # Enforce the BS scale being within these limits
        rmax = min(1 / 10, rwd * 4.0)
        rmin = rwd / 4.0

        scale = ancestor_param_dict["scale"].currVal
        if scale > rmax or scale < rmin:
            return False
        """

        # If we pass all that, then the parameters are valid.
        return True

    @property
    def cv_parnames(self):
        names = [
            "wdFlux",
            "dFlux",
            "sFlux",
            "rsFlux",
            "q",
            "dphi",
            "rdisc",
            "ulimb",
            "rwd",
            "scale",
            "az",
            "fis",
            "dexp",
            "phi0",
        ]

        return names

    @property
    def cv_parlist(self):
        """Construct the parameter list needed by the CV"""

        par_name_list = self.cv_parnames

        param_dict = self.ancestor_param_dict
        try:
            parlist = [param_dict[key].currVal for key in par_name_list]
        except KeyError as error:
            print("Parameter dict:")
            print("{")
            for key, val in param_dict.items():
                print("    {}: {}".format(key, val))
            print("}")
            raise error
        return parlist


class ComplexEclipse(SimpleEclipse):
    """Subclass of Node, specifically for storing a single eclipse.
    Uses the complex BS model.
    Lightcurve data is stored on this level.

    Inputs:
    -------
      lightcurve; Lightcurve:
        A Lightcurve object, containing data
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Node, optional:
        The parent of this node.
      children; list(Node), or Node:
        The children of this node. Single Node is also accepted
    """

    node_par_names = (
        "dFlux",
        "sFlux",
        "rdisc",
        "scale",
        "az",
        "fis",
        "dexp",
        "phi0",
        "exp1",
        "exp2",
        "yaw",
        "tilt",
    )

    @property
    def cv_parnames(self):
        names = [
            "wdFlux",
            "dFlux",
            "sFlux",
            "rsFlux",
            "q",
            "dphi",
            "rdisc",
            "ulimb",
            "rwd",
            "scale",
            "az",
            "fis",
            "dexp",
            "phi0",
            "exp1",
            "exp2",
            "tilt",
            "yaw",
        ]

        return names

    def check_validity(self) -> bool:
        """
        Check for validity of the parameters at this level.

        Some parameter combinations are invalid (e.g: no eclipse for q and dphi). Here we
        check for any of these invalid combinations
        """
        # Before we start, I'm going to collect the necessary parameters. By
        # only calling this once, we save a little effort.
        ancestor_param_dict = self.ancestor_param_dict

        ##############################################
        # ~~~~ Is the bright spot max sensible? ~~~~ #
        ##############################################
        # The location of the bright spot max is     #
        # (in units of scale lengths) given by       #
        #     pow(exp1/exp2, 1/exp2)                 #
        exp1 = ancestor_param_dict["exp1"].currVal
        exp2 = ancestor_param_dict["exp2"].currVal
        bs_max = pow(exp1 / exp2, 1 / exp2)
        if bs_max > 5:
            return False

        return super().check_validity()


class Band(Node):
    """Subclass of Node, specific to observation bands. Contains the eclipse
    objects taken in this band.

    Inputs:
    -------
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Node, optional:
        The parent of this node.
      children; list(Node), or Node:
        The children of this node. Single Node is also accepted
    """

    # What kind of parameters are we storing here?
    node_par_names = ("wdFlux", "rsFlux", "ulimb")

    @property
    def eclipses(self):
        return list(self.search_node_type("Eclipse"))


class LCModel(Node):
    """Top layer Node class. Contains Bands, which contain Eclipses.
    Inputs:
    -------
      label; str:
        A label to apply to the node. Mostly used when searching trees.
      parameter_objects; list(Param), or Param:
        The parameter objects that correspond to this node. Single Param is
        also accepted.
      parent; Node, optional:
        The parent of this node.
      children; list(Node), or Node:
        The children of this node. Single Node is also accepted
    """

    # Set the parameter names for this layer
    node_par_names = ("q", "dphi", "rwd")

    @property
    def eclipses(self):
        return list(self.search_node_type("Eclipse"))

    def check_validity(self) -> bool:
        """
        Check for invalid parameter combinations.

        This is the top level check (for q and dphi). If this is OK, we will also
        check the validity of the children nodes, which will check for other invalid combinations.
        """
        dphi = getattr(self, "dphi").currVal
        q = getattr(self, "q").currVal
        # check for invalid q, dphi (phase width too large for q)
        if roche.findi(q, dphi) < 0.0:
            return False

        # if the top level is OK, call Node's check validity, which will
        # descend into the children and check their validity as well
        return super().check_validity()

    def ln_like(self):
        """
        Calculate the log likelihood of the model.

        First we evaluate the validity of the model and return -inf if invalid.
        If valid, we call the super class ln_like, which will descend into the children
        and calculate the log likelihood of the model.
        """
        if not self.check_validity():
            return -np.inf
        return super().ln_like()


class GPLCModel(LCModel):
    """This is a subclass of the LCModel class. It uses the Gaussian Process.

    This version will, rather than evaluating via chisq, evaluate the
    likelihood of the model by calculating the residuals between the model
    and the data and computing the likelihood of those data given certain
    Gaussian Process hyper-parameters.

    These parameters require some explaination.
    tau_gp:
      the timescale of the covariance matrix
    ln_ampin_gp:
      The base amplitude of the covariance matrix
    ln_ampout_gp:
      The additional amplitude of the covariance matrix, when the WD
      is visible
    """

    # Add the GP params
    node_par_names = LCModel.node_par_names
    node_par_names += ("ln_ampin_gp", "ln_ampout_gp", "tau_gp")


class SimpleGPEclipse(SimpleEclipse):
    # Set the initial values of q, rwd, and dphi. These will be used to
    # caclulate the location of the GP changepoints. Setting to initially
    # unrealistically high values will ensure that the first time
    # calcChangepoints is called, the changepoints are calculated.
    _olddphi = 9e99
    _oldq = 9e99
    _oldrwd = 9e99

    # _dist_cp is initially set to whatever, it will be overwritten anyway.
    _dist_cp = 9e99

    def calcChangepoints(self):
        """Caclulate the WD ingress and egresses, i.e. where we want to switch
        on or off the extra GP amplitude.

        Requires an eclipse object, since this is specific to a given phase
        range.
        """

        self.log("SimpleGPEclipse.calcChangepoints", "Calculating GP changepoints")

        # Also get object for dphi, q and rwd as this is required to determine
        # changepoints
        pardict = self.ancestor_param_dict

        dphi = pardict["dphi"]
        q = pardict["q"]
        rwd = pardict["rwd"]
        phi0 = pardict["phi0"]

        # Have they changed significantly?
        # If not, dont bother recalculating dist_cp
        dphi_change = np.fabs(self._olddphi - dphi.currVal) / dphi.currVal
        q_change = np.fabs(self._oldq - q.currVal) / q.currVal
        rwd_change = np.fabs(self._oldrwd - rwd.currVal) / rwd.currVal

        # Check to see if our model parameters have changed enough to
        # significantly change the location of the changepoints.
        if (dphi_change > 1.2) or (q_change > 1.2) or (rwd_change > 1.2):
            self.log(
                "SimpleGPEclipse.calcChangepoints",
                "The GP changepoint locations have chnged significantly enough to warrant a recalculation...",
            )

            # Calculate inclination
            inc = roche.findi(q.currVal, dphi.currVal)

            # Calculate wd contact phases 3 and 4
            phi3, phi4 = roche.wdphases(q.currVal, inc, rwd.currVal, ntheta=10)

            # Calculate length of wd egress
            dpwd = phi4 - phi3

            # Distance from changepoints to mideclipse
            dist_cp = (dphi.currVal + dpwd) / 2.0

            # save these values for speed
            self._dist_cp = dist_cp
            self._oldq = q.currVal
            self._olddphi = dphi.currVal
            self._oldrwd = rwd.currVal
        else:
            self.log("SimpleGPEclipse.calcChangepoints", "Using old values of dist_cp")
            # Use the old values
            dist_cp = self._dist_cp

        # Find location of all changepoints
        min_ecl = int(np.floor(self.lc.x.min()))
        max_ecl = int(np.ceil(self.lc.x.max()))

        eclipses = [
            e
            for e in range(min_ecl, max_ecl + 1)
            if np.logical_and(e > self.lc.x.min(), e < 1 + self.lc.x.max())
        ]

        changepoints = []
        for e in eclipses:
            # When did the last eclipse end?
            egress = (e - 1) + dist_cp + phi0.currVal
            # When does this eclipse start?
            ingress = e - dist_cp + phi0.currVal
            changepoints.append([egress, ingress])

        self.log(
            "SimpleGPEclipse.calcChangepoints",
            "Computed GP changepoints as:\n{}".format(changepoints),
        )
        return changepoints

    def create_GP(self):
        """Constructs a kernel, which is used to create Gaussian processes.

        Creates kernels for both inside and out of eclipse,
        works out the location of any changepoints present, constructs a single
        (mixed) kernel and uses this kernel to create GPs

        Requires an Eclipse object to create the GP for."""

        self.log("SimpleGPEclipse.create_GP", "Creating a new GP")

        # Get objects for ln_ampin_gp, ln_ampout_gp, tau_gp and find the exponential
        # of their current values
        pardict = self.ancestor_param_dict

        ln_ampin = pardict["ln_ampin_gp"]
        ln_ampout = pardict["ln_ampout_gp"]
        tau = pardict["tau_gp"]

        ampin_gp = np.exp(ln_ampin.currVal)
        ampout_gp = np.exp(ln_ampout.currVal)
        tau_gp = tau.currVal

        # Calculate kernels for both out of and in eclipse WD eclipse
        # Kernel inside of WD has smaller amplitude than that of outside
        # eclipse.

        # First, get the changepoints
        changepoints = self.calcChangepoints()

        # We need to make a fairly complex kernel.
        # Global flicker
        self.log("SimpleGPEclipse.create_GP", "Constructing a new kernel")
        kernel = ampin_gp * george.kernels.Matern32Kernel(tau_gp**2)
        # inter-eclipse flicker
        for gap in changepoints:
            kernel += ampout_gp * george.kernels.Matern32Kernel(tau_gp**2, block=gap)

        # Use that kernel to make a GP object
        georgeGP = george.GP(kernel, solver=george.HODLRSolver)

        self.log("SimpleGPEclipse.create_GP", "Successfully created a new GP!")
        return georgeGP

    def ln_like(self):
        """The GP sits at the top of the tree. It replaces the LCModel
        class. When the evaluate function is called, this class should
        hijack it, calculate the residuals of all the eclipses in the tree,
        and find the likelihood of each of those residuals given the current GP
        hyper-parameters.

        Inputs:
        -------
        label; str:
            A label to apply to the node. Mostly used when searching trees.
        parameter_objects; list(Param), or Param:
            The parameter objects that correspond to this node. Single Param is
            also accepted.
        parent; Node, optional:
            The parent of this node.
        children; list(Node), or Node:
            The children of this node. Single Node is also accepted
        """

        self.log("SimpleGPEclipse.ln_like", "Computing ln_like for a GP")

        # For each eclipse, I want to know the log likelihood of its residuals
        gp_ln_like = 0.0

        # Get the residuals of the model
        residuals = self.lc.y - self.calcFlux()
        # Did the model turn out ok?
        if np.any(np.isinf(residuals)) or np.any(np.isnan(residuals)):
            if self.DEBUG:
                msg = "GP ln_like computed inf or nan residuals for the model. Returning -np.inf for the likelihood."
                self.log("SimpleGPEclipse.ln_like", msg)
            return -np.inf

        # Create the GP of this eclipse
        gp = self.create_GP()
        # Compute the GP
        gp.compute(self.lc.x, self.lc.ye)

        # The 'quiet' argument tells the GP to return -inf when you get
        # an invalid kernel, rather than throwing an exception.
        gp_ln_like = gp.log_likelihood(residuals, quiet=True)

        if self.DEBUG:
            self.log(
                "SimpleGPEclipse.ln_like",
                "GP computed a ln_like of {}".format(gp_ln_like),
            )

        return gp_ln_like


class ComplexGPEclipse(SimpleGPEclipse):
    # Exactly as the simple GP Eclipse, but this time with the extra 4 params.
    node_par_names = ComplexEclipse.node_par_names

    @property
    def cv_parnames(self):
        names = [
            "wdFlux",
            "dFlux",
            "sFlux",
            "rsFlux",
            "q",
            "dphi",
            "rdisc",
            "ulimb",
            "rwd",
            "scale",
            "az",
            "fis",
            "dexp",
            "phi0",
            "exp1",
            "exp2",
            "tilt",
            "yaw",
        ]

        return names


def construct_model(input_file, nodata=False):
    """Takes an input filename, and parses it into a model tree.

    Inputs:
    -------
      input_file, str:
        The input.dat file to be parsed
      debug, bool:
        Enable the debugging flag for the Nodes. Debugging will be written to
        a file.

    Output:
    -------
      model root node
    """

    input_dict = configobj.ConfigObj(input_file)

    # Do we use the complex model? Do we use the GP?
    is_complex = bool(int(input_dict["complex"]))
    use_gp = bool(int(input_dict["useGP"]))

    # neclipses no longer strictly necessary, but can be used to limit the
    # maximum number of fitted eclipses
    try:
        neclipses = int(input_dict["neclipses"])
    except KeyError:
        # Read in all the available eclipses
        neclipses = 9999

    # # # # # # # # # # # # # # # # #
    # Get the initial model setup # #
    # # # # # # # # # # # # # # # # #
    # Start by creating the overall Node. Gather the parameters:
    if use_gp:
        core_par_names = GPLCModel.node_par_names
        core_pars = [
            Param.fromString(name, input_dict[name]) for name in core_par_names
        ]

        # and make the model object with no children
        model = GPLCModel("core", core_pars)
    else:
        core_par_names = LCModel.node_par_names
        core_pars = [
            Param.fromString(name, input_dict[name]) for name in core_par_names
        ]

        # and make the model object with no children
        model = LCModel("core", core_pars)

    # # # # # # # # # # # # # # # # #
    # # # Now do the band names # # #
    # # # # # # # # # # # # # # # # #

    # Collect the bands and their params. Add them total model.
    band_par_names = Band.node_par_names
    if not use_gp:
        # Use the Eclipse class to find the parameters we're interested in
        if is_complex:
            ecl_pars = ComplexEclipse.node_par_names
        else:
            ecl_pars = SimpleEclipse.node_par_names

    else:
        # Use the Eclipse class to find the parameters we're interested in
        if is_complex:
            ecl_pars = ComplexGPEclipse.node_par_names

        else:
            ecl_pars = SimpleGPEclipse.node_par_names

    # I care about the order in which eclipses and bands are defined.
    # Collect that order here.
    defined_bands = []
    defined_eclipses = []
    with open(input_file, "r") as input_file_obj:
        for line in input_file_obj:
            line = line.strip().split()

            if len(line):
                key = line[0]

                # Check that the key starts with any of the band pars
                if np.any([key.startswith(par) for par in band_par_names]):
                    # Strip off the variable part, and keep only the label
                    _, key = extract_par_and_key(key)
                    if key not in defined_bands:
                        defined_bands.append(key)

                # Check that the key starts with any of the eclipse pars
                if np.any([key.startswith(par) for par in ecl_pars]):
                    # Strip off the variable part, and keep only the label
                    _, key = extract_par_and_key(key)
                    if key not in defined_eclipses:
                        defined_eclipses.append(key)

    # Collect the band params into their Band objects.
    for label in defined_bands:
        band_pars = []

        # Build the Param objects for this band
        for par in band_par_names:
            # Construct the parameter key and retrieve the string
            key = "{}_{}".format(par, label)
            string = input_dict[key]

            # Make the Param object, and save it
            band_pars.append(Param.fromString(par, string))

        # Define the band as a child of the model.
        Band(label, band_pars, parent=model)

    # # # # # # # # # # # # # # # # #
    # # Finally, get the eclipses # #
    # # # # # # # # # # # # # # # # #

    lo = float(input_dict["phi_start"])
    hi = float(input_dict["phi_end"])

    for label in defined_eclipses[:neclipses]:
        # Get the list of parameters, and their priors
        params = []
        for par in ecl_pars:
            key = "{}_{}".format(par, label)
            param = Param.fromString(par, input_dict[key])

            params.append(param)

        if nodata:
            print("Using a roughly blank data dummy.")
            x = np.linspace(-0.5, 0.5, 1000)
            y = np.zeros_like(x)
            yerr = np.ones_like(y)

            lc = Lightcurve("Dummy_Data_{}".format(label), x, y, yerr)

        else:
            # Get the observational data
            lc_fname = input_dict["file_{}".format(label)]
            lc = Lightcurve.from_calib(lc_fname)
            lc.trim(lo, hi)

        # Get the band object that this eclipse belongs to
        my_band_label = input_dict["band_{}".format(label)]
        my_band = model.search_Node("Band", my_band_label)

        if use_gp:
            if is_complex:
                ComplexGPEclipse(lc, label, params, parent=my_band)
            else:
                SimpleGPEclipse(lc, label, params, parent=my_band)
        else:
            if is_complex:
                ComplexEclipse(lc, label, params, parent=my_band)
            else:
                SimpleEclipse(lc, label, params, parent=my_band)

    # Make sure that all the model's Band have eclipses. Otherwise, prune them
    model.children = [band for band in model.children if len(band.children)]

    return model
