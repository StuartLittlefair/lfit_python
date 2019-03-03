from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter

import numpy as np
import mcmc_utils

try:
    import triangle
    # This triangle should have a method corner
    # There are two python packages with conflicting names
    getattr(triangle, "corner")
except (AttributeError, ImportError):
    # We want the other package
    import corner as triangle

# def triangle.corner(toPlot, labels=None, bins=20):

#     # Try filling in labels from pandas.DataFrame columns.
#     if labels is None:
#         try:
#             labels = toPlot.columns
#         except AttributeError:
#             pass

#     # Parse the parameter ranges.
#     range = [[x.min(), x.max()] for x in toPlot]
#     # Check for parameters that never change.
#     m = np.array([e[0] == e[1] for e in range], dtype=bool)
#     if np.any(m):
#         raise ValueError(("It looks like the parameter(s) in "
#                             "column(s) {0} have no dynamic range. ")
#                             .format(", ".join(map(
#                                 "{0}".format, np.arange(len(m))[m]))))

#     # Some magic numbers for pretty axis layout.
#     K = len(toPlot)
#     factor = 2.0           # size of one side of one panel
#     lbdim = 0.5 * factor   # size of left/bottom margin
#     trdim = 0.2 * factor   # size of top/right margin
#     whspace = 0.05         # w/hspace size
#     plotdim = factor * K + factor * (K - 1.) * whspace
#     dim = lbdim + plotdim + trdim

#     # Create a new figure
#     fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    
#     # Format the figure.
#     lb = lbdim / dim
#     tr = (lbdim + plotdim) / dim
#     fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
#                         wspace=whspace, hspace=whspace)

#     # Set up the default histogram keywords.
#     hist_kwargs = dict()
#     hist_kwargs["color"] = hist_kwargs.get("color", 'k')
#     hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")

#     for i, x in enumerate(toPlot):
#         # Deal with masked arrays.
#         if hasattr(x, "compressed"):
#             x = x.compressed()

#         if np.shape(toPlot)[0] == 1:
#             ax = axes
#         else:
#             ax = axes[i, i]
#         # Plot the histograms.
#         n, _, _ = ax.hist(x, bins=bins[i])
        
#         title = None
#         if labels is not None:
#             title = "{0}".format(labels[i])

#         if title is not None:
#             ax.set_title(title)

#         # Set up the axes.
#         ax.set_xlim(range[i])
#         ax.set_ylim(0, 1.1 * np.max(n))
#         ax.set_yticklabels([])
#         ax.xaxis.set_major_locator(MaxNLocator(5, prune="lower"))

#         if i < K - 1:
#             ax.set_xticklabels([])
#         else:
#             [l.set_rotation(45) for l in ax.get_xticklabels()]
#             if labels is not None:
#                 ax.set_xlabel(labels[i])
#                 ax.xaxis.set_label_coords(0.5, -0.3)

#             # use MathText for axes ticks
#             ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))

#         for j, y in enumerate(toPlot):
#             if np.shape(toPlot)[0] == 1:
#                 ax = axes
#             else:
#                 ax = axes[i, j]
#             if j > i:
#                 ax.set_frame_on(False)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 continue
#             elif j == i:
#                 continue

#             # Deal with masked arrays.
#             if hasattr(y, "compressed"):
#                 y = y.compressed()

#             hist2d(y, x, ax=ax, range=[range[j], range[i]], bins=[bins[j], bins[i]])

            
#             ax.xaxis.set_major_locator(MaxNLocator(5, prune="lower"))
#             ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))

#             if i < K - 1:
#                 ax.set_xticklabels([])
#             else:
#                 [l.set_rotation(45) for l in ax.get_xticklabels()]
#                 if labels is not None:
#                     ax.set_xlabel(labels[j])
#                     ax.xaxis.set_label_coords(0.5, -0.3)

#                 # use MathText for axes ticks
#                 ax.xaxis.set_major_formatter(
#                     ScalarFormatter(useMathText=False))

#             if j > 0:
#                 ax.set_yticklabels([])
#             else:
#                 [l.set_rotation(45) for l in ax.get_yticklabels()]
#                 if labels is not None:
#                     ax.set_ylabel(labels[i])
#                     ax.yaxis.set_label_coords(-0.3, 0.5)

#                 # use MathText for axes ticks
#                 ax.yaxis.set_major_formatter(
#                     ScalarFormatter(useMathText=False))

#     return fig



fname = 'chain_prod.txt'

print('Reading chain file...')
chain = mcmc_utils.readchain(fname)
print("Done!")

# The chain is 3d, (nwalkers, nsteps, ndim). Convert to just (nsamples, ndim)
flat = mcmc_utils.flatchain(chain, 49)
print("Flat chain has the shape {}".format(flat.shape))

# Column labels
parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',
        'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 
        'phi0_{0}', 'exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}',
        'q', 'dphi', 'rwd']




# Extract the desired columns.
var = [0,1,2,3,6,7,9,10,11,12,13,14,15,16,17, 4, 5, 8]
var = np.array(var)
var = np.array([4, 5, 8])

toPlot = flat[2250000::100,var]
print("Extracted the desired columns...")

print("Constructing corner plot...")
oname = 'eclipse_1.png'
parNames = [template.format('0') for template in parNameTemplate]
fig = triangle.corner(toPlot, bins=50, labels=['q', 'dphi', 'rwd'])
plt.savefig(oname)
print("Saved {}".format(oname))
plt.close()
del toPlot




var = list(range(18, 33))
var += [4, 5, 8]
var = np.array(var)
toPlot = flat[2250000::100,var]
print("Extracted the desired columns...")

print("Constructing corner plot...")
oname = 'eclipse_2.png'
parNames = [template.format('1') for template in parNameTemplate]
fig = triangle.corner(toPlot, bins=50, labels=parNames)
plt.savefig(oname)
print("Saved {}".format(oname))
plt.close()
del toPlot




var = list(range(33, 48))
var += [4, 5, 8]
var = np.array(var)
toPlot = flat[2250000::100,var]
print("Extracted the desired columns...")

print("Constructing corner plot...")
oname = 'eclipse_3.png'
parNames = [template.format('2') for template in parNameTemplate]
fig = triangle.corner(toPlot, bins=50, labels=parNames)
plt.savefig(oname)
print("Saved {}".format(oname))
plt.close()
del toPlot
