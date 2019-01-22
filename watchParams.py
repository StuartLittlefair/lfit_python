import matplotlib
from matplotlib import _pylab_helpers
from matplotlib.rcsetup import interactive_bk as _interactive_bk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation

from datetime import datetime

import numpy as np

import sys
import os
import time

### Currently, reading in the data takes a long time.
# Could open the file, store as a global variable, and every x seconds, read in from the last cursor
# position to the end of the file, and append that to a global data variable?

def watch_params(i, paramFig, paramAx, file, pars):
    # Read data from 'file.dat'
    if os.path.isfile(file):
        try:
            data = np.genfromtxt(file, delimiter=' ', dtype=float)
            nWalkers = data[-1,0] + 1
            nWalkers = int(nWalkers)
            # print("{} Walkers".format(nWalkers))

            # The data has one walker per line. Collect them
            walkerChains = np.array([data[i::nWalkers] for i in range(nWalkers)])

            N = len(walkerChains[0])

            # chisqs = walkerChains[:,:,-1] * (-1)

            # meanChisq = np.mean(chisqs, paramAxis=0)
            # stdChisq  = np.std(chisqs, paramAxis=0)

            paramAx[0].set_title(file)
            paramAx[-1].set_xlabel('Iteration')

            for i, p in enumerate(pars):
                # Collect the parameter values
                param_values = walkerChains[:,:,p]
                # Make them appropriate for plotting
                mean_param   = np.mean(param_values, axis=0)
                std_param    = np.std(param_values, axis=0)

                # Clear previous lines, or we have a memory leak
                paramAx[i].clear()
                # I don't like having a gap before the line starts
                paramAx[i].set_xlim(0.0, N)

                # Fill standard deviation
                paramAx[i].fill_between(range(N),
                                        mean_param+std_param,
                                        mean_param-std_param,
                                        color='green', alpha=0.3
                                        )
                # Plot mean line
                paramAx[i].plot(range(N), mean_param, color='red')

            # Add back in the labels
            for i, label in enumerate(labels):
                paramAx[i].set_ylabel(label)

        except:
            pass


def pause(interval):
    """
    Pause for *interval* seconds.
    If there is an active figure it will be updated and displayed,
    and the GUI event loop will run during the pause.
    If there is no active figure, or if a non-interactive backend
    is in use, this executes time.sleep(interval).
    This can be used for crude animation. For more complex
    animation, see :mod:`matplotlib.animation`.
    This function is experimental; its behavior may be changed
    or extended in a future release.
    """
    backend = matplotlib.rcParams['backend']
    if backend in _interactive_bk:
        figManager = _pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            plt.show(block=False)
            canvas.start_event_loop(interval)
            return

    # No on-screen figure is active, so sleep() is all we need.
    time.sleep(interval)


if __name__ == "__main__":
    '''
    watchParams <chain file> [<column> <column label>]*N

    Plots the ln(like), and optionally N parameters, but labels are needed too.
    '''

    plt.ion()

    args = sys.argv[1:]
    try:
        nWalkers = int(args[0])
        file = args[1]
        pars = np.array(args[2:])
        labels = pars[::2]
        pars   = np.array(pars[1::2], dtype=int)
    except:
        print("Please supply a file!")
        exit()

    if pars != []:
        print("Plotting of individual parameters not yet supported :(")
        # print("Plotting the parameters in columns:")
        # print(pars)
        # print(labels)

        # paramFig, paramAx = plt.subplots(len(pars), figsize=[7,4], sharex=True)
        # paramAx[0].set_title(file)
        # for i, label in enumerate(labels):
        #     paramAx[i].set_ylabel(label)
        # paramAx[-1].set_xlabel('Iteration')



    likeFig = plt.figure(figsize=[7,4])
    likeAx = likeFig.add_subplot(111)
    likeAx.set_title(file)
    likeAx.set_ylabel('Ln(liklihood)')
    likeAx.set_xlabel('Iteration')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    # nWalkers = input("How many walkers are you using: ")
    nWalkers = int(nWalkers)

    plt.show()

    f = open(file, 'r')
    curline = 1
    step = 1
    pstep = 0
    linelimit = int(500000/nWalkers) * nWalkers
    print("Limited to reading {} lines at a time ({} steps)".format(linelimit, linelimit/nWalkers))

    while True:
        print("  Reading file...")

        # Reset walkers
        walkers  = np.full((100, nWalkers, len(pars)+1), np.nan, dtype=np.float64)
        # For each step, we have an entry for each walker, with slots for each parameter
        len_walkers = walkers.shape[0]

        # Read to the end of the file
        nlines = 1          # How many lines have we read on this pass?
        i = 0               # What index to we store the data to?
        first_step = 9e99   # What is the first step read in on this pass?
        flag = False

        line = f.readline()
        while line:
            # Information gathering. Where do I put the data?
            line = np.array(line.split(), dtype=np.float64)

            # Check for infinities:
            if np.any(np.isinf(line)):
                line[np.isinf(line)] = np.nan

            # Which walker are we?
            w = int(line[0])
            # What step are we up to now?
            step = np.ceil(curline / nWalkers)
            step = int(step)

            # Where to store the data in our array?
            if first_step == 9e99:
                first_step = step
            i = step - first_step

            # Is our array going to be big enough?
            if i == len_walkers-1:
                shape = list(walkers.shape)
                newShape = list(shape)
                newShape[0] += 1000
                # print("  Expanding array from {} to {}...".format(shape, newShape))

                # nans are ignored when we ask for the mean and standard deviation
                new_array = np.full(newShape, np.nan, dtype=np.float64)
                # Inject the old data into the new array
                new_array[0:len_walkers, :, :] = walkers

                # Copy the array into the old variable
                walkers = np.array(new_array)
                len_walkers = walkers.shape[0]
                # Free up memory
                del new_array

            # Gather the desired numbers
            lnlike = line[-1]
            parVals = []
            for par in pars:
                parVals.append(line[par])
            values = [lnlike] + parVals
            values = np.array(values)

            # Store
            # print("Storing {}".format(i))
            walkers[i, w, :] = values

            if nlines == linelimit:
                flag = True
                break

            line = f.readline()
            curline += 1
            nlines  += 1

        curstep = step

        print("Read in {} lines, or {:.2f} steps!".format(nlines, nlines/nWalkers))
        print("I want to plot from step {} to step {}".format(pstep, curstep))
        print("Slicing the first {} steps from the walkers".format(i-1))
        chisqs = walkers[:i,:,0]
        print(chisqs.shape)

        meanChisq = np.nanmean(chisqs, axis=1, dtype=np.float64)
        stdChisq  = np.nanstd(chisqs, axis=1, dtype=np.float64)

        N = meanChisq.shape[0]
        Xrange = np.arange(first_step, first_step+N)

        # Tighten the range
        likeAx.set_xlim(0, curstep)
        # Plot data
        likeAx.fill_between(Xrange,
                            meanChisq+stdChisq,
                            meanChisq-stdChisq,
                            color='green', alpha=0.3
                            )
        likeAx.plot(Xrange, meanChisq, color='red')

        likeFig.tight_layout()
        likeFig.canvas.draw_idle()

        pstep = curstep

        del walkers

        if not flag:
            plt.pause(120)
        else:
            plt.pause(10)

    f.close()
