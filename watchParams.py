import matplotlib
from matplotlib import _pylab_helpers
from matplotlib.rcsetup import interactive_bk as _interactive_bk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation

from datetime import datetime
from datetime import timedelta

import numpy as np

import sys
import os
import time

### Currently, reading in the data takes a long time.
# Could open the file, store as a global variable, and every x seconds, read in from the last cursor
# position to the end of the file, and append that to a global data variable?

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
    watchParams <nWalkers> <chain file> [<column> <column label>]*N

    Plots the ln(like), and optionally N parameters, but labels are needed too.
    '''

    # Get the args passed via command line
    args = sys.argv[1:]
    # Check everything is fine
    try:
        nWalkers = int(args[0])
        file = args[1]
    except:
        print("Usage: watchParams <nWalkers> <chain file>")
        exit()

    pars   = []
    labels = []
    while True:
        cont = input("Plot parameter evolution [y/n]: ")
        cont = cont.lower() == 'y'

        if not cont:
            print("")
            break
        else:
            par = input("Which column in the chain file do you want to plot: ")
            label = input("What label should I apply to this: ")

            par = int(par)

            print("Plotting column {} with the label '{}'\n".format(par, label))

            pars.append(par)
            labels.append(label)

    twait = input("How long to wait between file reads (s): ")
    twait = float(twait)

    # Open the file, and keep it open
    while True:
        try:
            f = open(file, 'r')
            break
        except:
            for j in range(60):
                print(" Waiting for file to be created{: <4}".format('.'*(j%4)), end='\r')
                time.sleep(300)
    print("Opened file OK!                   ")

    # Ideally the code would figure this out
    # nWalkers = input("How many walkers are you using: ")
    nWalkers = int(nWalkers)

    # Interactivity on
    plt.ion()
    # Set up plotting area for log(like)
    likeFig = plt.figure(figsize=[7,4])
    likeAx = likeFig.add_subplot(111)
    likeAx.set_title(file)
    likeAx.set_ylabel('Ln(liklihood)')
    likeAx.set_xlabel('Iteration')

    # Plotting area for parameters to watch
    if pars != []:
        paramFig, paramAx = plt.subplots(len(pars), figsize=[7,4], sharex=True)
        # If we only asked for one parameter to be watched, we need to support indexing still
        if len(pars) == 1:
            paramAx = [paramAx]

        paramAx[0].set_title(file)
        for i, label in enumerate(labels):
            paramAx[i].set_ylabel(label)
        paramAx[-1].set_xlabel('Iteration')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    # Open the plotting window
    plt.show()

    # Tracker variables
    curline = 1
    step = 0
    pstep = 0

    # Limit the amount of memory we use at a time
    linelimit = int(300000/nWalkers) * nWalkers
    print("Limited to reading {} lines at a time ({} steps)".format(linelimit, linelimit/nWalkers))

    while True:
        print("  Reading file...")

        # Reset walkers, dims: (step, walker, parameters)
        walkers  = np.full((1000, nWalkers, len(pars)+1), np.nan, dtype=np.float64)
        # For each step, we have an entry for each walker, with slots for each parameter
        len_walkers = walkers.shape[0]

        # Read to the end of the file
        nlines = 0          # How many lines have we read on this pass?
        i = 0               # What index to we store the data to?
        flag = False

        # Read in the file until we hit the end. If a step wasn't fully read in, discard it.
        flag = True # If true at the end, wait a while before re-scanning the file. If false, immediately re-scan the file
        iloc = 0

        newline = True
        while newline:
            # Store a set of walkers' individual step
            temp = np.full((nWalkers, len(pars)+1), np.nan, dtype=np.float64)

            print("Step {:5}      ".format(step), end='\r')

            for j in range(nWalkers):
                # Get the next line
                newline = f.readline().strip()
                # Are we at the end of the file?
                if newline == '':
                    # The file is over.
                    print("Last line read in:\n{}".format(line))
                    flag = True
                    break
                else:
                    line = newline
                    nlines += 1
                    curline += 1

                    # Have we hit the memory limit?
                    if nlines == linelimit - 1:
                        print("Hit the line limit! Storing {} walkers".format(j+1))
                        flag = False
                        newline = False
                        break

                line = np.array(line.split(), dtype=np.float64)

                # Check for infinities:
                line[np.isinf(line)] = np.nan

                # Which walker are we?
                w = int(line[0])
                # What step are we up to now?
                step = np.ceil(curline / nWalkers) - 1
                step = int(step)

                # Gather the desired numbers
                lnlike = line[-1]
                parVals = []
                for par in pars:
                    parVals.append(line[par])
                values = [lnlike] + parVals
                values = np.array(values)


                temp[j, :] = values


            # Is our array going to be big enough?
            if iloc == len_walkers-1:
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

            walkers[iloc, :, :] = temp
            iloc += 1
            curstep = step


        print("Read in {} lines, or {:.2f} steps!".format(nlines, nlines/nWalkers))
        print("I want to plot from step {} to step {}".format(pstep, curstep))
        print("Slicing the first {} steps from the walkers".format(iloc))
        chisqs = walkers[:iloc,:,0]
        print(chisqs.shape)

        meanChisq = np.nanmean(chisqs, axis=1, dtype=np.float64)
        stdChisq  = np.nanstd(chisqs, axis=1, dtype=np.float64)

        N = meanChisq.shape[0]
        Xrange = np.arange(pstep, pstep+N)

        # Tighten the range
        likeAx.set_xlim(0, curstep)
        # Plot data
        likeAx.fill_between(Xrange,
                            meanChisq+stdChisq,
                            meanChisq-stdChisq,
                            color='green', alpha=0.3
                            )
        likeAx.plot(Xrange, meanChisq, color='red')

        for j, par in enumerate(pars):
            chisqs = walkers[:iloc, :, j+1]

            meanChisq = np.nanmean(chisqs, axis=1, dtype=np.float64)
            stdChisq  = np.nanstd(chisqs, axis=1, dtype=np.float64)

            paramAx[j].set_xlim(0, curstep)

            # Plot data
            paramAx[j].fill_between(Xrange,
                                meanChisq+stdChisq,
                                meanChisq-stdChisq,
                                color='green', alpha=0.3
                                )
            paramAx[j].plot(Xrange, meanChisq, color='red')


        likeFig.tight_layout()
        likeFig.canvas.draw_idle()
        if pars != []:
            paramFig.tight_layout()
            paramFig.canvas.draw_idle()

        pstep = curstep

        del walkers


        if flag:
            # Now we want to wait a while.
            now = datetime.now()
            t_next = now + timedelta(seconds=twait)
            print("Reading file next at {}".format(t_next.strftime("%H:%M:%S")))
            plt.pause(twait)
        else:
            plt.pause(0.1)

    f.close()
