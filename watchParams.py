import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
from datetime import datetime
import numpy as np
import sys
import os

def watch_lnlike(i, likeFig, likeAx, file):
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
            print("  {} Walkers, Chain is {} iterations long".format(nWalkers, N), end='\r')

            chisqs = walkerChains[:,:,-1] * (-1)

            meanChisq = np.mean(chisqs, axis=0)
            stdChisq  = np.std(chisqs, axis=0)

            likeAx.clear()
            # Add back in the labels
            likeAx.set_title(file)
            likeAx.set_ylabel('Ln(liklihood)')
            likeAx.set_xlabel('Iteration')
            # Tighten the range
            likeAx.set_xlim(0, N)
            # Plot data
            likeAx.fill_between(range(N),
                                meanChisq+stdChisq,
                                meanChisq-stdChisq,
                                color='green', alpha=0.3
                                )
            likeAx.plot(range(N), meanChisq, color='red')

            likeFig.tight_layout()
        except:
            pass

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


args = sys.argv[1:]
try:
    file = args[0]
    pars = np.array(args[1:])
    labels = pars[::2]
    pars   = np.array(pars[1::2], dtype=int)
except:
    print("Please supply a file!")
    exit()

if pars != []:
    print("Plotting the parameters in columns:")
    print(pars)
    print(labels)

    paramFig, paramAx = plt.subplots(len(pars), figsize=[7,4], sharex=True)
    paramAx[0].set_title(file)
    for i, label in enumerate(labels):
        paramAx[i].set_ylabel(label)
    paramAx[-1].set_xlabel('Iteration')
    paramAni = animation.FuncAnimation(
        paramFig, watch_params,
        fargs=(paramFig, paramAx, file, pars),
        interval=10000
        )

likeFig = plt.figure(figsize=[7,4])
likeAx = likeFig.add_subplot(111)
likeAx.set_title(file)
likeAx.set_ylabel('Ln(liklihood)')
likeAx.set_xlabel('Iteration')
likeAni = animation.FuncAnimation(
    likeFig, watch_lnlike,
    fargs=(likeFig, likeAx, file),
    interval=10000
    )

plt.tight_layout()
plt.subplots_adjust(hspace=0.0)
plt.show()