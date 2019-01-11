import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
from datetime import datetime
import numpy as np
import sys
import os

def animate(i, fig, ax, file):
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

            ax.clear()
            ax.fill_between(range(N), meanChisq+stdChisq, meanChisq-stdChisq, color='green', alpha=0.3)
            ax.plot(range(N), meanChisq, color='red')

            # ax.set_title(file)
            # ax.set_ylabel('Ln(liklihood)')
            # ax.set_xlabel('Iteration')

            # Format the x-axis for dates (label formatting, rotation)
            fig.tight_layout()
        except:
            print("  Empty chain_prod file!", end='\r')
    else:
        print("  No file, '{}'!".format(file), end='\r')


args = sys.argv[1:]
try:
    file = args[0]
except:
    print("Please supply a file!")
    exit()

fig = plt.figure(figsize=[7,4])
ax = fig.add_subplot(111)
ax.set_title(file)
ax.set_ylabel('Ln(liklihood)')
ax.set_xlabel('Iteration')
plt.tight_layout()
ani = animation.FuncAnimation(fig, animate, fargs=(fig, ax, file), interval=10000)
plt.tight_layout()
plt.show()