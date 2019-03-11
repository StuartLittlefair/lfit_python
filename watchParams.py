import bokeh as bk
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Band
from bokeh.plotting import curdoc, figure
from bokeh.server.callbacks import NextTickCallback
from bokeh.models.widgets import inputs
from bokeh.models.widgets.buttons import Toggle
from bokeh.models.widgets import Slider

import numpy as np

import sys
import time


# function that reads in the next step
def readStep():
    global f

    stepData = np.zeros((nWalkers, len(pars)))
    flag = True # If true, return the data. If False, the end of the file was reached before the step was fully read in.
    init = f.tell()

    for i in np.arange(nWalkers):
        # Get the next line
        line = f.readline().strip()
        # Are we at the end of the file?
        if line == '':
            # The file is over.
            flag = False
            break

        line = np.array(line.split(), dtype=np.float64)

        # Check for infinities, replace with nans
        line[np.isinf(line)] = np.nan

        # Which walker are we?
        w = int(line[0])
        if w != i:
            print("Walker mismatch!")
            time.sleep(5)

        # Gather the desired numbers
        values = line[pars]
        
        stepData[w, :] = values

    if flag:
        return stepData
    else: 
        # The most recent step wasn't completely read in
        f.seek(init)
        
        for j in range(twait):
            print("\rWatching for file to updates{: <6} ".format(load[j%len(load)]), end='')
            time.sleep(0.1)
        print('\r                                   ', end='\r')
        return None

def update():
    if labels != []:
        step = readStep()

        if step is None:
            pass
        else:
            global s

            s += 1
            means = np.mean(step, axis=0)
            stds  = np.std(step,  axis=0)

            newdata = dict()
            newdata['step'] = [s]

            for i, label in enumerate(labels):
                newdata[label+'Mean'] = [means[i]]
                newdata[label+'StdUpper']  = [means[i]+stds[i]]
                newdata[label+'StdLower']  = [means[i]-stds[i]]
            
            source.stream(newdata, tail)

def update_ecl(attr, old, new):
    global necl
    global parNames

    val = eclipses.value
    complex = complex_button.active
    try:
        necl = int(val)
        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
            'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
            'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']

        # complex has extra parameters
        if complex:
            parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])

        # Format labels
        for i in range(necl-1):
            for name in parNameTemplate:
                parNames.append(name.format(i+1))

        parNames.append('Likelihood')
        selectList = list(parNames)
        selectList.insert(0, '')
        plotPars.options = selectList
    except:
        eclipses.value = ''

def update_complex(new):
    global necl
    global parNames

    val = eclipses.value
    try:
        necl = int(val)
    except:
        val = 1
    complex = complex_button.active

    
    parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
        'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
    parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
        'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']

    # complex has extra parameters
    # complex = 'n' #input("Complex bright spot? ")
    if complex:
        parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
        parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])

    # Format labels
    for i in range(necl-1):
        for name in parNameTemplate:
            parNames.append(name.format(i+1))

    parNames.append('Likelihood')
    selectList = list(parNames)
    selectList.insert(0, '')
    plotPars.options = selectList

def add_plot(attr, old, new):
    global labels
    global pars
    global source

    label = str(plotPars.value)
    if label == '':
        return
    par = parNames.index(label) + 1
    labels.append(label)
    pars.append(par)

    # Clear data from the source structure 
    source.data = {'step': []}
    for label in labels:
        source.data[label+'Mean']     = []
        source.data[label+'StdUpper'] = []
        source.data[label+'StdLower'] = []

    # Move the file cursor back to the beginning of the file
    global f
    global s
    f.close()
    f = open(file, 'r')
    s = 0

    new_plot = bk.plotting.figure(title=label, plot_height=300, plot_width=1200,
        toolbar_location='above', y_axis_location="right") 
        # tools="ypan,ywheel_zoom,ybox_zoom,reset")
    new_plot.line(x='step', y=label+'Mean', alpha=1, line_width=3, color='red', source=source)
    band = Band(base='step', lower=label+'StdLower', upper=label+'StdUpper', source=source, 
                level='underlay', fill_alpha=0.5, line_width=0, line_color='black', fill_color='green')
    new_plot.add_layout(band)

    new_plot.x_range.follow = "end"
    new_plot.x_range.follow_interval = tail
    new_plot.x_range.range_padding = 0
    new_plot.y_range.range_padding_units = 'percent'
    new_plot.y_range.range_padding = 0.2

    new_plot.y_range.range_padding_units = 'percent'
    new_plot.y_range.range_padding = 0.3

    curdoc().add_root(row(new_plot))


# Filename
file = 'chain_prod.txt'
# How many data do we want to follow with?
tail = 2500
# Initial values
necl = 1
parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
        'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']

# Lists of what parameters we want to plot
pars   = []
labels = []

# fun loading animation
load = ['.', '..', '...']
for i in range(5):
    load.extend(['... -', '... \\', '... |', '... /', '... -', '... \\', '... |', '... /', '... -'])
load.extend(['...', '..', '.'])
load.extend(['', '', '', '', '']*4)
twait = len(load)

# Open the file, and keep it open
while True:
    try:
        f = open(file, 'r')
        break
    except:
        for j in range(300):
            print("\rWaiting for file to be created{: <6} ".format(load[j%len(load)]), end='')
            time.sleep(0.1)
print("Opened file OK!                      ")

# Determine the number of walkers
nWalkers = 0
while True:
    line = f.readline()
    line = line.split()
    walker = int(line[0])
    if nWalkers == walker:
        nWalkers += 1
    else:
        break
print("The file has {} walkers.".format(nWalkers))

# Close and reopen the file to move the cursor back to the beginning.
f.close()
# We're at step 0 now
f = open(file, 'r')
s = 0

# Initialise data storage
source = ColumnDataSource(dict(step=[]))
for label in labels:
    source.add(data=[], name=label+'Mean')
    source.add(data=[], name=label+'StdUpper')
    source.add(data=[], name=label+'StdLower')


# Drop down box that should add extra parameters to the plot
selectList = list(parNames)
selectList.insert(0, '')
plotPars = inputs.Select(width=120, title='Optional Parameters', options=selectList, value='')
plotPars.on_change('value', add_plot)

# I want a toggle to enable or disable the extra params
complex_button = Toggle(label='Complex BS?', width=120)
complex_button.on_click(update_complex)

# Ask the user how many eclipses are in the data
eclipses = Slider(title='How many eclipses?', width=200, start=1, end=10, step=1, value=1)
eclipses.on_change('value', update_ecl)

# Add stuff to the visible area
layout = gridplot([eclipses, complex_button, plotPars], ncols=2)
curdoc().add_root(layout)
curdoc().title = 'MCMC Chain Supervisor'
curdoc().add_periodic_callback(update, 1)
