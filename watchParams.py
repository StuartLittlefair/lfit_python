import bokeh as bk
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Band
from bokeh.plotting import curdoc, figure
from bokeh.server.callbacks import NextTickCallback
from bokeh.models.widgets import inputs
from bokeh.models.widgets.buttons import Toggle
from bokeh.models.widgets import Slider

import numpy as np

import time


class Watcher():
    def __init__(self, file='chain_prod.txt', tail=5000, thin=0):
        # Filename
        self.file = file
        # How many data do we want to follow with?
        self.tail = tail
        # How many data do we want to skip?
        self.thin = thin
        
        # Keep track of how many we've skipped so far
        self.thinstep = 0

        # Initial values
        self.necl = 1        # Number of eclipses
        self.s    = 0        # Number of steps read in so far
        self.f    = False    # File object, initially false so we can wait for it to be created
        self.nWalkers = 0    # Number of walkers

        # Lists of what parameters we want to plot
        self.pars   = []     # List of params
        self.labels = []     # The labels, in the same order as pars
        self.parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
                'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']

        # fun loading animation
        load = ['.', '..', '...']
        for i in range(5):
            load.extend(['... -', '... \\', '... |', '... /', '... -', '... \\', '... |', '... /', '... -'])
        load.extend(['...', '..', '.'])
        load.extend(['', '', '', '', '']*4)
        self.load = load
        self.twait = len(load)

        # Initialise data storage
        source = ColumnDataSource(dict(step=[]))
        for label in self.labels:
            source.add(data=[], name=label+'Mean')
            source.add(data=[], name=label+'StdUpper')
            source.add(data=[], name=label+'StdLower')
        self.source = source


        # Drop down box that should add extra parameters to the plot
        self.selectList = ['']
        self.plotPars = inputs.Select(width=120, title='Optional Parameters', options=self.selectList, value='')
        self.plotPars.on_change('value', self.add_plot)

        # I want a toggle to enable or disable the extra params
        self.complex_button = Toggle(label='Complex BS?', width=120, button_type='success')
        self.complex_button.on_click(self.update_complex)

        # Ask the user how many eclipses are in the data
        self.eclipses = Slider(title='How many eclipses?', width=200, start=0, end=10, step=1, value=0)
        self.eclipses.on_change('value', self.update_ecl)

        # Add stuff to the visible area
        self.layout = gridplot([self.plotPars, self.eclipses, self.complex_button], ncols=1)
        curdoc().add_root(self.layout)
        curdoc().title = 'MCMC Chain Supervisor'
        self.check_file = curdoc().add_periodic_callback(self.open_file, 500)

    def open_file(self):
        '''Check if the chain file has been created yet. If not, do nothing. If it is, set it to self.f'''
        # Open the file, and keep it open
        file = self.file
        try:
            self.f = open(file, 'r')
        except:
            return

        # Determine the number of walkers
        while True:
            line = self.f.readline()
            line = line.split()
            walker = int(line[0])
            if self.nWalkers == walker:
                self.nWalkers += 1
            else:
                break

        # Close and reopen the file to move the cursor back to the beginning.
        self.f.close()
        # We're at step 0 now
        self.f = open(file, 'r')

        # Remove the callback that keeps trying to open the file
        curdoc().remove_periodic_callback(self.check_file)
        # Create a new callback that periodically reads the file
        curdoc().add_periodic_callback(self.update, 0.001)

        print("Opened the file {}!".format(file))

    def readStep(self):
        '''Appempt to read in the next step of the chain file. 
        If we get an unexpected number of walkers, quit the script. 
        If we're at the end of the file, do nothing.'''
        
        stepData = np.zeros((self.nWalkers, len(self.pars)))
        flag = True # If true, return the data. If False, the end of the file was reached before the step was fully read in.
        init = self.f.tell()

        try:
            for i in np.arange(self.nWalkers):
                # Get the next line
                line = self.f.readline().strip()
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
                    exit()

                # Gather the desired numbers
                values = line[self.pars]

                stepData[w, :] = values
        except:
            flag = False
        
        self.thinstep += 1
        if self.thin:
            if self.thinstep % self.thin != 0:
                flag = None


        if flag is True:
            # We successfully read in the chunk!
            self.s += 1
            return stepData
        elif flag is False:
            # The most recent step wasn't completely read in
            self.f.seek(init)
            return None
        else:
            # We read in a step but we don't want it.
            self.s += 1
            return None

    def update(self):
        '''Call the readStep() function, and stream the data to the plotter.'''
        # Do we have anything to plot?
        if self.labels != []:
            step = self.readStep()

            if step is None:
                # No data to plot
                pass
            else:
                means = np.mean(step, axis=0)
                stds  = np.std(step,  axis=0)

                newdata = dict()
                newdata['step'] = [self.s]

                for i, label in enumerate(self.labels):
                    newdata[label+'Mean'] = [means[i]]
                    newdata[label+'StdUpper']  = [means[i]+stds[i]]
                    newdata[label+'StdLower']  = [means[i]-stds[i]]

                self.source.stream(newdata, self.tail)

    def update_ecl(self, attr, old, new):
        '''Update the variables that depend on the number of eclipses we have'''
        val = self.eclipses.value
        complex = self.complex_button.active
        try:
            self.necl = int(val)
            parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
                'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
            parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
                'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']

            # complex has extra parameters
            if complex:
                parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
                parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])

            # Format labels
            for i in range(self.necl-1):
                for name in parNameTemplate:
                    parNames.append(name.format(i+1))
            parNames.append('Likelihood')

            self.parNames = parNames
            self.selectList = list(parNames)
            self.selectList.insert(0, '')
            self.plotPars.options = self.selectList
        except:
            self.eclipses.value = 0

    def update_complex(self, new):
        '''Identical functionality to update_ecl(), but the handler for toggle buttons works differently.'''
        val = self.eclipses.value
        try:
            self.necl = int(val)
        except:
            val = 0
        complex = self.complex_button.active

        if complex:
            self.complex_button.button_type = 'success'
        else:
            self.complex_button.button_type = 'danger'

        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
            'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
            'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']

        # complex has extra parameters
        if complex:
            parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])

        # Format labels
        for i in range(self.necl-1):
            for name in parNameTemplate:
                parNames.append(name.format(i+1))
        parNames.append('Likelihood')
        
        self.parNames = parNames
        self.selectList = list(parNames)
        self.selectList.insert(0, '')
        self.plotPars.options = self.selectList

    def add_plot(self, attr, old, new):
        '''Add a plot to the page'''
        if self.f is False:
            return

        label = str(self.plotPars.value)
        if label == '':
            return
        par = self.parNames.index(label) + 1
        self.labels.append(label)
        self.pars.append(par)

        # Clear data from the source structure
        self.source.data = {'step': []}
        for label in self.labels:
            self.source.data[label+'Mean']     = []
            self.source.data[label+'StdUpper'] = []
            self.source.data[label+'StdLower'] = []

        # Move the file cursor back to the beginning of the file
        self.f.close()
        self.f = open(self.file, 'r')
        self.s = 0

        new_plot = bk.plotting.figure(title=label, plot_height=300, plot_width=1200,
            toolbar_location='above', y_axis_location="right")
            # tools="ypan,ywheel_zoom,ybox_zoom,reset")
        new_plot.line(x='step', y=label+'Mean', alpha=1, line_width=3, color='red', source=self.source)
        band = Band(base='step', lower=label+'StdLower', upper=label+'StdUpper', source=self.source,
                    level='underlay', fill_alpha=0.5, line_width=0, line_color='black', fill_color='green')
        new_plot.add_layout(band)

        new_plot.x_range.follow = "end"
        new_plot.x_range.follow_interval = self.tail
        new_plot.x_range.range_padding = 0
        new_plot.y_range.range_padding_units = 'percent'
        new_plot.y_range.range_padding = 0.3

        curdoc().add_root(row(new_plot))

if __name__ in '__main__':
    # import argparse

    # parser = argparse.ArgumentParser(description='Monitor an MCMC chain as it runs')
    # parser.add_argument('file', default='chain_prod.txt', type=str, nargs=1,
    #                     help='The outputted chain file to monitor')
    # parser.add_argument('thin', default=0, type=int, nargs=1,
    #                     help='I only read in every [thin] steps. 0 will read every step.',
    #                     )
    # parser.add_argument('tail', default=1000, type=int, nargs=1,
    #                     help='I will plot only the last [tail] steps',
    #                     )

    # args = parser.parse_args()
    # fname = args.file[0]
    # tail  = args.tail[0]
    # thin  = args.thin[0]

    fname = 'chain_prod.txt'
    tail = 2000
    thin = 10

    watcher = Watcher(file=fname, tail=tail, thin=thin)