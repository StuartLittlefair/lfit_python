import bokeh as bk
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Band, Whisker
from bokeh.models.annotations import Title
from bokeh.plotting import curdoc, figure
from bokeh.server.callbacks import NextTickCallback
from bokeh.models.widgets import inputs
from bokeh.models.widgets.buttons import Toggle
from bokeh.models.widgets import Slider, Panel, Tabs, Dropdown

import numpy as np
from pandas import read_csv
import configobj
import time

try:
    from lfit import CV
    print("Successfully imported CV class from lfit!")
    lc = True
except:
    print("Failed to import lfit! Plotting lightcurves not supported!")
    lc = False


def parseInput(file):
    """Splits input file up making it easier to read"""
    input_dict = configobj.ConfigObj(file)
    return input_dict

class Watcher():
    '''This class will initialise a bokeh page, with some useful lfit MCMC chain supervision tools.
        - Ability to plot a live chain file's evolution over time
        - Interactive lightcurve model, with input sliders or the ability to grab the last step's mean
    '''

    def __init__(self, chain='chain_prod.txt', mcmc_input='mcmc_input.dat', tail=5000, thin=0, lightcurve=True):
        '''
        '''
        # Can we preview the fits?
        self.isTabbed = lightcurve

        # Filename
        self.chain_file = chain
        # Grab mcmc_input
        try:
            self.mcmc_input_dict = parseInput(mcmc_input)
        except FileNotFoundError:
            print("Couldn't find MCMC input file, {}".format(mcmc_input))
            self.mcmc_input_dict = None

        # How many data do we want to follow with?
        self.tail = tail
        # How many data do we want to skip?
        self.thin = thin

        # Keep track of how many we've skipped so far
        self.thinstep = 0

        # Initial values
        self.necl = int(self.mcmc_input_dict['neclipses'])       # Number of eclipses
        self.s    = 0                                       # Number of steps read in so far
        self.f    = False                                   # File object, initially false so we can wait for it to be created
        self.nWalkers = int(self.mcmc_input_dict['nwalkers'])    # Number of walkers

        # Lists of what parameters we want to plot
        self.pars   = []     # List of params
        self.labels = []     # The labels, in the same order as pars
        self.parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
                'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
                'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']


        # Here, read the parameters all into a dict of {key: [value, lowerlim, upperlim]}
        parNames = self.parNames
        for i in range(self.necl):
            parNames.extend([template.format(i) for template in parNameTemplate])

        self.parDict = {}
        for param in parNames:
            line = self.mcmc_input_dict[param].strip().split(' ')
            line = [x for x in line if x != '']
            line = [line[0], line[2], line[3]]

            parameter = [float(x) for x in line]
            self.parDict[param] = list(parameter)


        # fun loading animation
        load = ['.', '..', '...']
        for _ in range(5):
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

        ## First tab - Parameter History:
        # Drop down box that should add extra parameters to the plot
        self.selectList = ['']
        self.plotPars = inputs.Select(width=120, title='Optional Parameters', options=self.selectList, value='')
        self.plotPars.on_change('value', self.add_plot)
        # I want a toggle to enable or disable the extra params
        self.complex_button = Toggle(label='Complex BS?', width=120, button_type='danger')
        self.complex_button.on_click(self.update_complex)
        # Ask the user how many eclipses are in the data
        self.eclipses = Slider(title='How many eclipses?', width=200, start=0, end=10, step=1)
        self.eclipses.on_change('value', self.update_necl)
        self.eclipses.value = int(self.mcmc_input_dict['neclipses'])

        # This needs to be done after the eclipses slider has been created
        self.complex_button.active = bool(int(self.mcmc_input_dict['complex']))

        # Add stuff to a layout for the area
        self.layout = gridplot([self.eclipses, self.complex_button, self.plotPars], ncols=1)


        if self.isTabbed:
            # Add that layout to a tab
            self.tab1 = Panel(child=self.layout, title="Parameter History")


            ## Second tab - Plot the latest lightcurve
            # I need a myriad of sliders. The ranges on these should be set by the priors, or maybe I could have a series of +/- buttons?
            slider_wdFlux  = Slider(title='White Dwarf Flux',      width=200, start=0.001, end=0.15,   step=0.01,     value=0.025108  )
            slider_dFlux   = Slider(title='Disc Flux',             width=200, start=0.001, end=0.15,   step=0.01,     value=0.059197  )
            slider_sFlux   = Slider(title='BrightSpot Flux',       width=200, start=0.001, end=0.15,   step=0.01,     value=0.036787  )
            slider_rsFlux  = Slider(title='Secondary Flux',        width=200, start=0.001, end=0.15,   step=0.01,     value=0.004832  )
            slider_q       = Slider(title='Mass Ratio',            width=200, start=0.001, end=0.5,    step=0.01,     value=0.111144  )
            slider_dphi    = Slider(title='dPhi',                  width=200, start=0.030, end=0.1,    step=0.01,     value=0.065319  )
            slider_rdisc   = Slider(title='Disc Radius',           width=200, start=0.250, end=0.7,    step=0.01,     value=0.454316  )
            slider_ulimb   = Slider(title='Limb darkening',        width=200, start=0.1,   end=0.4,    step=0.01,     value=0.284287  )
            slider_rwd     = Slider(title='White Dwarf Radius',    width=200, start=0.001, end=0.1,    step=0.01,     value=0.026284  )
            slider_scale   = Slider(title='Bright Spot Scale',     width=200, start=0.001, end=0.2,    step=0.01,     value=0.021121  )
            slider_az      = Slider(title='Bright Spot Azimuth',   width=200, start=50,    end=180,    step=1,        value=95.11449  )
            slider_fis     = Slider(title='Isotrpoic BS fraction', width=200, start=0.001, end=1.0,    step=0.01,     value=0.008963  )
            slider_dexp    = Slider(title='Disc profile exponent', width=200, start=0.001, end=3,      step=0.01,     value=1.707775  )
            slider_phi0    = Slider(title='Time offset',           width=200, start=-0.1,  end=0.1,    step=0.01,     value=0.0       )

            # Create a gridplot of the sliders, in a long column
            self.par_sliders = [
                slider_wdFlux,
                slider_dFlux,
                slider_sFlux,
                slider_rsFlux,
                slider_q,
                slider_dphi,
                slider_rdisc,
                slider_ulimb,
                slider_rwd,
                slider_scale,
                slider_az,
                slider_fis,
                slider_dexp,
                slider_phi0,
            ]

            # If we're complex, add the extra sliders
            if self.complex_button.active:
                slider_exp1    = Slider(title='Complex BS exp1',   width=200, start=-10,   end=-2,     step=0.1,      value=-2.0)
                slider_exp2    = Slider(title='Complex BS exp2',   width=200, start=-10,   end=-2,     step=0.1,      value=-2.0)
                slider_tilt    = Slider(title='Complex BS tilt',   width=200, start=-90,   end=90,     step=1,        value=-2.0)
                slider_yaw     = Slider(title='Complex BS yaw' ,   width=200, start=-90,   end=90,     step=1,        value=-2.0)

                self.par_sliders.extend([
                    slider_exp1,
                    slider_exp2,
                    slider_tilt,
                    slider_yaw
                ])

            # These can be bulk-edited like this now:
            for slider in self.par_sliders:
                slider.on_change('value', self.update_lc_model)


            # Plot the actual lightcurve
            # self.data_fname = inputs.TextInput(title='Filename', value=self.mcmc_input_dict['file_0'])
            print("Grabbing data files from the input dict:")
            menu = []
            for i in range(int(self.mcmc_input_dict['neclipses'])):
                fname = self.mcmc_input_dict["file_{}".format(i)]
                print(fname)
                menu.append((fname.split('/')[-1], fname))
            self.data_fname = Dropdown(label="Filename", button_type="danger", menu=menu)
            self.data_fname.on_change('value', self.update_lc_obs)

            # Grab the data from the file
            self.lc_obs = read_csv(menu[0][1],
                    sep=' ', comment='#', header=None, names=['phase', 'flux', 'err'])
            # Compute errors
            # self.lc_obs['upper'] = self.lc_obs['flux'] + self.lc_obs['err']
            # self.lc_obs['lower'] = self.lc_obs['flux'] - self.lc_obs['err']

            # Generate the model lightcurve
            pars = [slider.value for slider in self.par_sliders]
            self.cv = CV(pars)
            self.lc_obs['calc'] = self.cv.calcFlux(pars, np.array(self.lc_obs['phase']))
            # Whisker can only take the ColumnDataSource, not the pandas array
            self.lc_obs = ColumnDataSource(self.lc_obs)
            # I want a button that'll turn red when the parameters are invalid
            self.lc_isvalid = Toggle(label='Valid parameters', width=200, button_type='success')

            # Initialise the figure
            self.lc_plot = bk.plotting.figure(title='Lightcurve', plot_height=500, plot_width=1200,
                toolbar_location='above', y_axis_location="left")
            # Plot the lightcurve data
            self.lc_plot.scatter(x='phase', y='flux', source=self.lc_obs, size=5, color='black')

            # # Plot the error bars - Bokeh doesnt have a built in errorbar!?!
            # self.lc_plot.add_layout(
            #     Whisker(base='phase', upper='upper', lower='lower', source=self.lc_obs,
            #     upper_head=None, lower_head=None, line_color='black', )
            # )

            # Plot the model
            self.lc_plot.line(x='phase', y='calc', source=self.lc_obs, line_color='red')


            # Add all this to a tab
            self.layout2 = column([
                    self.data_fname,
                    gridplot(self.par_sliders+[self.lc_isvalid], ncols=4),
                    self.lc_plot
                ])
            self.tab2 = Panel(child=self.layout2, title="Lightcurve Parameters")

            # Add the tabs to the figure
            self.tabs = Tabs(tabs=[self.tab1, self.tab2])
            curdoc().add_root(self.tabs)

        else:
            # We can only use the first tab, so dont bother with the others.
            curdoc().add_root(self.layout)

        curdoc().title = 'MCMC Chain Supervisor'
        try:
            curdoc().theme = 'dark_minimal'
        except:
            pass

        # Is the file open?
        self.open_file()
        # If it doesn't exist, self.f is false. Then, check every second for it again.
        if self.f is False:
            self.check_file = curdoc().add_periodic_callback(self.open_file, 1000)

    def open_file(self):
        '''Check if the chain file has been created yet. If not, do nothing. If it is, set it to self.f'''
        # Open the file, and keep it open
        file = self.chain_file
        try:
            self.f = open(file, 'r')
        except:
            self.f = False
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
        try:
            curdoc().remove_periodic_callback(self.check_file)
        except:
            pass
        # Create a new callback that periodically reads the file
        curdoc().add_periodic_callback(self.update, 1)

        print("Opened the file {}!".format(file))

    def readStep(self):
        '''Appempt to read in the next step of the chain file.
        If we get an unexpected number of walkers, quit the script.
        If we're at the end of the file, do nothing.'''

        stepData = np.zeros((self.nWalkers, len(self.pars)), dtype=float)
        # If true, return the data. If False, the end of the file was reached before the step was fully read in.
        flag = True

        # Remember where we started
        init = self.f.tell()

        ## This could? be:
        # stepData = numpy.loadtxt(self.f,           # File to read
        #   dtype=np.float,                          # All floats
        #   delimiter=' ',                           # Space separated
        #   usecols=pars,                            # Desired columns
        #   skiprows=(self.nWalkers*self.thin),      # Thin the data by self.thin steps
        #   max_rows=self.nWalkers                   # Only read in one step
        # )
        # followed by some checking routine that makes sure the array is fully populated. if not,
        # rewind.
        try:
            for i in np.arange(self.nWalkers): ## very slow!
                # Get the next line
                line = self.f.readline().strip()
                # Are we at the end of the file?
                if line == '':
                    # The file is over.
                    flag = False
                    break

                line = np.array(line.split(), dtype=np.float64)

                # Check for infinities, replace with nans. Handles bad walkers
                line[np.isinf(line)] = np.nan

                # Which walker are we?
                w = int(line[0])
                if w != i:
                    flag = False
                    break

                # Gather the desired numbers
                values = line[self.pars]

                stepData[w, :] = values
        except IndexError:
            # Sometimes empty lines slip through. Catch the exceptions
            print(line)
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

    def update_necl(self, attr, old, new):
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

            self.selectList = list(parNames)
            self.selectList.insert(0, '')
            self.plotPars.options = self.selectList
        except:
            self.eclipses.value = 0

    def update_complex(self, new):
        '''Identical functionality to update_necl(), but the handler for toggle buttons works differently.'''
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
        self.selectList = list(parNames)
        self.selectList.insert(0, '')
        self.plotPars.options = self.selectList

    def update_lc_obs(self, attr, old, new):
        '''redraw the model and observations for the lightcurve'''
        try:
            # Re-read the observations
            fname = self.data_fname.value
            fname = str(fname)

            print("Moving to file {}".format(fname))
            new_obs = read_csv(fname,
                    sep=' ', comment='#', header=None, names=['phase', 'flux', 'err'])
            # new_obs['upper'] = new_obs['flux'] + new_obs['err']
            # new_obs['lower'] = new_obs['flux'] - new_obs['err']

            fname = fname.split('/')[-1]
            self.lc_plot.title = Title(text=fname)

            # Regenerate the model lightcurve
            pars = [slider.value for slider in self.par_sliders]
            # try:
            self.cv = CV(pars)
            new_obs['calc'] = self.cv.calcFlux(pars, np.array(new_obs['phase']))

            # Push that into the data frame
            toRemove = len(new_obs['phase'])
            self.lc_obs.stream(new_obs, toRemove)
            # TODO: This does not 'un-draw' old errorbars, but leaves them as an artifact on the figure. How can I even fix this?

            self.lc_isvalid.button_type = 'success'
            self.lc_isvalid.label = 'Valid Parameters'

        except FileExistsError:
            self.lc_isvalid.button_type = 'danger'
            self.lc_isvalid.label = 'Invalid Parameters'

    def update_lc_model(self, attr, old, new):
        '''Redraw the model lightcurve in the second tab'''
        try:
            # Regenerate the model lightcurve
            pars = [slider.value for slider in self.par_sliders]

            self.cv = CV(pars)
            self.lc_obs.data['calc'] = self.cv.calcFlux(pars, np.array(self.lc_obs.data['phase']))
            self.lc_isvalid.button_type = 'success'
            self.lc_isvalid.label = 'Valid Parameters'

        except Exception:
            self.lc_isvalid.button_type = 'danger'
            self.lc_isvalid.label = 'Invalid Parameters'

    def add_plot(self, attr, old, new):
        '''Add a plot to the page'''

        label = str(self.plotPars.value)
        if label == '':
            return
        par = self.parNames.index(label) + 1
        self.labels.append(label)
        self.pars.append(par)

        # Clear data from the source structure
        self.source.data = {'step': []}
        for l in self.labels:
            self.source.data[l+'Mean']     = []
            self.source.data[l+'StdUpper'] = []
            self.source.data[l+'StdLower'] = []

        # Move the file cursor back to the beginning of the file
        if not self.f is False:
            self.f.close()
            self.f = open(self.chain_file, 'r')
            self.s = 0

        new_plot = bk.plotting.figure(title=label, plot_height=300, plot_width=1200,
            toolbar_location='above', y_axis_location="right",
            # tools="ypan,ywheel_zoom,ybox_zoom,reset")
            tools=[])
        new_plot.line(x='step', y=label+'Mean', alpha=1, line_width=3, color='red', source=self.source)
        band = Band(base='step', lower=label+'StdLower', upper=label+'StdUpper', source=self.source,
                    level='underlay', fill_alpha=0.5, line_width=0, line_color='black', fill_color='green')
        new_plot.add_layout(band)

        new_plot.x_range.follow = "end"
        new_plot.x_range.follow_interval = self.tail
        new_plot.x_range.range_padding = 0
        new_plot.y_range.range_padding_units = 'percent'
        new_plot.y_range.range_padding = 1

        curdoc().add_root(row(new_plot))


fname = 'chain_prod.txt'
tail = 30000
thin = 20

watcher = Watcher(chain=fname, tail=tail, thin=thin, lightcurve=lc)
