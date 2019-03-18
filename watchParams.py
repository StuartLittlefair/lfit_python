import bokeh as bk
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Band, Whisker
from bokeh.models.annotations import Title
from bokeh.plotting import curdoc, figure
from bokeh.server.callbacks import NextTickCallback
from bokeh.models.widgets import inputs
from bokeh.models.widgets.buttons import Toggle, Button
from bokeh.models.widgets import Slider, Panel, Tabs, Dropdown

import numpy as np
from pandas import read_csv, DataFrame
import configobj
import time

from pprint import pprint

try:
    from lfit import CV
    print("Successfully imported CV class from lfit!")
except:
    print("Failed to import lfit!!")
    exit()

def parseInput(file):
    """Splits input file up making it easier to read"""
    input_dict = configobj.ConfigObj(file)
    return input_dict

class Watcher():
    '''This class will initialise a bokeh page, with some useful lfit MCMC chain supervision tools.
        - Ability to plot a live chain file's evolution over time
        - Interactive lightcurve model, with input sliders or the ability to grab the last step's mean
    '''
    def __init__(self, chain, mcmc_input, tail=5000, thin=0):
        '''
        In the following order:
            - Save the tail and thin parameters to the self.XX
            - Read in the mcmc_input file to self.parDict
            - Set up self.parNames to have all 18 parameters in. Non-complex instances will select the first 14.
            - Set the other misc. trackers in handling files
            - Initialise the data storage object
            - Set up the first tab, with the live chain tracking page
            - Now, regardless of if we're complex or not, generate all 18 of the relevant sliders
                - If we're not complex BS model, the last 4 will just not do anything. Remove their on_change calls?
            - Set up the second tab, with the parameter tweaking tool.
            - Start watching for the creation of the chain file
        '''

        #####################################################
        ############### Information Gathering ###############
        #####################################################

        # Save the tail and thin optional parameters to the self object
        self.tail = tail
        self.thin = thin

        # Save these, just in case I need to use them again later.
        self.mcmc_fname  = mcmc_input
        self.chain_fname = chain

        # Parse the mcmc_input file
        self.parse_mcmc_input()

        # Get the observation data file from the input
        print("Grabbing data files from the input dict:")
        menu = []
        for i in range(self.necl):
            # Grab the filename
            fname = self.mcmc_input_dict["file_{}".format(i)]
            # Append it to menu
            menu.append((fname.split('/')[-1], fname))
        # The observational data filenames will be safe enough in a button.


        # This is a list of the model-generation-relevant parameter names. i.e., 18 parameters. 
        # The model generation function itself will filter out the unwanted ones, rather than changing this.
        self.parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi', 'rdisc_0', 'ulimb_0', 
            'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0', 
            'exp1_0', 'exp2_0', 'tilt_0', 'yaw_0']
        self.parDesc = ['White Dwarf Flux', 'Disc Flux', 'Bright Spot Flux', 'Secondary Flux', 'Mass Ratio', 
            'Eclipse Duration', 'Disc Radius', 'Limb Darkening', 'White Dwarf Radius', 'Bright Spot Scale', 
            'Bright Spot Azimuth', 'Isotropic Emission Fract.', 'Disc Profile', 'Phase Offset', 
            'BS Exponent 1', 'BS Exponent 2', 'BS Emission Tilt', 'BS Emission Yaw']


        
        # fun loading animation
        load = ['.', '..', '...']
        for _ in range(5):
            load.extend(['... -', '... \\', '... |', '... /', '... -', '... \\', '... |', '... /', '... -'])
        load.extend(['...', '..', '.'])
        load.extend(['', '', '', '', '']*4)
        self.load = load
        self.twait = len(load)



        #####################################################
        ########### First tab - Parameter History ###########
        #####################################################
        
        # Drop down box to add parameters to track
        self.selectList = ['']
        self.plotPars = inputs.Select(width=120, title='Track Parameter', options=self.selectList, value='')
        self.plotPars.on_change('value', self.add_tracking_plot)
        # Call the update_necl function
        self.update_necl()

        # Button to switch from the complex to simple BS model, and vice versa
        if self.complex:
            col = 'success'
        else:
            col = 'danger'
        self.complex_button = Toggle(label='Complex BS?', width=120, button_type=col, active=self.complex)
        self.complex_button.on_click(self.update_complex)

        #TODO: TABS!
        # Add stuff to a layout for the area
        # self.tab1_layout = 
        curdoc().add_root(column([self.complex_button, self.plotPars]))

        # Add that layout to a tab
        # self.tab1 = Panel(child=self.tab1_layout, title="Parameter History")



        ######################################################
        ############ Second tab - Model inspector ############
        ######################################################

        # I need a myriad of parameter sliders. The ranges on these should be set by the priors.
        self.par_sliders = []
        for par, title in zip(self.parNames[:14], self.parDesc[:14]):
            param = self.parDict[par]
            slider = Slider(
                title = title,
                start = param[1],
                end   = param[2],
                value = param[0],
                step  = (param[2] - param[1]) / 100,
                width = 200,
            )
            slider.on_change('value', self.update_lc_model)

            self.par_sliders.append(slider)
        
        # Stored in a separate list, so that I can haev them on their own row.
        self.par_sliders_complex = []
        for par, title in zip(self.parNames[14:], self.parDesc[14:]):
            param = self.parDict[par]
            slider = Slider(
                title = title,
                start = param[1],
                end   = param[2],
                value = param[0],
                step  = (param[2] - param[1]) / 100,
                width = 200,
            )
            # If we aren't using a complex model, changing these shouldn't bother updating the model.
            if self.complex:
                slider.on_change('value', self.update_lc_model)
            
            self.par_sliders_complex.append(slider)

        # Data file picker
        self.data_fname = Dropdown(label="Filename", button_type="danger", menu=menu, width=200)
        self.data_fname.on_change('value', self.update_lc_obs)


        # The CV model object needs to be seeded with initial values. Extract these from the sliders
        pars = [slider.value for slider in self.par_sliders]
        if self.complex:
            pars.extend([slider.value for slider in self.par_sliders_complex])
        # Initialise the model
        self.cv = CV(pars)

        # Grab the data from the file, to start with just use the first in the list
        self.lc_obs = read_csv(menu[0][1],
                sep=' ', comment='#', 
                header=None, 
                names=['phase', 'flux', 'err'],
                skipinitialspace=True)
        self.lc_obs['calc'] = self.cv.calcFlux(pars, np.array(self.lc_obs['phase']))

        print("Read in the observation, with the shape {}".format(self.lc_obs.shape))

        # Whisker can only take the ColumnDataSource, not the pandas array
        self.lc_obs = ColumnDataSource(self.lc_obs)

        # Initialise the figure
        self.lc_plot = bk.plotting.figure(title='Lightcurve', plot_height=500, plot_width=1200,
            toolbar_location='above', y_axis_location="left")
        # Plot the lightcurve data
        self.lc_plot.scatter(x='phase', y='flux', source=self.lc_obs, size=5, color='black')

        # # Plot the error bars - Bokeh doesnt have a built in errorbar!?!
        # # The following function does NOT remove old errorbars when new ones are supplied!
        # # This is because they are plotted as annotations, NOT something readliy modifiable!
        # self.lc_plot.add_layout(
        #     Whisker(base='phase', upper='upper', lower='lower', source=self.lc_obs,
        #     upper_head=None, lower_head=None, line_color='black', )
        # )

        # Plot the model
        self.lc_plot.line(x='phase', y='calc', source=self.lc_obs, line_color='red')

        # I want a button that'll turn red when the parameters are invalid. TODO: It should reset the parameters on click
        self.lc_isvalid = Button(label='Valid parameters', width=200, button_type='success')
        self.lc_isvalid.on_click(self.reset_sliders)

        #TODO: TABS!
        # self.tab2_layout = 
        curdoc().add_root(column([
                row([self.data_fname, self.complex_button, self.lc_isvalid]),
                gridplot(self.par_sliders, ncols=4),
                gridplot(self.par_sliders_complex, ncols=4),
                row(self.lc_plot)
            ])
        )
        # self.tab2 = Panel(child=self.tab2_layout, title="Lightcurve Inspector")

        ######################################################
        ############# Add the tabs to the figure #############
        ######################################################

        # # Make a tabs object
        # self.tabs = Tabs(tabs=[self.tab1, self.tab2])
        # # Add it
        # curdoc().add_root(self.tabs)
        curdoc().title = 'MCMC Chain Supervisor'
        try:
            curdoc().theme = 'dark_minimal'
        except:
            pass

        ######################################################
        ## Setup for, and begin watching for the chain file ##
        ######################################################

        # Keep track of how many steps we've skipped so far
        self.thinstep = 0

        # Initial values
        self.s    = 0                                       # Number of steps read in so far
        self.f    = False                                   # File object, initially false so we can wait for it to be created

        # Initialise data storage
        paramFollowSource = ColumnDataSource(dict(step=[]))
        self.paramFollowSource = paramFollowSource

        # Lists of what parameters we want to plot
        self.pars   = []     # List of params
        self.labels = []     # The labels, in the same order as pars

        # Is the file open? Check once a second until it is, then once we find it remove this callback.
        self.check_file = curdoc().add_periodic_callback(self.open_file, 1000)

    def parse_mcmc_input(self):
        '''Parse the mcmc input dict, and store the following:
            - self.complex: bool
                Is the model using the simple or complex BS?
            - self.nWalkers: int
                How many walkers are expected to be in the chain?
            - self.necl: int
                How many eclipses are we using?
            - self.parDict: dict
                Storage for the variables, including priors and initial guesses.
        '''
        self.mcmc_input_dict = parseInput(self.mcmc_fname)

        # Gather the parameters we can use
        self.complex  = bool(self.mcmc_input_dict['complex'])
        self.nWalkers = int(self.mcmc_input_dict['nwalkers'])
        self.necl     = int(self.mcmc_input_dict['neclipses'])

        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
                'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
                'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']
        
        if self.complex:
            parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])

        for i in range(self.necl):
            parNames.extend([template.format(i) for template in parNameTemplate])

        ##### Here, read the parameters all into a dict of {key: [value, lowerlim, upperlim]} #####
        self.parDict = {}
        for param in parNames:
            line = self.mcmc_input_dict[param].strip().split(' ')
            line = [x for x in line if x != '']
            line = [line[0], line[2], line[3]]
            
            parameter = [float(x) for x in line]
            self.parDict[param] = list(parameter)

    def open_file(self):
        '''Check if the chain file has been created yet. If not, do nothing. If it is, set it to self.f'''
        # Open the file, and keep it open
        file = self.chain_fname
        try:
            self.f = open(file, 'r')
        except:
            self.f = False
            return

        # Determine the number of walkers, just to check
        nWalkers = 0
        while True:
            line = self.f.readline()
            line = line.split()
            walker = int(line[0])
            if nWalkers == walker:
                nWalkers += 1
            else:
                break

        # Close and reopen the file to move the cursor back to the beginning.
        self.f.close()
        # We're at step 0 now
        self.f = open(file, 'r')
        print("Expected {} walkers, got {} walkers in the file!".format(self.nWalkers, nWalkers))
        if nWalkers != self.nWalkers:
            self.f.close()
            self.f = False

        # Remove the callback that keeps trying to open the file. 
        # This is down here, in case the above fails. This way, 
        # if it does we should check again in a bit until it works
        try:
            curdoc().remove_periodic_callback(self.check_file)
        except:
            pass
        
        # Create a new callback that periodically reads the file
        curdoc().add_periodic_callback(self.update_chain, 1)

        print("Opened the file {}!".format(file))

    def readStep(self):
        '''Attempt to read in the next step of the chain file.
        If we get an unexpected number of walkers, quit the script.
        If we're at the end of the file, do nothing.'''

        stepData = np.zeros((self.nWalkers, len(self.pars)), dtype=float)
        # If true, return the data. If False, the end of the file was reached before the step was fully read in.
        flag = True

        # Remember where we started
        init = self.f.tell()

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

    def reset_sliders(self):
        '''Set the parameters to the initial guesses.'''
        print("Resetting the sliders!")
        for par, slider in zip(self.parNames[:15], self.par_sliders):
            param = self.parDict[par][0]
            slider.value = param
        if self.complex:
            for par, slider in zip(self.parNames[15:], self.par_sliders_complex):
                param = self.parDict[par][0]
                slider.value = param

    def update_chain(self):
        '''Call the readStep() function, and stream the live chain data to the plotter.'''
        # Do we have anything to plot?
        if self.labels != []:
            step = self.readStep()

            if step is None:
                # No data to plot
                pass
            else:
                # Generate summaries
                means = np.mean(step, axis=0)
                stds  = np.std(step,  axis=0)

                # Stream accepts a dict of lists
                newdata = dict()
                newdata['step'] = [self.s]

                for i, label in enumerate(self.labels):
                    newdata[label+'Mean'] = [means[i]]
                    newdata[label+'StdUpper']  = [means[i]+stds[i]]
                    newdata[label+'StdLower']  = [means[i]-stds[i]]

                # Add to the plot.
                self.paramFollowSource.stream(newdata, self.tail)

    def update_necl(self):
        '''Change the options on self.plotPars to reflect how many eclipses are in the MCMC chain'''

        print("Updating the number of eclipses in the plotPars list.")

        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
            'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
            'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']

        # complex has extra parameters
        if self.complex:
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

    def update_complex(self, new):
        '''Handler for toggling the complex button. This should just enable/disable the complex sliders '''
        
        self.complex = self.complex_button.active

        if self.complex:
            print("Changing to the complex BS model")
            # Complex sliders update the model
            for slider in self.par_sliders_complex:
                slider.on_change('value', self.update_lc_model)
            
            # Initialise a new CV object with the new BS model
            pars = [slider.value for slider in self.par_sliders]
            pars.extend([slider.value for slider in self.par_sliders_complex])
            self.cv = CV(pars)

            self.complex_button.button_type = 'success'

        else:
            print("Changing to the simple BS model")
            # Change the complex sliders to do nothing
            for slider in self.par_sliders_complex:
                slider.on_change('value', self.junk)

            # Initialise a new CV object with the new BS model
            pars = [slider.value for slider in self.par_sliders]

            self.complex_button.button_type = 'danger'

    def update_lc_obs(self, attr, old, new):
        '''redraw the observations for the lightcurve'''
        print("Redrawing the observations")
        # Re-read the observations
        fname = self.data_fname.value
        fname = str(fname)

        print("Plotting data from file {}".format(fname))
        new_obs = read_csv(fname,
                sep=' ', comment='#', header=None, names=['phase', 'flux', 'err'])
        # new_obs['upper'] = new_obs['flux'] + new_obs['err']
        # new_obs['lower'] = new_obs['flux'] - new_obs['err']

        # Figure out which eclipse we're looking at
        template = 'file_{}'
        for i in range(self.necl):
            if self.mcmc_input_dict[template.format(i)] == fname:
                break
        print('This is file {}'.format(i))

        # Set the sliders to the initial guesses for that file
        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi', 'rdisc_0', 'ulimb_0', 
            'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        for par, slider in zip(parNames, self.par_sliders):
            value = self.parDict[par][0]
            slider.value = value
        # Are we complex? If yes, set those too
        if self.complex:
            parNamesComplex = ['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0']
            for par, slider in zip(parNamesComplex, self.par_sliders_complex):
                value = self.parDict[par][0]
                slider.value = value

        # Regenerate the model lightcurve
        pars = [slider.value for slider in self.par_sliders]
        if self.complex:
            pars.extend([slider.value for slider in self.par_sliders_complex])

        new_obs['calc'] = self.cv.calcFlux(pars, np.array(new_obs['phase']))

        # Push that into the data frame
        rollover = len(new_obs['phase'])
        self.lc_obs.stream(new_obs, rollover)
        # TODO: This does not 'un-draw' old errorbars, but leaves them as an artifact on the figure. How can I even fix this?

        # Set the plotting area title
        fname = fname.split('/')[-1]
        title = Title()
        title.text = fname
        print("Trying to change the title of the plot")
        print("Old title: {}".format(self.lc_plot.title.text))
        self.lc_plot.title = title
        print("The title should now be {}".format(self.lc_plot.title.text))

    def update_lc_model(self, attr, old, new):
        '''Redraw the model lightcurve in the second tab'''
        try:
            # Regenerate the model lightcurve
            pars = [slider.value for slider in self.par_sliders]
            if self.complex:
                pars.extend([slider.value for slider in self.par_sliders_complex])
            
            self.lc_obs.data['calc'] = self.cv.calcFlux(pars, np.array(self.lc_obs.data['phase']))
            self.lc_isvalid.button_type = 'success'
            self.lc_isvalid.label = 'Valid Parameters'

        except Exception:
            self.lc_isvalid.button_type = 'danger'
            self.lc_isvalid.label = 'Invalid Parameters'

    def add_tracking_plot(self, attr, old, new):
        '''Add a plot to the page'''

        label = str(self.plotPars.value)
        if label == '':
            return
        par = self.parNames.index(label) + 1
        self.labels.append(label)
        self.pars.append(par)

        # Clear data from the source structure
        self.paramFollowSource.data = {'step': []}
        for l in self.labels:
            self.paramFollowSource.data[l+'Mean']     = []
            self.paramFollowSource.data[l+'StdUpper'] = []
            self.paramFollowSource.data[l+'StdLower'] = []

        # Move the file cursor back to the beginning of the file
        if not self.f is False:
            self.f.close()
            self.f = open(self.chain_fname, 'r')
            self.s = 0

        new_plot = bk.plotting.figure(title=label, plot_height=300, plot_width=1200,
            toolbar_location='above', y_axis_location="right",
            # tools="ypan,ywheel_zoom,ybox_zoom,reset")
            tools=[])
        new_plot.line(x='step', y=label+'Mean', alpha=1, line_width=3, color='red', source=self.paramFollowSource)
        band = Band(base='step', lower=label+'StdLower', upper=label+'StdUpper', source=self.paramFollowSource,
                    level='underlay', fill_alpha=0.5, line_width=0, line_color='black', fill_color='green')
        new_plot.add_layout(band)

        new_plot.x_range.follow = "end"
        new_plot.x_range.follow_interval = self.tail
        new_plot.x_range.range_padding = 0
        new_plot.y_range.range_padding_units = 'percent'
        new_plot.y_range.range_padding = 1

        #TODO: Make this add to the right tab
        curdoc().add_root(row(new_plot))

    def junk(self, attr, old, new):
        '''Sometimes, you just don't want to do anything :\ '''
        pass

fname = 'chain_prod.txt'
tail = 30000
thin = 20

watcher = Watcher(chain=fname, mcmc_input='mcmc_input.dat', tail=tail, thin=thin)
