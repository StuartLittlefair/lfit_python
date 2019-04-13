import bokeh as bk
from bokeh.layouts import row, column, gridplot, layout
from bokeh.models import ColumnDataSource, Band, Whisker
from bokeh.models.annotations import Title
from bokeh.plotting import curdoc, figure
from bokeh.server.callbacks import NextTickCallback
from bokeh.models.widgets import inputs, markups, DataTable, TableColumn
from bokeh.models.widgets.buttons import Toggle, Button
from bokeh.models.widgets import Slider, Panel, Tabs, Dropdown, TextInput

# For corner plots
import matplotlib
matplotlib.use("Agg")

import numpy as np
from pandas import read_csv, DataFrame
import configobj
import time
from os import path

from pprint import pprint

try:
    from lfit import CV
    print("Successfully imported CV class from lfit!")
except:
    print("Failed to import lfit!!")
    exit()

try:
    import mcmc_utils as u
    print("Successfully imported mcmc_utils")
except:
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

        # TODO:
        # - Add a 'plot all parameters' button to the parameter tracker, per eclipse
        # - Make the corner plot function threaded?
        # - Time how frequent steps are, and predict how long the chain will take to complete.

        #####################################################
        ############### Information Gathering ###############
        #####################################################

        print("Gathering information about my initial conditions...")
        # Save the tail and thin optional parameters to the self object
        self.tail = tail
        self.thin = thin
        print("I'll follow {:,d} data points, and thin the chain file by {:,d}".format(self.tail, self.thin))

        # Save these, just in case I need to use them again later.
        self.mcmc_fname  = mcmc_input
        self.chain_fname = chain
        print("Looking for the mcmc input '{}', and the chain file {}".format(self.mcmc_fname, self.chain_fname))

        # Parse the mcmc_input file
        self.parse_mcmc_input()

        # Get the observation data file from the input
        print("Grabbing data files from the input dict...")
        menu = []
        for i in range(self.necl):
            # Grab the filename
            fname = self.mcmc_input_dict["file_{}".format(i)]
            # Append it to menu
            menu.append((fname.split('/')[-1], fname))
        # The observational data filenames will be safe enough in a button.

        # Parameter keys
        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
                'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
                'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']

        # Extra parameters for the complex model
        if self.complex:
            parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])
            print("Using the complex BS model!")
        else:
            print("Using the simple BS model!")

        # Extra parameters for the GP
        if self.GP:
            print("Using the GP!")
            parNames.extend(['ampin_gp', 'ampout_gp', 'tau_gp'])

        # Extend the parameter names for each eclipse
        for i in range(1, self.necl):
            parNames.extend([template.format(i) for template in parNameTemplate])
        
        # Copy the above onto self
        self.parNames = list(parNames)
        # Human-readable names
        self.parDesc = ['White Dwarf Flux', 'Disc Flux', 'Bright Spot Flux', 'Secondary Flux', 'Mass Ratio',
            'Eclipse Duration', 'Disc Radius', 'Limb Darkening', 'White Dwarf Radius', 'Bright Spot Scale',
            'Bright Spot Azimuth', 'Isotropic Emission Fract.', 'Disc Profile', 'Phase Offset',
            'BS Exponent 1', 'BS Exponent 2', 'BS Emission Tilt', 'BS Emission Yaw']

        if self.GP:
            self.parNames.extend(['ampin_gp', 'ampout_gp', 'tau_gp'])

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

        print("Creating the Parameter History tab...")
        # Drop down box to add parameters to track
        self.selectList = [('', '')]
        self.plotPars = Dropdown(width=120, label='Track Parameter', button_type='primary', menu=self.selectList)
        self.plotPars.on_change('value', self.add_tracking_plot)
        # Call the update_selectList function
        self.update_selectList()
        print("Made the parameter picker...")

        # Lets report some characteristics of the chain
        self.reportChain_label = markups.Div(width=500)
        self.make_header()
        print("Made the little header")

        # Thinning input
        self.thin_input = TextInput(placeholder='Number of steps to skip over', width=200)
        self.thin_input.on_change('value', self.update_thinning)

        # # Tail length - no way to update existing plot tail lengths...
        # self.tail_input = TextInput(placeholder='Number of steps to plot', width=200)
        # self.tail_input.on_change('value', self.update_tail)
        
        # Add stuff to a layout for the area
        self.tab1_layout = column([self.reportChain_label, self.thin_input, self.plotPars])

        # Add that layout to a tab
        self.tab1 = Panel(child=self.tab1_layout, title="Parameter History")
        print("First tab done!")


        ######################################################
        ############ Second tab - Model inspector ############
        ######################################################

        print("Creating the second tab...")
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
                # format='{:.5}'
            )
            slider.on_change('value', self.update_lc_model)
            # slider.callback_throttle = 100 #ms?

            self.par_sliders.append(slider)

        # Stored in a separate list, so that I can haev them on their own row.
        # Default values:
        defaults = {
            'exp1_0': [ 4.71, 0.001,  5.0],
            'exp2_0': [ 1.19, 0.001,  5.0],
            'tilt_0': [40.60, 0.001,  180],
            'yaw_0': [35.35, -90.0, 90.0],
        }
        self.par_sliders_complex = []
        for par, title in zip(self.parNames[14:], self.parDesc[14:]):
            try:
                param = self.parDict[par]
            except:
                param = defaults[par]
            slider = Slider(
                title = title,
                start = param[1],
                end   = param[2],
                value = param[0],
                step  = (param[2] - param[1]) / 100,
                width = 200,
                # format='{:.5}'
            )
            # slider.callback_throttle = 100 #ms?
            # If we aren't using a complex model, changing these shouldn't bother updating the model.
            if self.complex:
                slider.on_change('value', self.update_lc_model)

            self.par_sliders_complex.append(slider)
        print("Made the sliders...")

        # Data file picker
        self.lc_change_fname_button = Dropdown(label="Choose Data", button_type="success", menu=menu, width=250)
        self.lc_obs_fname = menu[0][0]
        self.lc_change_fname_button.on_change('value', self.update_lc_obs)
        print("Made the data picker...")

        # Button to switch from the complex to simple BS model, and vice versa
        if self.complex:
            col = 'success'
        else:
            col = 'danger'
        self.complex_button = Toggle(label='Complex BS?', width=120, button_type=col, active=self.complex)
        self.complex_button.on_click(self.update_complex)
        print("Made the complex button...")

        # The CV model object needs to be seeded with initial values. Extract these from the sliders
        print("Initialising the CV model...", end='')
        pars = [slider.value for slider in self.par_sliders]
        if self.complex:
            pars.extend([slider.value for slider in self.par_sliders_complex])
        # Initialise the model
        self.cv = CV(pars)
        print(" Done.")

        print("Grabbing the observations...")
        # Grab the data from the file, to start with just use the first in the list
        self.lc_obs = read_csv(menu[0][1],
                sep=' ', comment='#',
                header=None,
                names=['phase', 'flux', 'err'],
                skipinitialspace=True)
            
        # Total mode lightcurve
        self.lc_obs['calc'] = self.cv.calcFlux(pars, np.array(self.lc_obs['phase']))
        # Components 
        self.lc_obs['sec']   = self.cv.yrs
        self.lc_obs['bspot'] = self.cv.ys
        self.lc_obs['wd']    = self.cv.ywd
        self.lc_obs['disc']  = self.cv.yd

        print("Read in the observation, with the shape {}".format(self.lc_obs.shape))

        # Whisker can only take the ColumnDataSource, not the pandas array
        self.lc_obs = ColumnDataSource(self.lc_obs)

        print("Creating the LC plot...", end='')
        # Initialise the figure
        title = menu[0][0]
        self.lc_plot = bk.plotting.figure(title=title, plot_height=500, plot_width=1200,
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
        self.lc_plot.line(x='phase', y='calc',  source=self.lc_obs,            line_color='red')
        self.lc_plot.line(x='phase', y='sec',   source=self.lc_obs, alpha=0.5, line_color='brown')
        self.lc_plot.line(x='phase', y='wd',    source=self.lc_obs, alpha=0.5, line_color='blue')
        self.lc_plot.line(x='phase', y='bspot', source=self.lc_obs, alpha=0.5, line_color='green')
        self.lc_plot.line(x='phase', y='disc',  source=self.lc_obs, alpha=0.5, line_color='magenta')
        print(" Done")

        # I want a button that'll turn red when the parameters are invalid.
        self.lc_isvalid = Button(label='Initial Parameters', width=200)
        self.lc_isvalid.on_click(self.reset_sliders)
        print("Made the valid parameters button")

        # Write the current slider values to mcmc_input.dat
        self.write2input_button = Button(label='Write current values', width=200)
        self.write2input_button.on_click(self.write2input)
        print("Made the write2input button") 

        # Arrange the tab layout
        self.tab2_layout = column([
            row([self.lc_change_fname_button, self.complex_button, self.lc_isvalid, self.write2input_button]),
            self.lc_plot,
            row([gridplot(self.par_sliders, ncols=4),
                 gridplot(self.par_sliders_complex, ncols=1)]),
        ])

        self.tab2 = Panel(child=self.tab2_layout, title="Lightcurve Inspector")
        print("Constructed the Lightcurve Inspector tab!")

        ######################################################
        ################# Tab 3: Corner plot #################
        ######################################################

        # Make corner plots. I need to know how long the burn in is though!
        self.burn_input = TextInput(placeholder='No. steps to discard', )
        self.corner_plot_button = Button(label='Make corner plots')
        self.corner_plot_button.on_click(self.make_corner_plots)
        print("Defualt button type is {}".format(self.corner_plot_button.button_type))

        self.cornerReporter = markups.Div(width=700)
        self.cornerReporter.text = "The chain file will have <b>{:,d}</b> steps when completed</br>".format(self.nProd)
        self.cornerReporter.text += "We're using <b>{:,d}</b> walkers, making for <b>{:,d}</b> total lines to read in.</br>".format(
            self.nWalkers, self.nProd*self.nWalkers)
        self.cornerReporter.text += "I've not yet added support for embedded images here, and bokeh isn't a great tool for corner plots this big. You'll probably have to scp the files manually."
        curdir = path.dirname(path.realpath(__file__))
        self.cornerReporter.text += "This one-liner should do it:</br><b>scp callisto:{}/eclipse*.png .</b>".format(curdir)

        #TODO: 
        # - Show the corner plots in the page? Or, add a link to download them?
        # - Corner plots can make the server run out of memory for large files! can we fix this?

        self.tab3_layout = column([self.burn_input, self.corner_plot_button, self.cornerReporter])
        self.tab3 = Panel(child=self.tab3_layout, title="Corner Plotting")


        ######################################################
        ################# Tab 4: Param Table #################
        ######################################################

        self.tableColumns = ['file', 'wdFlux', 'dFlux', 'sFlux', 'rsFlux', 'rdisc', 
            'ulimb', 'scale', 'az', 'fis', 'dexp', 'phi0']
        
        # Extra parameters for the complex model
        if self.complex:
            self.tableColumns.extend(['exp1', 'exp2', 'tilt', 'yaw'])

        tableDict = {}
        for p in self.tableColumns:
            get = p + "_{}"
            tableDict[p] = [self.parDict[get.format(i)][0] for i in range(self.necl)]

        self.lastStep_CDS = ColumnDataSource(tableDict)
        columns = [
            TableColumn(field=par, title=par) for par in self.tableColumns
        ]
        self.parameter_table = DataTable(source=self.lastStep_CDS, columns=columns, 
            fit_columns=True, index_header='Eclipse Number')

        self.tab4_layout = column([self.parameter_table])
        self.tab4 = Panel(child=self.tab4_layout, title="Parameter table")


        ######################################################
        ############# Add the tabs to the figure #############
        ######################################################

        # Make a tabs object
        self.tabs = Tabs(tabs=[self.tab1, self.tab2, self.tab3, self.tab4])
        # Add it
        self.doc = curdoc()
        self.doc.add_root(self.tabs)
        print("Added the tabs to the document!")
        self.doc.title = 'MCMC Chain Supervisor'

        ######################################################
        ## Setup for, and begin watching for the chain file ##
        ######################################################

        print("Setting up the chain file watcher...")
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
        self.check_file = self.doc.add_periodic_callback(self.open_file, 1000)

        print("Finished initialising the dashboard!")

    def parse_mcmc_input(self):
        '''Parse the mcmc input dict, and store the following:
            - self.complex: bool
                Is the model using the simple or complex BS
            - self.GP: bool
                Is the model using the gaussian process?
            - self.nWalkers: int
                How many walkers are expected to be in the chain?
            - self.necl: int
                How many eclipses are we using?
            - self.parDict: dict
                Storage for the variables, including priors and initial guesses.
            - self.nBurn: int
                The number of burn-in steps.
            - self.nProd: int
                The number of product steps.
        '''
        print("Parsing the mcmc_input file, '{}'...".format(self.mcmc_fname))
        self.mcmc_input_dict = parseInput(self.mcmc_fname)

        # Gather the parameters we can use
        self.complex  = bool(int(self.mcmc_input_dict['complex']))
        self.nWalkers = int(self.mcmc_input_dict['nwalkers'])
        self.necl     = int(self.mcmc_input_dict['neclipses'])
        self.GP       = bool(int(self.mcmc_input_dict['useGP']))
        self.nBurn    = int(self.mcmc_input_dict['nburn'])
        self.nProd    = int(self.mcmc_input_dict['nprod'])

        # Parameter keys
        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
                'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
                'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']

        if self.complex:
            parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])
            print("Using the complex BS model!")
        else:
            print("Using the simple BS model!")

        if self.GP:
            print("Using the GP!")
            parNames.extend(['ampin_gp', 'ampout_gp', 'tau_gp'])

        for i in range(1, self.necl):
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
            print("Found the file, '{}'!".format(file))
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
            print("Got a walker mismatch. Using the walkers I found by looking in the file...")
            self.nWalkers = nWalkers

        # Remove the callback that keeps trying to open the file.
        # This is down here, in case the above fails. This way,
        # if it does we should check again in a bit until it works
        try:
            self.doc.remove_periodic_callback(self.check_file)
            print("No longer re-checking for the file.")
        except:
            pass

        # Create a new callback that periodically reads the file
        print('Adding next tick callback')
        self.doc.add_next_tick_callback(self.update_chain)

        print("Succesfully opened the chain '{}'!".format(file))

    def readStep(self):
        '''Attempt to read in the next step of the chain file.
        If we get an unexpected number of walkers, quit the script.
        If we're at the end of the file, do nothing.'''

        stepData = np.zeros((self.nWalkers, len(self.pars)), dtype=float)
        # If true, return the data. If False, the end of the file was reached before the step was fully read in.
        flag = True


        self.thinstep += 1

        if self.thin:
            if self.thinstep % self.thin != 0:
                # This only processes the line once every <thin> lines
                flag = None

        try:
            # Remember where we started
            start = self.f.tell()
            lastStep = np.zeros(len(self.selectList)-2, dtype=np.float64)
            for i in np.arange(self.nWalkers):
                # Get the next line
                line = self.f.readline().strip()

                # Do we actually want to process this data?
                if flag is True:
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
                    lastStep += line[1:-1]
                    values = line[self.pars]

                    stepData[w, :] = values
            lastStep /= float(self.nWalkers)
        except IndexError:
            # Sometimes empty lines slip through. Catch the exceptions
            print(line)
            flag = False

        if flag is True:
            # We successfully read in the chunk!
            self.s += 1
            # print("Successful step, Adding next tick callback")
            self.next_read = self.doc.add_next_tick_callback(self.update_chain)
            self.init = start
            self.lastStep = np.array(lastStep)
            return stepData
        elif flag is False:
            # The most recent step wasn't completely read in
            self.f.seek(start)
            # print("Adding timeout callback")
            print('  End of file! waiting for new step to be written...', end='\r')
            self.next_read = self.doc.add_timeout_callback(self.update_chain, 10000)
            print("\nUpdating the table with lastStep...")
            for i, n in enumerate(self.parNames):
                self.lastStep_CDS.data[n] = lastStep[i]
            return None
        else:
            # We read in a step but we don't want it.
            self.s += 1
            # print("Step we want to skip, adding next tick callback")
            self.next_read = self.doc.add_next_tick_callback(self.update_chain)
            self.init = start
            return None

    def reset_sliders(self):
        '''Set the parameters to the initial guesses.'''
        print("Resetting the sliders!")

        # Figure out which eclipse we're looking at
        fname = self.lc_obs_fname
        template = 'file_{}'
        for i in range(self.necl):
            this = self.mcmc_input_dict[template.format(i)]
            if fname in this:
                break
            else: i = None
        if i is None:
            print("Couldn't find the index of that file!")
            return
        fileNumber = int(i)
        print('This is file {}'.format(fileNumber))

        parNames = self.parNames

        # If we have s > 0, that means we've read in some chain. Get the last step.
        if self.s > 0:
            stepData = self.lastStep
        else:
            print('Getting values from the parameter dict.')
            stepData = [self.parDict[key][0] for key in parNames]
            # for i, j in zip(parNames, stepData):
            #     print("{:>15s}: {}".format(i, j))

        # Set the values of the sliders to the right values
        for par, slider in zip(parNames[:15], self.par_sliders):
            get = par.replace('_0', '_{}'.format(fileNumber))
            index = parNames.index(get)
            param = stepData[index]
            
            print("Setting the slider for {} to {}".format(get, param))
            slider.on_change('value', self.junk)
            slider.value = param
            slider.on_change('value', self.update_lc_model)

        # Set the complex values, if needs be
        if self.complex:
            print("Setting the complex sliders")
            complex_names = ['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0']
            for par, slider in zip(complex_names, self.par_sliders_complex):
                get = par.replace('_0', '_{}'.format(fileNumber))
                index = parNames.index(get)
                param = stepData[index]

                print("Setting the slider for {} to {}".format(get, param))
                slider.on_change('value', self.junk)
                slider.value = param
                slider.on_change('value', self.update_lc_model)
        
        self.lc_isvalid.button_type = 'default'

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
                means = np.nanmean(step, axis=0)
                stds  = np.nanstd(step,  axis=0)

                # Stream accepts a dict of lists
                newdata = dict()
                newdata['step'] = [self.s]

                for i, label in enumerate(self.labels):
                    newdata[label+'Mean'] = [means[i]]
                    newdata[label+'StdUpper']  = [means[i]+stds[i]]
                    newdata[label+'StdLower']  = [means[i]-stds[i]]

                # Add to the plot.
                self.paramFollowSource.stream(newdata, self.tail)

    def update_selectList(self):
        '''Change the options on self.plotPars to reflect how many eclipses are in the MCMC chain'''

        print("Updating the number of eclipses in the plotPars list.")

        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
            'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
            'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']

        # complex has extra parameters
        if self.complex:
            print("Adding complex params")
            parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            parNameTemplate.extend(['exp1_{0}', 'exp2_{0}', 'tilt_{0}', 'yaw_{0}'])

        if self.GP:
            print("Adding GP params")
            parNames.extend(['ampin_gp', 'ampout_gp', 'tau_gp'])

        # Format labels
        for i in range(self.necl-1):
            for name in parNameTemplate:
                parNames.append(name.format(i+1))

        self.parNames = list(parNames)
        
        parNames.append('Likelihood')

        self.selectList = [(par, par) for par in parNames]
        self.selectList.insert(0, ('', ''))
        self.plotPars.menu = self.selectList

    def update_complex(self, new):
        '''Handler for toggling the complex button. This should just enable/disable the complex sliders '''

        print("Toggling the complex model...")
        self.complex = self.complex_button.active
        print("The complex variable is now {}".format('on' if self.complex else 'off'))

        if self.complex:
            print("Changing to the complex BS model")
            # Complex sliders update the model
            for slider in self.par_sliders_complex:
                slider.on_change('value', self.update_lc_model)
            print("Enabled the comlpex sliders")

            # Initialise a new CV object with the new BS model
            pars = [slider.value for slider in self.par_sliders]
            pars.extend([slider.value for slider in self.par_sliders_complex])
            self.cv = CV(pars)
            print("Re-initialised the CV model")

            self.complex_button.button_type = 'success'

        else:
            print("Changing to the simple BS model")
            # Change the complex sliders to do nothing
            for slider in self.par_sliders_complex:
                slider.on_change('value', self.junk)
            print("Disabled the comlpex sliders")

            # Initialise a new CV object with the new BS model
            pars = [slider.value for slider in self.par_sliders]
            print("Re-initialised the CV model")

            self.complex_button.button_type = 'danger'

        self.update_lc_model('value', None, None)

    def update_thinning(self, attr, old, new):
        thin = self.thin_input.value
        try:
            thin = int(thin)
        except:
            self.thin_input.value = ''
        
        self.thin = thin
        self.make_header()

        # Clear data from the source structure
        self.paramFollowSource.data = {'step': []}
        for l in self.labels:
            self.paramFollowSource.data[l+'Mean']     = []
            self.paramFollowSource.data[l+'StdUpper'] = []
            self.paramFollowSource.data[l+'StdLower'] = []

        print("Reset the data storage to empty")

        # Move the file cursor back to the beginning of the file
        if not self.f is False:
            self.f.close()
            self.f = open(self.chain_fname, 'r')
            self.s = 0

        print("Closed and re-opened the file!")

    def make_header(self):
        self.reportChain_label.text =  'This chain has <b>{:,d}</b> burn steps, and <b>{:,d}</b> product steps.</br>'.format(
            self.nBurn, self.nProd)
        self.reportChain_label.text += " We're using <b>{:,d}</b> walkers,".format(self.nWalkers)

        if bool(int(self.mcmc_input_dict['usePT'])):
            self.reportChain_label.text += ' with parallel tempering sampling <b>{:,d}</b> temperatures,'.format(
                int(self.mcmc_input_dict['ntemps']))

        self.reportChain_label.text += ' and running on <b>{:,d}</b> cores.</br>'.format(int(self.mcmc_input_dict['nthread']))
        if self.thin:
            p = str(self.thin)
            if p[-1] == '1':
                suffix = 'st'
            elif p[-1] == '2':
                suffix = 'nd'
            elif p[-1] == '3':
                suffix = 'rd'
            else:
                suffix = 'th'

            self.reportChain_label.text += " When plotting parameter evolutions, I'm skipping every "
            self.reportChain_label.text += "<b>{}{}</b> step and only keeping the last <b>{:,d}</b> data".format(self.thin, suffix, self.tail)
        else:
            self.reportChain_label.text += " When plotting parameter evolutions, I'll plot every step."

    def update_lc_obs(self, attr, old, new):
        '''redraw the observations for the lightcurve'''

        print("Redrawing the observations")
        # Re-read the observations
        fname = self.lc_change_fname_button.value
        fname = str(fname)
        self.lc_obs_fname = fname

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
        parNames = ['wdFlux_{}', 'dFlux_{}', 'sFlux_{}', 'rsFlux_{}', 'q', 'dphi', 'rdisc_{}', 'ulimb_{}',
            'rwd', 'scale_{}', 'az_{}', 'fis_{}', 'dexp_{}', 'phi0_{}']
        parNames = [x.format(i) for x in parNames]
        for par, slider in zip(parNames, self.par_sliders):
            value = self.parDict[par][0]
            slider.on_change('value', self.junk)
            slider.value = value
            slider.on_change('value', self.update_lc_model)
        # Are we complex? If yes, set those too
        if self.complex:
            parNamesComplex = ['exp1_{}', 'exp2_{}', 'tilt_{}', 'yaw_{}']
            parNamesComplex = [x.format(i) for x in parNamesComplex]
            for par, slider in zip(parNamesComplex, self.par_sliders_complex):
                value = self.parDict[par][0]
            slider.on_change('value', self.junk)
            slider.value = value
            slider.on_change('value', self.update_lc_model)

        # Regenerate the model lightcurve
        pars = [slider.value for slider in self.par_sliders]
        if self.complex:
            pars.extend([slider.value for slider in self.par_sliders_complex])

        # Total mode lightcurve
        new_obs['calc'] = self.cv.calcFlux(pars, np.array(new_obs['phase']))
        # Components 
        new_obs['sec']   = self.cv.yrs
        new_obs['bspot'] = self.cv.ys
        new_obs['wd']    = self.cv.ywd
        new_obs['disc']  = self.cv.yd

        # Push that into the data frame
        rollover = len(new_obs['phase'])
        self.lc_obs.stream(new_obs, rollover)

        # Set the plotting area title
        fname = fname.split('/')[-1]
        print("Trying to change the title of the plot")
        print("Old title: {}".format(self.lc_plot.title.text))
        self.lc_plot.title.text = fname
        print("The title should now be {}".format(self.lc_plot.title.text))

    def update_lc_model(self, attr, old, new):
        '''Redraw the model lightcurve in the second tab'''

        try:
            # Regenerate the model lightcurve
            pars = [slider.value for slider in self.par_sliders]
            if self.complex:
                pars.extend([slider.value for slider in self.par_sliders_complex])

            self.lc_obs.data['calc'] = self.cv.calcFlux(pars, np.array(self.lc_obs.data['phase']))
            # Total mode lightcurve
            self.lc_obs.data['calc'] = self.cv.calcFlux(pars, np.array(self.lc_obs.data['phase']))
            # Components 
            self.lc_obs.data['sec']   = self.cv.yrs
            self.lc_obs.data['bspot'] = self.cv.ys
            self.lc_obs.data['wd']    = self.cv.ywd
            self.lc_obs.data['disc']  = self.cv.yd


        except Exception:
            print("Invalid parameters!")
            self.lc_isvalid.button_type = 'danger'
            self.lc_isvalid.label = 'Invalid Parameters'

    def write2input(self):
        '''Get the slider values, and modify mcmc_input.dat to match them.'''
        
        # Figure out which eclipse we're looking at
        fname = self.lc_obs_fname
        template = 'file_{}'
        for i in range(self.necl):
            this = self.mcmc_input_dict[template.format(i)]
            if fname in this:
                break
            else: i = None
        if i is None:
            print("Couldn't find the index of that file!")
            return
        fileNumber = int(i)
        print('This is file {}'.format(fileNumber))

        # Get a list of the parameters we're going to change
        values = [slider.value for slider in self.par_sliders]

        labels = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
                'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        if self.complex:
            labels.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            values.extend([slider.value for slider in self.par_sliders_complex])

        labels = [l.replace('_0', '_{}'.format(fileNumber)) for l in labels]

        newvalues = {}
        for par, value in zip(labels, values):
            newvalues[par] = value

        # Make a copy of the mcmc_input file and edit that.
        mcmc_file = []
        with open(self.mcmc_fname, 'r') as f:
            for line in f:
                line_components = line.strip().split()
                if len(line_components) > 0:
                    par = line_components[0]
                    if par in labels:
                        value = newvalues[par]
                        
                        newline = line_components.copy()
                        newline[2] = value
                        newline = "{:>10s} = {:>12.4f} {:>12} {:>12.4f} {:>12.4f} {:>12}\n".format(
                            str(newline[0]),
                            float(newline[2]), 
                            str(newline[3]),
                            float(newline[4]), 
                            float(newline[5]), 
                            int(newline[6])
                            )
                        
                        line = newline
                mcmc_file.append(line)

        # Overwrite the old mcmc_input file.
        print('Writing new file, {}'.format(self.mcmc_fname))
        with open(self.mcmc_fname, 'w') as f:
            for line in mcmc_file:
                f.write(line)
        self.parse_mcmc_input()

    def add_tracking_plot(self, attr, old, new):
        '''Add a plot to the page'''

        print("Attempting to add a plot to the page")

        label = str(self.plotPars.value)
        params = [par[0] for par in self.selectList]
        par = params.index(label)
        self.labels.append(label)
        self.pars.append(par)

        # Clear data from the source structure
        self.paramFollowSource.data = {'step': []}
        for l in self.labels:
            self.paramFollowSource.data[l+'Mean']     = []
            self.paramFollowSource.data[l+'StdUpper'] = []
            self.paramFollowSource.data[l+'StdLower'] = []

        print("Reset the data storage to empty")

        # Move the file cursor back to the beginning of the file
        if not self.f is False:
            self.f.close()
            self.f = open(self.chain_fname, 'r')
            self.s = 0

        print("Closed and re-opened the file!")

        new_plot = bk.plotting.figure(title=label, plot_height=300, plot_width=1200,
            toolbar_location='above', y_axis_location="right",
            tools="ypan,ywheel_zoom,ybox_zoom,reset")
            # tools=[])
        new_plot.line(x='step', y=label+'Mean', alpha=1, line_width=3, color='red', source=self.paramFollowSource)
        band = Band(base='step', lower=label+'StdLower', upper=label+'StdUpper', source=self.paramFollowSource,
                    level='underlay', fill_alpha=0.5, line_width=0, line_color='black', fill_color='green')
        new_plot.add_layout(band)

        new_plot.x_range.follow = "end"
        new_plot.x_range.follow_interval = self.tail
        new_plot.x_range.range_padding = 0
        new_plot.y_range.range_padding_units = 'percent'
        new_plot.y_range.range_padding = 1

        # Make this add to the right tab
        self.tab1_layout.children.append(row(new_plot))

        print("Added a new plot!")

        self.doc.add_next_tick_callback(self.update_chain)

        self.lc_isvalid.label = 'Get current step'
        self.lc_isvalid.button_type = 'default'

    def make_corner_plots(self):
        print("Making corner plots...")
        self.cornerReporter.text += "</br>Reading chain file (this can take a while)...  "
        print("Reading chain file...")
        chainFile = open('chain_prod.txt', 'r')
        chain = u.readchain(chainFile)
        self.cornerReporter.text += "Done!"
        print("Done!")

        N = self.burn_input.value
        try:
            N = int(N)
            self.cornerReporter.text += "</br>Throwing away the first {:,d} steps of the product phase...".format(N)
        except:
            N = 0
        
        chain = chain[:, N:, :]
        self.cornerReporter.text += "</br>Using {:,d} steps".format(chain.shape[1])
        flat = u.flatchain(chain, chain.shape[2])
        self.cornerReporter.text += ", which is {:,d} lines of the file...".format(flat.shape[0])

        # Label all the columns in the chain file
        necl    = self.necl
        complex = self.complex
        useGP   = self.GP

        # Get labels
        parNames = ['wdFlux_0', 'dFlux_0', 'sFlux_0', 'rsFlux_0', 'q', 'dphi',\
                'rdisc_0', 'ulimb_0', 'rwd', 'scale_0', 'az_0', 'fis_0', 'dexp_0', 'phi0_0']
        parNameTemplate = ['wdFlux_{0}', 'dFlux_{0}', 'sFlux_{0}', 'rsFlux_{0}',\
                'rdisc_{0}', 'ulimb_{0}', 'scale_{0}', 'az_{0}', 'fis_{0}', 'dexp_{0}', 'phi0_{0}']

        perm = [4,5,8]

        c = 11
        if complex:
            self.cornerReporter.text += "</br>Using the complex model"
            parNames.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            parNameTemplate.extend(['exp1_0', 'exp2_0', 'tilt_0', 'yaw_0'])
            c = 15
        if useGP:
            self.cornerReporter.text += "</br>Using the Gaussian process"
            parNames.extend(['ampin_gp', 'ampout_gp', 'tau_gp'])
            perm.extend([parNames.index(i) for i in  ['ampin_gp', 'ampout_gp', 'tau_gp']])

        for i in range(necl-1):
            for name in parNameTemplate:
                parNames.append(name.format(i+1))

        # Make the corner plots. This is pretty CPU intensive!
        a = 0; b = 14; j = 0
        while b <= len(parNames):
            night = flat[:, a:b]
            labels = parNames[a:b]
            if a:
                for i in perm:
                    labels.append(parNames[i])
                
                night = np.concatenate((night, flat[:, perm]), axis=1)

            self.cornerReporter.text += "</br>Making the figure for eclipse {}...".format(j)
            print("Making the figure for eclipse {}".format(j))
            fig = u.thumbPlot(night, labels)
            oname = 'eclipse{}.png'.format(j)
            print("Done!")
            self.cornerReporter.text += "</br>Done! Saving to memory..."
            fig.savefig(oname)
            self.cornerReporter.text += "</br>Done figure '{}'".format(oname)
            del fig
            del night

            a = b
            b += c
            j += 1

    def junk(self, attr, old, new):
        '''Sometimes, you just don't want to do anything'''
        print("Calling the junk pile")
        pass

if __name__ in '__main__':
    print("This script must be run within a bokeh server:")
    print("  bokeh serve --show watchParams.py")
    print("Stopping!")
else:
    fname = 'chain_prod.txt'
    mc_fname = 'mcmc_input.dat'
    tail = 10000
    thin = 20

    watcher = Watcher(chain=fname, mcmc_input=mc_fname, tail=tail, thin=thin)
