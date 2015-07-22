import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from trm import roche
import sys
import lfit
import emcee
import george
import GaussianProcess as GP
from george import kernels
from mcmc_utils import *
import seaborn
from collections import MutableSequence

# parallellise with MPIPool
from emcee.utils import MPIPool

class LCModel(MutableSequence):
    '''CV lightcurve model for multiple eclipses
       can be passed to routines for calculating model, chisq, prior prob etc
       
       can add eclipses at will with routing addEcl. All eclipses share q, dphi, rwd, 
       limb darkening. All other parameters vary. You cannot mix and match complex
       and simple bright spot models for eclipses; all must use the same type of
       bright spot.
       
       Also behaves as a list of the current values of all variable parameters - 
       this allows it to be seamlessly used with emcee'''
       
    def __init__(self,parList,nel_disc=1000,nel_donor=400):
        '''parameter list should be a 16 element or 20 element dictionary of Param objects
        in order these are:
        amp_gp, tau_gp, wdFlux, dFlux, sFlux, rsFlux, q, dphi, rdisc, ulimb, rwd, scale, az, fis, dexp, phi0
        and optional pars are
        exp1, exp2, tilt, yaw
        '''
        amp_gp,tau_gp,wdFlux,dFlux,sFlux,rsFlux,q,dphi,rdisc,ulimb,rwd,scale,az,fis,dexp,phi0 = parList[0:15]
        complex = False
        if len(parList) > 16:
            exp1, exp2, tilt, yaw = parList[16:]
            complex = True
        # put into class instance
        self.q = q
        self.rwd = rwd
        self.dphi = dphi
        self.amp_gp = amp_gp
        self.tau_gp = tau_gp
        # make sure these are not variables
        self.amp_gp.isVar = False
        self.tau_gp.isVar = False
        
        # we actually need an LFIT CV object to do the calculations
        parVals = [par.startVal for par in parList[2:]]
        self.cv = lfit.CV(parVals)
        
        # now parameters that differ for each eclipse. put these in a list
        # this allows us to get params for a given eclipse
        self.wdFlux = [wdFlux]
        self.dFlux  = [dFlux]
        self.sFlux  = [sFlux]
        self.rsFlux  = [rsFlux]
        self.rdisc  = [rdisc]
        self.ulimb = [ulimb]
        self.scale  = [scale]
        self.az     = [az]
        self.fis    = [fis]
        self.dexp   = [dexp]
        self.phi0   = [phi0]
        self.complex = complex
        if self.complex:
            self.exp1   = [exp1]
            self.exp2   = [exp2]
            self.tilt   = [tilt]
            self.yaw    = [yaw]
        
        self.necl = 1
        
        # now we need a data property which yields a simple list of all active parameters

        # for list itself, this should be all the parameters which are considered
        # variable. We use the @property decorator to do clever stuff here
        # the property _data contains the list of all params, we want just the variable ones
        self._data = parList
        
    '''by having data be a list of all variable parameters, we can have the model
       keep track of all parameters, but to emcee (which treats the model like a list)
       it looks like it has a smaller number of parameters, which vary'''
    @property
    def data(self):
        return [param for param in self._data if param.isVar]
    @property
    def npars(self):
        return len(self.data)
                    
    # the following routines determine what happens when we try to extract, set or
    # delete this list items
    def __getitem__(self,ind):
        # returns the current value of this parameter
        return self.data[ind].currVal
    def __setitem__(self,ind,val):
        # set the current value of this parameter
        self.data[ind].currVal = val
    def __delitem__(self,ind):
        # delete this from the list (should never use this, but needed for implementation)
        self.data.remove(ind)
    def __len__(self):
        # returns len of list
        return len(self.data)
    def insert(self,ind,val):
        # add a value into list at index (again, shouldn't be used)
        self.data.insert(ind,val)

    def addEclipse(self,parList):
        '''parList should be a list of 11 or 15 Param objects, depending on the complexity
        of the bright spot model. In turn these should be
        wdFlux, dFlux, sFlux, rsFlux, rdisc, ulimb, scale, az, fis, dexp, phi0
        and optional pars are
        exp1, exp2, tilt, yaw
        '''    
        if self.complex:
            assert len(parList) == 15, "Wrong number of parameters"
            wdFlux,dFlux,sFlux,rsFlux,rdisc,ulimb,scale,az,fis,dexp,phi0,exp1,exp2,tilt,yaw = parList[0:14]
        else:
            assert len(parList) == 11, "Wrong number of parameters"
            wdFlux,dFlux,sFlux,rsFlux,rdisc,ulimb,scale,az,fis,dexp,phi0 = parList[0:10]
            
        self.necl += 1
        self.wdFlux.append(wdFlux)
        self.dFlux.append(dFlux)
        self.sFlux.append(sFlux)
        self.rsFlux.append(rsFlux)
        self.rdisc.append(rdisc)
        self.ulimb.append(ulimb)
        self.scale.append(scale)
        self.az.append(az)
        self.fis.append(fis)
        self.dexp.append(dexp)
        self.phi0.append(phi0)
        if self.complex:
            self.exp1.append(exp1)
            self.exp2.append(exp2)
            self.tilt.append(tilt)
            self.yaw.append(yaw)
        # add params to _data list
        self._data.extend(parList)
        
    def calc(self,ecl,phi,width=None):
        '''we have to extract the current value of the parameters for this ecl, and 
        calculate the CV flux'''
        eclPars = [self.wdFlux[ecl].currVal, self.dFlux[ecl].currVal, self.sFlux[ecl].currVal, \
        	self.rsFlux[ecl].currVal, self.q.currVal, self.dphi.currVal, self.rdisc[ecl].currVal, \
        	self.ulimb[ecl].currVal, self.rwd.currVal, self.scale[ecl].currVal, self.az[ecl].currVal, \
        	self.fis[ecl].currVal, self.dexp[ecl].currVal, self.phi0[ecl].currVal]
        if self.complex:
            eclPars.extend([self.exp1[ecl].currVal, self.exp2[ecl].currVal, \
                self.tilt[ecl].currVal, self.yaw[ecl].currVal])
        return self.cv.calcFlux(eclPars,phi,width)
        
    def chisq(self,phi,y,e,width=None):
        retVal = 0.0
        for iecl in range(self.necl):
            if width:
                thisWidth=width[iecl]
            else:
                thisWidth=None
            resids = ( y[iecl] - self.calc(iecl,phi[iecl],thisWidth) ) / e[iecl]
            retVal += np.sum(resids**2)
        return retVal
        
    def ln_prior_base(self):
        retVal = 0.0
        priors_pars_shared = ['q','rwd']
        priors_pars_unique = ['wdFlux', 'dFlux', 'sFlux', 'rsFlux', 'ulimb', 'fis', 'dexp', 'phi0']
        if self.complex:
            priors_pars_unique.extend(['exp1','exp2','tilt','yaw'])
        for par in priors_pars_shared:
            param = getattr(self,par)
            if param.isVar:
                retVal += param.prior.ln_prob(param.currVal)
        for par in priors_pars_unique:
            parArr = getattr(self,par)
            for iecl in range(self.necl):
                param = parArr[iecl]
                if param.isVar:
                    retVal += param.prior.ln_prob(param.currVal)

        # just add in special cases here, e.g.:
        
		# dphi
		tol = 1.0e-6
		try:
			q = getattr(self,'q')
			dphi = getattr(self,'dphi')
			maxphi = roche.findphi(q.currVal,90.0) #dphi when i is slightly less than 90
			if dphi.currVal > maxphi:
				retVal += -np.inf
			else:
				retVal += dphi.prior.ln_prob(dphi.currVal)
		except:
			# we get here when roche.findphi raises error - usually invalid q
			retVal += -np.inf
        
        #Disc radius (XL1) 
    	try:
    		q = getattr(self,'q')
        	xl1 = roche.xl1(q) # xl1/a
        	rdisc = getattr(self,'rdisc')
        	maxrdisc = 0.46/xl1 # maximum size disc can be without precessing
        	if rdisc.currVal > maxrdisc:
        		retVal += -np.inf
        	else:
        		retVal += rdisc.prior.ln_prob(rdisc.currVal)
    	except:
        	# we get here when roche.findphi raises error - usually invalid q
        	retVal += -np.inf
        
        #BS scale (XL1)
    	rwd = getattr(self,'rwd')
    	scale = getattr(self,'scale')
    	minscale = rwd/3.
    	maxscale = rwd*3.
    	if scale < minscale or scale > maxscale:
    		retVal += -np.inf
    	else:
    		retVal += scale.prior.ln_prob(scale.currVal)
    		
    	#BS az
    	slop = 40.0
		try:
        	# find position of bright spot where it hits disc
        	# will fail if q invalid
        	q = getattr(self,'q')
        	xl1 = roche.xl1(q) # xl1/a
        	rdisc = getattr(self,'rdisc')
        	rd_a = rdisc*xl1
        	az = getattr(self,'az')
       		# Does stream miss disc? (disc/a < 0.2 or > 0.65 )
        	# if so, Tom's code will fail
        	x,y,vx,vy = roche.bspot(q,rd_a)
        	# find tangent to disc at this point
        	alpha = np.degrees(np.arctan2(y,x))
        	# alpha is between -90 and 90. if negative spot lags disc ie alpha > 90
        	if alpha < 0: alpha = 90-alpha
        	tangent = alpha + 90 # disc tangent
    		minaz = max(0,tangent-slop)
    		maxaz = min(178,tangent+slop)
    		if az < minaz or az > maxaz:
    			retVal += -np.inf
    		else:
    			retVal += az.prior.ln_prob(az.currVal)
    	except:
    		# we get here when roche.findphi raises error - usually invalid q
        	retVal += -np.inf
        	
        return retVal
        
    def ln_prior(self):
    	# this needs to be different in GPLCModel
    	return self.ln_prior_base()
 
    def ln_likelihood(self,parList,phi,y,e,width=None):
        lnlike = 0.0
        for iecl in range(self.necl):
            lnlike += np.sum( np.log (2.0*np.pi*e[iecl]**2) )
        return -0.5*(lnlike + self.chisq(phi,y,e,width))
        
    def ln_prob(self,parList,phi,y,e,width=None):
        lnp = self.ln_prior()
        if np.isfinite(lnp):
            try:
                return lnp + self.ln_likelihood(parList,phi,y,e,width)
            except:
                return -np.inf
        else:
            return lnp
            
class GPLCModel(LCModel):
	def __init__(self,parList,nel_disc=1000,nel_donor=400):
		LCModel.__init__(self,parList,nel_disc,nel_donor)
		# make sure GP params are variables
        self.amp_gp.isVar = True
        self.tau_gp.isVar = True
        
    def addEclipse(self,parList):
    	LCModel.addEclipse(self,parList)
    	
    def calc(self,ecl,phi,width=None):
    	LCModel.calc(self,ecl,phi,width)
    	
    def chisq(self,phi,y,e,width=None):
    	LCModel.chisq(self,phi,y,e,width)
    	
	def ln_prior_base(self):
		LCModel.ln_prior_base(self)
    	
	def ln_prior_gp(self):
		retVal=0.0
		priors_pars_shared = ['amp_gp','tau_gp']
		for par in priors_pars_shared:
            param = getattr(self,par)
            if param.isVar:
                retVal += param.prior.ln_prob(param.currVal)
		return retVal
		
	def ln_prior(self):
		return self.ln_prior_base() + self.ln_prior_gp()	
		
	def createGP(self,parList,phi):
    	a, tau = np.exp(parList[:2])
    	dphi, phiOff = parList[7],parList[15]
    
    	k_out = a*GP.Matern32Kernel(tau)
    	k_in  = 0.01*a*GP.Matern32Kernel(tau)
    
    	# Find location of all changepoints
    	changepoints = []
    	for n in range (int(phi[1]),int(phi[-1])+1,1):
        	changepoints.append(n-dphi/2.)
        	changepoints.append(n+dphi/2.)  

    	# Depending on number of changepoints, create kernel structure
    	kernel_struc = [k_out]    
    	for k in range (int(phi[1]),int(phi[-1])+1,1):
        	kernel_struc.append(k_in)
        	kernel_struc.append(k_out)
    
    	# create kernel with changepoints 
    	# obviously need one more kernel than changepoints!
    	kernel = GP.DrasticChangepointKernel(kernel_struc,changepoints)
    
    	# create GPs using this kernel
    	gp = GP.GaussianProcess(kernel)
    	return gp
        
    def ln_likelihood(self,parList,phi,y,e,width=None):
		# new ln_like function, using GPs, looping over each eclipse
		lnlike = 0.0
		for iecl in range(self.necl):
			gp = createGP(parList,phi)
    		gp.compute(phi,e)
    		resids = y - LCModel(parList[2:]).cv
    		
    	# check for bugs in model
    	if np.any(np.isinf(resids)) or np.any(np.isnan(resids)):
        print parList
        print 'Warning: model gave nan or inf answers'
        #raise Exception('model gave nan or inf answers')
        return -np.inf
        
        # now calculate ln_likelihood
        
        lnlike += gp.lnlikelihood(resids)
        
    	return -0.5*(lnlike + self.chisq(phi,y,e,width))
        
	'''def lnlike_gp(params, phi, width, y, e, cv): 
    gp = createGP(params,phi)
    gp.compute(phi,e)
    
    resids = y - model(params[2:],phi,width,cv)
    
    # check for bugs in model
    if np.any(np.isinf(resids)) or np.any(np.isnan(resids)):
        print params
        print 'Warning: model gave nan or inf answers'
        #raise Exception('model gave nan or inf answers')
        return -np.inf
 
    # now calculate ln_likelihood
    return gp.lnlikelihood(resids) '''
		
		########
		
		'''lnlike = 0.0
        for iecl in range(self.necl):
            lnlike += np.sum( np.log (2.0*np.pi*e[iecl]**2) )
        return -0.5*(lnlike + self.chisq(phi,y,e,width))'''
        
    def ln_prob(self,parList,phi,y,e,width=None):
    	LCModel.ln_prob(self,parList,phi,y,e,width)
    	
def parseInput(file):
        blob = np.loadtxt(file,dtype='string',delimiter='\n')
        input_dict = {}
        for line in blob:
                k,v = line.split('=')
                input_dict[k.strip()] = v.strip()
        return input_dict
            
if __name__ == "__main__":

    #Input lightcurve data from txt file
    import argparse
    parser = argparse.ArgumentParser(description='Fit CV lightcurves with lfit')
    parser.add_argument('file',action='store',help='input file')
    args = parser.parse_args()

    input_dict = parseInput(args.file)  

    nburn    = int( input_dict['nburn'] )
    nprod    = int( input_dict['nprod'] )
    nthreads = int( input_dict['nthread'] )
    nwalkers = int( input_dict['nwalkers'] )
    scatter  = float( input_dict['scatter'] )
    toFit    = int( input_dict['fit'] )
    
    neclipses = int( input_dict['neclipses'] )
    complex    = bool( int(input_dict['complex']) )
    useGP      = bool( int(input_dict['useGP']) )
    amp_gp = Param.fromString( input_dict['amp_gp'])
    tau_gp = Param.fromString( input_dict['tau_gp'])
    q      = Param.fromString( input_dict['q'] )
    dphi   = Param.fromString( input_dict['dphi'] )
    rwd    = Param.fromString( input_dict['rwd'] )
    
    files = []
    output_files = []
    fwd = []
    fdisc = []
    fbs = []
    fd  = []
    rdisc = []
    ulimb = []
    scale = []
    az = []
    frac = []
    rexp = []
    off = []
    exp1 = []
    exp2 = []
    tilt = []
    yaw = []
    for ecl in range(1,1+neclipses):
        files.append( input_dict['file_%d' % ecl] )
        output_files.append( input_dict['out_%d' % ecl] )
        fwd.append(Param.fromString( input_dict['fwd_%d' % ecl] ))
        fdisc.append(Param.fromString( input_dict['fdisc_%d' % ecl] ))
        fbs.append(Param.fromString( input_dict['fbs_%d' % ecl] ))
        fd.append(Param.fromString(input_dict['fd_%d' % ecl] ))
        rdisc.append(Param.fromString(input_dict['rdisc_%d' % ecl] ))
        ulimb.append(Param.fromString(input_dict['ulimb_%d' % ecl] ))
        scale.append(Param.fromString(input_dict['scale_%d' % ecl] ))
        az.append(Param.fromString(input_dict['az_%d' % ecl] ))
        frac.append(Param.fromString(input_dict['frac_%d' % ecl] ))
        rexp.append(Param.fromString(input_dict['rexp_%d' % ecl] ))
        off.append(Param.fromString(input_dict['off_%d' % ecl] ))
        if complex:
            exp1.append(Param.fromString(input_dict['exp1_%d' % ecl] ))
            exp2.append(Param.fromString(input_dict['exp2_%d' % ecl] ))
            tilt.append(Param.fromString(input_dict['tilt_%d' % ecl] ))
            yaw.append(Param.fromString(input_dict['yaw_%d' % ecl] ))

    # OUTPUT FILE CODE
    # setup header for output file format
    outfile_header = """#This file contains the data and best fit. 
    #The first three columns are the data (x, y and y error)
    #The next column is the CV flux
    #The next columns are the flux from wd, bright spot, disc and donor
    """

    # create a model from the first eclipses parameters
    parList = [amp_gp,tau_gp,fwd[0],fdisc[0],fbs[0],fd[0],q,dphi,rdisc[0],ulimb[0],rwd, \
                scale[0],az[0],frac[0],rexp[0],off[0]]
    if complex:
        parList.extend([exp1[0],exp2[0],tilt[0],yaw[0]])
    if useGP:
		model = GPLCModel(parList)
    else:
		model = LCModel(parList)

    # then add in additional eclipses as necessary
    for ecl in range(1,neclipses):
        parList = [fwd[ecl],fdisc[ecl],fbs[ecl],fd[ecl],rdisc[ecl],ulimb[ecl], \
                    scale[ecl],az[ecl],frac[ecl],rexp[ecl],off[ecl]]
        if complex:
            parList.extend([exp1[ecl],exp2[ecl],tilt[ecl],yaw[ecl]])
        model.addEclipse(parList)
        
    # store your data in python lists, so that x[0] are the times for eclipse 0, etc.
    x = []
    y = []
    e = []
    w = []
    
    # crop to range if required
    start = float( input_dict['phi_start'] )
    end = float( input_dict['phi_end'] )
    for file in files:
        xt,yt,et = np.loadtxt(file,skiprows=16).T
        wt = np.mean(np.diff(xt))*np.ones_like(xt)/2.
        #xt,wt,yt,et,_ = np.loadtxt(file).T
        mask = (xt>start)&(xt<end)
     	x.append(xt[mask])
     	y.append(yt[mask])
     	e.append(et[mask])
     	w.append(wt[mask])
        
    
    # is our starting position legal?
    if np.isinf( model.ln_prior() ):
        print 'Error: starting position violates priors'
        sys.exit(-1)
        
    npars = model.npars
    params = [par for par in model]
    
    def ln_prob(pars,model,x,y,e,w):
        # we update the model to contain the params suggested by emcee, and calculate lnprob
        for i in range(model.npars):
            model[i] = pars[i]
        return model.ln_prob(x,y,e,w)

    if toFit:
        # Initialize the MPI-based pool used for parallelization.
        # MPI not used as found to not be quicker
        '''
        pool = MPIPool()

        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)
        '''    
        p0 = np.array(params)
        
        print ln_prob(p0,model,x,y,e,w)
        p0 = emcee.utils.sample_ball(p0,scatter*p0,size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers,npars,ln_prob,args=[model,x,y,e,w],threads=nthreads)

        #Burn-in
        print 'starting burn-in'
        pos, prob, state = run_burnin(sampler,p0,nburn)

        # run second burn-in scattered around best fit of previous burnin
        # DFM reports this can help convergence in difficult cases
        print 'starting second burn-in'
        p0 = pos[np.argmax(prob)]
        p0 = emcee.utils.sample_ball(p0,scatter*p0,size=nwalkers)
        pos, prob, state = run_burnin(sampler,p0,nburn)

        #Production
        sampler.reset()
        print 'starting main mcmc chain'
        sampler = run_mcmc_save(sampler,pos,nprod,state,"chain2.txt")  
        
        # stop parallelism
        #pool.close()
        
        chain = flatchain(sampler.chain,npars,thin=10)
        
        # Print out parameters
        
        pars_shared = ['amp_gp','tau_gp','q','dphi','_','_','rwd']
        pars_unique = ['_','_''wdFlux', 'dFlux', 'sFlux', 'rsFlux','_','_','rdisc','ulimb','_','scale', 'az', 'fis', 'dexp', 'phi0']
        pars_unique_2 = ['wdFlux', 'dFlux', 'sFlux', 'rsFlux', 'rdisc', 'ulimb', 'scale', 'az', 'fis', 'dexp', 'phi0']
        if complex:
            pars_unique.extend(['exp1','exp2','tilt','yaw'])
            pars_unique_2.extend(['exp1','exp2','tilt','yaw'])
        pars_unique += pars_unique_2*(neclipses-1)
        
        params = []
        
        print "\nShared params:\n"
        p = 0
        for i in range(0,7):
        	if p == 2 or p == 3 or p == 4 or p == 5 or p == 8 or p == 9:
                    p += 1
        	else:
        		par = chain[:,p]
        		lolim,best,uplim = np.percentile(par,[16,50,84])
        		print "%s = %f +%f -%f" % (pars_shared[i],best,uplim-best,best-lolim)
        		p += 1
        		'''params.append(best)
        		model[i] = best'''
                
        ind_npars = (npars-5)/neclipses
        a = 0 
        b = 5     
        
        for iecl in range(neclipses):
            print "\nindividual params for eclipse %d:\n" % (iecl+1)
            p = 0+(iecl*ind_npars)
            a += (ind_npars*iecl)
            b += (a + ind_npars)
            if a <= npars:
				for i in range(a,b):
					if p == 0 or p == 1 or p == 6 or p == 7 or p == 10:
						p += 1
					else:
						if iecl == 0:
							par = chain[:,p]
							lolim,best,uplim = np.percentile(par,[16,50,84])
							print "%s = %f +%f -%f" % (pars_unique[i],best,uplim-best,best-lolim)
						else:
							par = chain[:,p+5] 
							lolim,best,uplim = np.percentile(par,[16,50,84])
							print "%s = %f +%f -%f" % (pars_unique[i],best,uplim-best,best-lolim)
						p += 1	  
				a = 5
				b = 0
            '''params.append(best)
            model[i] = best'''
                         
    print '\nFor this model:\n'
    dataSize = np.sum((xa.size for xa in x))
    print "Chisq          =  %.2f (%d D.O.F)" % (model.chisq(x,y,e,w),dataSize - model.npars - 1)
    print "ln probability = %.2f" % model.ln_prob(x,y,e,w)
    print "ln prior       = %.2f" % model.ln_prior()
    
    # Plot model & data
    gs = gridspec.GridSpec(2,neclipses,height_ratios=[2,1])
    gs.update(hspace=0.0)
    seaborn.set()

    LHplot = True
    for iecl in range(neclipses):
        xp = x[iecl]
        yp = y[iecl]
        ep = e[iecl]
        wp = w[iecl]
           
        xf = np.linspace(xp.min(),xp.max(),1000)
        wf = 0.5*np.mean(np.diff(xf))*np.ones_like(xf)
        yp_fit = model.calc(iecl,xp,wp)
        yf = model.calc(iecl,xf,wf)

        ax1 = plt.subplot(gs[0,iecl])
        
        # CV model
        ax1.plot(xf,yf)
        ax1.plot(xf,model.cv.yrs)
        ax1.plot(xf,model.cv.ys)
        ax1.plot(xf,model.cv.ywd)
        ax1.plot(xf,model.cv.yd)
        # data
        
        ax1.errorbar(xp,yp,yerr=ep,fmt='.',color='k',capsize=0,alpha=0.5)

        ax2 = plt.subplot(gs[1,iecl],sharex=ax1)
        ax2.errorbar(xp,yp-yp_fit,yerr=ep,color='k',fmt='.',capsize=0,alpha=0.5)
        #ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlim(-0.1,0.15)

        #labels
        if LHplot:
               ax1.set_ylabel('Flux (mJy)')
               ax2.set_ylabel('Residuals (mJy)')
               LHplot = False
        ax2.set_xlabel('Orbital Phase')
        ax2.yaxis.set_major_locator(MaxNLocator(4,prune='both'))
        
    for ax in plt.gcf().get_axes()[::2]:
        ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
        
        
    plt.savefig('bestFit.pdf')
    plt.xlim(-0.1,0.15)
    plt.show()
     
