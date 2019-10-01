'''Uses the output fluxes of a White Dwarf from an MCMC chain, and uses them
to calculate the limb darkening coefficient that should be used when fitting
that body. Data is typically not sensitive enough to this to fit it
independantly, so a few iterations through this script should aid in the
science.
'''


from __future__ import print_function

from builtins import input, zip

import numpy as np
from scipy.interpolate import RectBivariateSpline, SmoothBivariateSpline


def ld (band,logg,teff,law='linear'):
    if not band in ['u','g','r','i','z']:
        raise NotImplementedError("Band {} not recognised!".format(band))

    if not law in ['linear','quad','sqr']:
        raise NotImplementedError("Law {} not implimented!".format(law))
    filename = '/'.join(__file__.split('/')[:-1])
    filename += '/Gianninas13/ld_coeffs_%s.txt' % band
    data=np.loadtxt(filename)
    x=data[:,0] #logg (grid increments through all teffs at a single logg, then +logg)
    y=data[:,1] #teff
    t1 = np.unique(y) # unique temps
    g1 = np.unique(x) # unique loggs
    nt = len(t1)
    ng = len(g1)

    z0=data[:,2] #linear ld coefficient
    z1=data[:,3] # first quad term
    z2=data[:,4] # 2nd quad term
    z3=data[:,5] # first square-root term
    z4=data[:,6] # second square-root term

    #func = SmoothBivariateSpline(x,y,z)
    #return func(logg,teff)[0]
    if law == 'linear':
        func = RectBivariateSpline(g1,t1,z0.reshape((ng,nt)),kx=3,ky=3)
        return func(logg,teff)[0,0]
    elif law == 'quad':
        funca = RectBivariateSpline(g1,t1,z1.reshape((ng,nt)),kx=3,ky=3)
        funcb = RectBivariateSpline(g1,t1,z2.reshape((ng,nt)),kx=3,ky=3)
        return (funca(logg,teff)[0,0],funcb(logg,teff)[0,0])
    elif law == 'sqr':
        funca = RectBivariateSpline(g1,t1,z3.reshape((ng,nt)),kx=3,ky=3)
        funcb = RectBivariateSpline(g1,t1,z4.reshape((ng,nt)),kx=3,ky=3)
        return (funca(logg,teff)[0,0],funcb(logg,teff)[0,0])

def main():
    logg, gerr = input('> Give log g and error: ').split()
    teff, terr = input('> Give eff. temp. and error: ').split()
    logg = float(logg); gerr = float(gerr)
    teff = float(teff); terr = float(terr)

    gvals=np.random.normal(loc=logg,scale=gerr,size=100)
    tvals=np.random.normal(loc=teff,scale=terr,size=100)

    #ldvals = []
    #for g,t in zip(gvals,tvals):
    #    ldvals.extend( ld('i',g,t) )
    print('-------------------')
    for band in ['u','g','r','i','z']:
        ldvals = [ld(band,g,t) for g,t in zip(gvals,tvals)]
        print('%s band LD coeff = %f +/- %f' % (band, np.median(ldvals),np.std(ldvals)))

        ldvals = [ld(band,g,t,law='quad') for g,t in zip(gvals,tvals)]
        # unpack list of tuples into two lists
        a, b = list(zip(*ldvals)) # use splat operator to expand list into positional arguments
        print('%s band quad coeff a = %f +/- %f' % (band, np.median(a),np.std(a)))
        print('%s band quad coeff b = %f +/- %f' % (band, np.median(b),np.std(b)))

        ldvals = [ld(band,g,t,law='sqr') for g,t in zip(gvals,tvals)]
        # unpack list of tuples into two lists
        a, b = list(zip(*ldvals)) # use splat operator to expand list into positional arguments
        print('%s band sqr coeff d = %f +/- %f' % (band, np.median(a),np.std(a)))
        print('%s band sqr coeff f = %f +/- %f' % (band, np.median(b),np.std(b)))
        print('-------------------')

if __name__ == "__main__":
    main()
