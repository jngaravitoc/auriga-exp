#!/usr/bin/env python
# coding: utf-8

# # Building a basis
# ### Mike Petersen, 18 May
# 
# Updates since the last time we built a spherical basis.

# Old tools:
# 1. makemodel (slight upgrade with addition of makemodel_empirical)
# 2. haloprof
# 3. (slcheck, other tools for basis inspection) 

# The tutorial material mostly holds; I'm going to introduce two new tools:
# 1. simpleSL (Python tool)
# 2. modelfit (smoothing strategy for empirical bases)

# ### Step 1
# 
# 1. Install **EXP**: review?
# 2. export PYTHONPATH=$HOME/lib/python3.8 (or where you installed it)
# 3. python -> import simpleSL

# ### Some definitions


# standard python modules
import numpy as np
import time
import copy

# plotting utilities
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy.interpolate as interpolate
import subprocess
#from astropy.io import fits


#G= 6.674*1e-11 #m3⋅kg−1⋅s−2
#m2kpc, kg2Msun = 3.2408*1e-20,5e-31
#G=G*(m2kpc**3) /kg2Msun # kpc3 Msun-1 s-2

def return_density(logr,weights=1.,rangevals=[-2, 6],bins=500,d2=False):
    """return_density
    
    simple binned density using logarithmically spaced bins
    
    inputs
    ---------
    logr        : (array) log radii of particles to bin
    weights     : (float or array) if float, single-mass of particles, otherwise array of particle masses
    rangevals   : (two value list) minimum log r, maximum log r
    bins        : (int) number of bins
    d2          : (bool) if True, compute surface density
    
    returns
    ---------
    rcentre     : (array) array of sample radii (NOT LOG)
    density     : (array) array of densities sampled at rcentre (NOT LOG)
    
    """
    
    # assume evenly spaced logarithmic bins
    dr      = (rangevals[1]-rangevals[0])/bins
    rcentre = np.zeros(bins)
    density = np.zeros(bins)
    
    # check if single mass, or an array of masses being passed
    # construct array of weights
    if isinstance(weights,np.float):
        w = weights*np.ones(logr.size)
    else:
        w = weights
    
    for indx in range(0,bins):
        
        # compute the centre of the bin (log r)
        rcentre[indx] = rangevals[0] + (indx+0.5)*dr
        
        # compute dr (not log)
        rmin,rmax = 10.**(rangevals[0] + (indx)*dr),10.**(rangevals[0] + (indx+1)*dr)
        if d2:
            shell = np.pi*(rmax**2-rmin**2)
        else:
            shell = (4./3.)*np.pi*(rmax**3.-rmin**3.)
            
        # find all particles in bin
        inbin = np.where((logr>=(rangevals[0] + (indx)*dr)) & (logr<(rangevals[0] + (indx+1)*dr)))
        
        # compute M/V for the bin
        density[indx] = np.nansum(w[inbin])/shell
        
    # return
    return 10.**rcentre,density



def makemodel(func,M,funcargs,rvals = 10.**np.linspace(-2.,4.,2000),pfile='',plabel = '',verbose=True):
    """make an EXP-compatible spherical basis function table
    
    inputs
    -------------
    func        : (function) the callable functional form of the density
    M           : (float) the total mass of the model, sets normalisations
    funcargs    : (list) a list of arguments for the density function.
    rvals       : (array of floats) radius values to evaluate the density function
    pfile       : (string) the name of the output file. If '', will not print file
    plabel      : (string) comment string
    verbose     : (boolean)
    outputs
    -------------
    R           : (array of floats) the radius values
    D           : (array of floats) the density
    M           : (array of floats) the mass enclosed
    P           : (array of floats) the potential
    
    """


    R = np.nanmax(rvals)
    
    # query out the density values
    dvals = func(rvals,*funcargs)

    # make the mass and potential arrays
    mvals = np.zeros(dvals.size)
    pvals = np.zeros(dvals.size)
    pwvals = np.zeros(dvals.size)

    # initialise the mass enclosed an potential energy
    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed and potential energy by recursion
    for indx in range(1,dvals.size):
        mvals[indx] = mvals[indx-1] +          2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] +                 rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
        pwvals[indx] = pwvals[indx-1] +           2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
    
    # evaluate potential (see theory document)
    pvals = -mvals/(rvals+1.e-10) - (pwvals[dvals.size-1] - pwvals)

    # get the maximum mass and maximum radius
    M0 = mvals[dvals.size-1]
    R0 = rvals[dvals.size-1]

    # compute scaling factors
    Beta = (M/M0) * (R0/R);
    Gamma = np.sqrt((M0*R0)/(M*R)) * (R0/R);
    if verbose:
        print("! Scaling:  R=",R,"  M=",M)

    rfac = np.power(Beta,-0.25) * np.power(Gamma,-0.5);
    dfac = np.power(Beta,1.5) * Gamma;
    mfac = np.power(Beta,0.75) * np.power(Gamma,-0.5);
    pfac = Beta;

    if verbose:
        print(rfac,dfac,mfac,pfac)

    # save file if desired
    if pfile != '':
        f = open(pfile,'w')
        print('! ',plabel,file=f)
        print('! R    D    M    P',file=f)

        print(rvals.size,file=f)

        for indx in range(0,rvals.size):
            print('{0} {1} {2} {3}'.format( rfac*rvals[indx],              dfac*dvals[indx],              mfac*mvals[indx],              pfac*pvals[indx]),file=f)
    
        f.close()
    
    return rvals*rfac,dfac*dvals,mfac*mvals,pfac*pvals


def makemodel_empirical(rvals,dvals,pfile='',plabel = '',verbose=True,M=1.):
    """make an EXP-compatible spherical basis function table
    
    inputs
    -------------
    rvals       : (array of floats) radius values to evaluate the density function
    pfile       : (string) the name of the output file. If '', will not print file
    plabel      : (string) comment string
    verbose     : (boolean)
    outputs
    -------------
    R           : (array of floats) the radius values
    D           : (array of floats) the density
    M           : (array of floats) the mass enclosed
    P           : (array of floats) the potential
    
    """
    #M = 1.
    R = np.nanmax(rvals)
    
    # query out the density values
    #dvals = D#func(rvals,*funcargs)
    #print(R.size,)

    # make the mass and potential arrays
    mvals = np.zeros(dvals.size)
    pvals = np.zeros(dvals.size)
    pwvals = np.zeros(dvals.size)

    # initialise the mass enclosed an potential energy
    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed and potential energy by recursion
    for indx in range(1,dvals.size):
        mvals[indx] = mvals[indx-1] +          2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] +                 rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
        pwvals[indx] = pwvals[indx-1] +           2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
    
    # evaluate potential (see theory document)
    pvals = -mvals/(rvals+1.e-10) - (pwvals[dvals.size-1] - pwvals)

    # get the maximum mass and maximum radius
    M0 = mvals[dvals.size-1]
    R0 = rvals[dvals.size-1]

    # compute scaling factors
    Beta = (M/M0) * (R0/R);
    Gamma = np.sqrt((M0*R0)/(M*R)) * (R0/R);
    if verbose:
        print("! Scaling:  R=",R,"  M=",M,"  M0=",M0,"  R0=",R0)

    rfac = np.power(Beta,-0.25) * np.power(Gamma,-0.5);
    dfac = np.power(Beta,1.5) * Gamma;
    mfac = np.power(Beta,0.75) * np.power(Gamma,-0.5);
    pfac = Beta;

    if verbose:
        print(rfac,dfac,mfac,pfac)

    # save file if desired
    if pfile != '':
        f = open(pfile,'w')
        print('! ',plabel,file=f)
        print('! R    D    M    P',file=f)

        print(rvals.size,file=f)

        for indx in range(0,rvals.size):
            print('{0} {1} {2} {3}'.format( rfac*rvals[indx],              dfac*dvals[indx],              mfac*mvals[indx],              pfac*pvals[indx]),file=f)
    
        f.close()
    
    return rvals*rfac,dfac*dvals,mfac*mvals,pfac*pvals


# ### Second strategy: Making an empirical basis
# 
# compute 3d radius
#R = np.sqrt((hdul[1].data['X'])**2.+(hdul[1].data['Y'])**2.+((hdul[1].data['Z'])**2.))
#print(np.nanmin(R),np.nanmax(R))


#R,D,M,P = makemodel_empirical(rbins,dreturn,pfile='GSEbasis_empirical.txt')



# ### Build the basis!
#import simpleSL

#E = simpleSL.slfunctions('/home/mpetersen/GSEbasis_empirical.txt',2,6,0.,2.,2000)
# In[25]:


#from exptool.utils import halo_methods

#sph_file = '.slGSE_cache'
#mod_file = 'GSEbasis_empirical.txt'

#lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file)
#xi,rarr,p0,d0 = halo_methods.init_table(mod_file,numr,rmin,rmax,cmap,scale)


# In[30]:


# plot the first 5 potential functions
#plt.figure(figsize=(6,4))
#for n in range(0,5): plt.plot(np.log10(rarr),eftable[0][n]*p0,color=cm.viridis(n/4.,1.))
#plt.xlabel('log radius (kpc)')
#plt.ylabel('function value')

# modelfit --data=/disk01/mpetersen/Disk080/SLGridSph.NFW77 --type=TwoPowerTrunc --iterations=100
# 
# 
# 

# ### workflow examples
# 
# These are specific to my machine, so approach with care!
#mport numpy as np;import matplotlib.pyplot as plt

#mport simpleSL

#E = simpleSL.slfunctions('/home/mpetersen/GSEbasis_empirical.txt',2,6,0.,2.,2000)
#xvals = 10.**(np.linspace(0.,2.,2000))

# plot the first 5 potential functions
#or n in range(0,5): 
#    plt.plot(xvals,E[0][n],color='black')
#plt.savefig('/home/mpetersen/testfig.png')



# plot the first 5 potential functions
#for n in range(0,5): plt.plot(xvals,E[0][n],color='black')

#plt.savefig('/home/mpetersen/testfig.png')



#E = simpleSL.coefsl(O.mass,O.xpos-np.nanmean(O.xpos),O.ypos-np.nanmean(O.ypos),O.zpos-np.nanmean(O.zpos),'/disk01/mpetersen/Disk080/SLGridSph.NFW77',2,10)

# this workflow will also save the cache
#mpirun haloprof  --LMAX=4 --NMAX=16 --MODFILE=/disk01/mpetersen/Disk080/SLGridSph.NFW77 --dir=/disk01/mpetersen/Disk080/ --beg=0 --end=1 --prefix=OUT  --filetype=PSPout --RMAX=1 --RSCALE=0.067 --CONLY -v --runtag=system1_3m --compname="mw"
