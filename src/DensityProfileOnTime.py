import numpy as np
import scipy.stats as ss
from numpy import Inf
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import sys
import pyEXP
from schwimmbad import SerialPool,MultiPool
from scipy.stats import binned_statistic_2d, binned_statistic
from datetime import datetime
##  exp
#sys.path.append("/u/svarel/exp/build/utils/Analysis/")
from spherical_basis_builder import *
#import simpleSL

## Auriga
import LibAu as la
import warnings
warnings.filterwarnings('ignore')
print(datetime.now())
  

config= lambda numr,rmin,rmax,nt: """
---
id: sphereSL
parameters :
  numr: %s
  rmin: %s
  rmax: %s
  Lmax: 6
  nmax: 20
  modelname: %s
...
"""%(numr,rmin,rmax,nt);


km2kpc=3.24078e-17
G= 6.674*1e-11 #m3⋅kg−1⋅s−2
G_=4.300917270036279e-06#
m2kpc, kg2Msun = 3.2408*1e-20,5e-31
G=G*(m2kpc**3) /kg2Msun # kpc3 Msun-1 s-2
#=================================================================

nhalo = la.L3MHDsam[0]
Lsnap = np.arange(39,64,1,dtype=int)
colors = pl.cm.Reds(np.linspace(0,1,len(Lsnap)))[::-1]



'''plt.figure(figsize=(8,6))
plt.xlabel('r',fontsize=14)
plt.ylabel(r'$log\;\rho$',fontsize=14)
plt.title('Au%s from makemodel_empirical function'%nhalo)'''

for ii,ns in enumerate(Lsnap[::-1]):
    #if ns not in (63,55): continue
    sim = la.Reader_Au(Nhalo=nhalo,Nsnap=ns)
    header = sim.Header()
    h=header['hubbleparam']
    sc=header['time']

    verb='Au-%s => snapshot %s z %s '%(nhalo,ns,sim.sf.redshift)
    la.PrintPercent(ii,Lsnap.size,'fraper',text=verb)

    Rvir =sim.sf.data['frc2'][0]*1000*sc/h #[kpc]
    Mvir =sim.sf.data['fmc2'][0]*1e10/h    #[Msun]

    fields_stars = ['pos','vel','id','mass','type','pot','age']
    Datstars = sim.Snapshot_Subhalo(idSubhalo=0,snapType='stars',fields=fields_stars,CalcStellarAges='CalcAgesAlt')
    fields_dm = ['pos','vel','id','mass','type','pot']
    DatDM = sim.Snapshot_Subhalo(idSubhalo=0,snapType='dm1',fields=fields_dm)
    
    verb='Au-%s => snapshot %s AgeU %s'%(nhalo,ns,sim.AgeU)
    la.PrintPercent(ii,Lsnap.size,'fraper',text=verb)

    #Rotate Halo---------------------------------------------------------------------
    Data = {'stars':Datstars,'dm1':DatDM}
    param = {'spos':sim.sf.data['spos'][0,:],'svel':sim.sf.data['svel'][0,:],'header':sim.Header()}
    gal = la.ToolRot(Data=Data, param=param)
    Data = gal.Rotate()#gal.Centered()


    Datstars=Data['stars']
    DatDM = Data['dm1']
    #--------------------------------------------------------------------------------

    potdm = np.float64(DatDM['pot'])
    pos = np.float64(DatDM['pos']) #part_rot[not_in_subs]
    mass = np.float64(DatDM['mass'] )# #part['dark']['mass'][not_in_subs]

    poss,masss=Datstars['pos'],Datstars['mass']
    #print(type(mass[0]))
    DMmass = np.max(mass)
    rr = np.sqrt((pos[:,0]**2) + (pos[:,1]**2) + (pos[:,2]**2))
    #rbord = np.argsort(rr)
    #rb,mb,pwau = rr[rbord],mass[rbord],potdm[rbord]
    #posb = pos[rbord]

    binrad = 200

    rrmin,rrmax=np.nanmin(np.log10(rr)),np.nanmax(np.log10(rr))
    r,rho = return_density(np.log10(rr),weights= DMmass, rangevals=[rrmin, rrmax],bins=binrad)

    if  ns==63:
        nametab = "Au{}_Snap{}_table.txt".format(nhalo,63)
        #rad,rho = empirical_density_profile(pos,mb, nbins=500, rmin=np.min(R), rmax=np.max(R))
        #R0,D0,M0,P0,scal = makemodel_empirical(r, rho, nametab,GetScaling=True)
        '''config_=config(len(R0),round(np.nanmin(R0),3), round(np.nanmax(R0),3),nametab)
        print(config_)
        # Construct the basis instances
        #
        basis = pyEXP.basis.Basis.factory(config_)
        
        lrmin = np.log10(np.nanmin(R0))
        lrmax = np.log10(np.nanmax(R0))
        rnum  = len(R0) 
        basis_grid = basis.getBasis(lrmin, lrmax, rnum)


        r = np.linspace(lrmin, lrmax, rnum)
        r = np.power(10.0, r)'''
        
        #---------------------
        #plt.plot(R0,np.log10(D0),color=colors[ii],zorder=ns)
        plt.plot(r,rho,color=colors[ii],zorder=ns)
        
    if ns==63: continue

    
    #R,D,M,P = makemodel_empirical(r, rho, "Au{}_Snap{}_table.txt".format(nhalo,ns))
        
    plt.plot(r,rho,color=colors[ii],zorder=ns)
    plt.text(r[-1],rho[-1],str(ns))
plt.xscale('log')    
plt.savefig('plots/check63vsT_basis/Au%s_rho-r.png'%nhalo)
plt.close()
        