import numpy as np
import scipy.stats as ss
from numpy import Inf
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import sys
import pyEXP
from schwimmbad import SerialPool,MultiPool
##  exp
#sys.path.append("/u/svarel/exp/build/utils/Analysis/")
from spherical_basis_builder import *
#import simpleSL

## Auriga
import LibAu as la
#import warnings
#warnings.filterwarnings('ignore')}




def EBF(R,basis,idsubhalo,nsnap):
    lrmin = np.log10(np.nanmin(R))
    lrmax = np.log10(np.nanmax(R))
    rnum  = len(R) 
    basis_grid = basis.getBasis(lrmin, lrmax, rnum)


    r = np.linspace(lrmin, lrmax, rnum)
    r = np.power(10.0, r)


    for l in range(6):
        for n in range(10):
            plt.semilogx(r, basis_grid[l][n], '-', label="n={}".format(n))
        plt.xlabel('r')
        plt.ylabel('potential')
        plt.title('l={}'.format(l))
        plt.legend()
        plt.savefig('plots/'+idsubhalo+'_Snap'+nsnap+'_l'+str(l)+'.png')
        plt.show()
        
        
nhalo = 16
Lsnap = np.arange(39,64,1,dtype=int)
colors = pl.cm.Blues(np.linspace(0,1,len(Lsnap)))[::-1]


RL,DL = [],[]
Datapos = {}
for ii,ns in enumerate(Lsnap):
    if ns not in (63,40): continue
    print('Au-%s snapshot %s'%(nhalo,ns))
    
    sim = la.Reader_Au(Nhalo=nhalo,Nsnap=ns)
    #print(sim.base)
    
    header = sim.Header()
    h=header['hubbleparam']
    sc=header['time']
    
    Rvir =sim.sf.data['frc2'][0]*1000*sc/h #[kpc] 
    Mvir =sim.sf.data['fmc2'][0]*1e10/h    #[Msun]
    
    try:
        fields_stars = ['pos','vel','id','mass','type','pot','age']
        Datstars = sim.Snapshot_Subhalo(idSubhalo=0,snapType='stars',fields=fields_stars)
    except KeyError:
        fields_stars = ['pos','vel','id','mass','type','pot']
        Datstars = sim.Snapshot_Subhalo(idSubhalo=0,snapType='stars',fields=fields_stars)

    fields_dm = ['pos','vel','id','mass','type','pot']
    DatDM = sim.Snapshot_Subhalo(idSubhalo=0,snapType='dm1',fields=fields_dm)
    
    #Rotate Halo---------------------------------------------------------------------
    Data = {'stars':Datstars,'dm1':DatDM}
    param = {'spos':sim.sf.data['spos'][0,:],'svel':sim.sf.data['svel'][0,:],'header':sim.Header()}
    gal = la.ToolRot(Data=Data, param=param)
    Data = gal.Centered()#gal.Rotate()

    Datstars=Data['stars']
    DatDM = Data['dm1']
    #--------------------------------------------------------------------------------

    potdm = np.float64(DatDM['pot'])
    pos = np.float64(DatDM['pos']) #part_rot[not_in_subs]
    mass = np.float64(np.ones_like(pos[:,0]))#DatDM['mass']  #part['dark']['mass'][not_in_subs]

    poss,masss=Datstars['pos'],Datstars['mass']
    #print(type(mass[0]))
    
    rr = np.sqrt((pos[:,0]**2) + (pos[:,1]**2) + (pos[:,2]**2))
    
    binrad = 200
    
    rrmin,rrmax=np.nanmin(np.log10(rr)),np.nanmax(np.log10(rr))
    r,rho = return_density(np.log10(rr),weights= 1., rangevals=[rrmin, rrmax],bins=binrad)
    
    
    R,D,M,P = makemodel_empirical(r, rho, "Au{}_table.txt".format(nhalo))
    
    plt.plot(R,np.log10(D),color=colors[ii],lw=1)#,zorder=ns)
    RL.append(R)
    DL.append(D)
    if ns in (63,40):Datapos['Snap'+str(ns)]=pos

RL,DL = np.array(RL),np.array(DL)



R0,D0,M0,P0 = makemodel_empirical(RL[-1], DL[-1], "Au{}_Snap{}_table.txt".format(nhalo,63))
config0="""
---
id: sphereSL
parameters :
  numr: %s
  rmin: %s
  rmax: %s
  Lmax: 6
  nmax: 20
  modelname: %s_%s_table.txt
...
"""%(len(R0),round(np.nanmin(R0),3), round(np.nanmax(R0),3),'Au'+str(nhalo),'Snap'+str(63))
print(config0)


R39,D39,M39,P39 = makemodel_empirical(RL[0], DL[0], "Au{}_Snap{}_table.txt".format(nhalo,39))
config39="""
---
id: sphereSL
parameters :
  numr: %s
  rmin: %s
  rmax: %s
  Lmax: 6
  nmax: 20
  modelname: %s_%s_table.txt
...
"""%(len(R39),round(np.nanmin(R39),3), round(np.nanmax(R39),3),'Au'+str(nhalo),'Snap'+str(39))
print(config39)

def func(rs,ii,ns):
    basis = pyEXP.basis.Basis.factory(ii)
    EBF(rs,basis,'Au16',ns)
for ii,rs,ns in [(config0,R0,str(63)),(config39,R39,str(39))]:
    func(rs,ii,ns)
    #del basis