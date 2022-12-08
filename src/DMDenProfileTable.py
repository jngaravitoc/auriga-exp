import numpy as np
import scipy.stats as ss
from numpy import Inf
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import sys
import pyEXP
import h5py
##  exp
#sys.path.append("/u/svarel/exp/build/utils/Analysis/")
from spherical_basis_builder import *
#import simpleSL

## Auriga
import LibAu as la

hf = h5py.File('Table/DMDenProfileTable.h5py','w')
for nhalo in la.L3MHDsam:
    #nhalo = 16
    Lsnap = np.arange(39,64,1,dtype=int)
    colors = pl.cm.Blues(np.linspace(0,1,len(Lsnap)))[::-1]

    plt.figure(figsize=(8,6))
    RL,DL = [],[]
    for ii,ns in enumerate(Lsnap):

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

        #plt.plot(R,np.log10(D),color=colors[ii],lw=1,zorder=ns)
        RL.append(R)
        DL.append(D)

    RL,DL = np.array(RL),np.array(DL)
    hf.create_dataset( "Au{}".format(nhalo)   ,data=[RL,DL])
hf.close()
