import numpy as np
from numpy import Inf
import sys

#from ctypes import *
#simpleSL=cdll.LoadLibrary('/u/svarel/exp/build/utils/Analysis/simpleSL.cpython-36m-x86_64-linux-gnu.so')

##  exp
sys.path.append("/u/svarel/exp/build/utils/Analysis/")
from spherical_basis_builder import *
import simpleSL


## Auriga
import LibAu as la
 



if __name__ == "__main__":
    nhalo=21 #Name Halo
    nsnap= 63 #z=0
    sim = la.Reader_Au(Nhalo=nhalo,Nsnap=nsnap) #Read Simulation
    header = sim.Header()
    h=header['hubbleparam']
    sc=header['time']

    Rvir =sim.sf.data['frc2'][0]*1000*sc/h

    fields_stars = ['pos','vel','id','mass','type','pot','age']
    Datstars = sim.Snapshot_Subhalo(idSubhalo=0,snapType='stars',fields=fields_stars)


    fields_dm = ['pos','vel','id','mass','type','pot']
    DatDM = sim.Snapshot_Subhalo(idSubhalo=0,snapType='dm1',fields=fields_dm)
    #Rotate Halo
    Data = {'stars':Datstars,'dm1':DatDM}
    param = {'spos':sim.sf.data['spos'][0,:],'svel':sim.sf.data['svel'][0,:],'header':sim.Header()}
    gal = la.ToolRot(Data=Data, param=param)
    Data = gal.Rotate()

    Datstars=Data['stars']
    DatDM = Data['dm1']

    #*************************  Compute BFE: *****************************
    
    pos = DatDM['pos'] #part_rot[not_in_subs]
    mass = DatDM['mass']  #part['dark']['mass'][not_in_subs]
    print(np.sum(mass), mass[0], len(pos), len(mass))
    M_tot = np.sum(mass)

    #selecting 1million random particles
    ind = np.arange(0,len(mass),1,dtype=int)
    sel = np.random.choice(ind, 1000000, replace=False).astype(int)
    pos,mass = pos[sel],mass[sel]

    rr = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    rbins,dreturn = return_density(np.log10(rr), 1., rangevals=[0., 2.5],bins=100)
    R,D,M,P = makemodel_empirical(rbins, dreturn, pfile='m12b_SPHLbasis_empirical.txt')

    # Build the basis
    ebf = simpleSL.slfunctions('./m12b_SPHLbasis_empirical.txt',2,6,0.,2.5, 2000)

     # Compute the coefficients using the empirical basis function ebf
    coefficients = simpleSL.coefsl(mass, pos[:,0], pos[:,1], pos[:,2],  './m12b_SPHLbasis_empirical.txt',2,6)
    print(np.shape(ebf))
    xvals = 10.**(np.linspace(0,2.5, 2000))

    for n in range(0,5):
        plt.plot(xvals, ebf[0][n])

    plt.savefig('testfig.png')
    plt.close()

    print(np.shape(coefficients))

