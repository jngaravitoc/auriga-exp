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

    try:
        fields_stars = ['pos','vel','id','mass','type','pot','age']
        Datstars = sim.Snapshot_Subhalo(idSubhalo=0,snapType='stars',fields=fields_stars)
    except KeyError:
        print('No read age stars')
        fields_stars = ['pos','vel','id','mass','type','pot']
        Datstars = sim.Snapshot_Subhalo(idSubhalo=0,snapType='stars',fields=fields_stars)

    fields_dm = ['pos','vel','id','mass','type','pot']
    DatDM = sim.Snapshot_Subhalo(idSubhalo=0,snapType='dm1',fields=fields_dm)

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

    binrad = 200

    rrmin,rrmax=np.nanmin(np.log10(rr)),np.nanmax(np.log10(rr))
    r,rho = return_density(np.log10(rr),weights= DMmass, rangevals=[rrmin, rrmax],bins=binrad)



    if  ns==63:
        nametab = "Au{}_Snap{}_table.txt".format(nhalo,63)
        R0,D0,M0,P0,scal = makemodel_empirical(r, rho, nametab,GetScaling=True)
        config_=config(len(R0),round(np.nanmin(R0),3), round(np.nanmax(R0),3),nametab)
        print(config_)
        # Construct the basis instances
        #
        basis = pyEXP.basis.Basis.factory(config_)
        
        lrmin = np.log10(np.nanmin(R0))
        lrmax = np.log10(np.nanmax(R0))
        rnum  = len(R0) 
        basis_grid = basis.getBasis(lrmin, lrmax, rnum)


        r = np.linspace(lrmin, lrmax, rnum)
        r = np.power(10.0, r)
        
        #---------------------
        #plt.plot(R0,np.log10(D0),color=colors[ii],zorder=ns)
        
        #-------------------------------------
        Nmax,lmax=20,6+1
        colors =plt.get_cmap('cool', Nmax)
        fig, ax = plt.subplots(nrows=lmax, ncols=1,figsize=(8,15))
        plt.subplots_adjust(hspace=0)
        for l in range(lmax):
            #plt.subplot(4,1,m+1)
            #ax=fig.add_subplot(4,1,m+1)
            for n in range(0,Nmax):
                ax[l].semilogx(r, basis_grid[l][n],c=colors(n),label='n = %s'%n,zorder=Nmax-n)
                ax[l].set_ylabel('m=%s'%l,fontsize=12)
                ax[l].set_xlim(0,300)
                if l!=lmax-1:ax[l].set_xticks([])

        #plt.legend(title='radial order',fontsize=14,title_fontsize=18)

        norm = mpl.colors.Normalize(vmin=0, vmax=Nmax)
        sm = plt.cm.ScalarMappable(cmap=colors, norm=norm)
        sm.set_array([])  
        #cbar_ax = fig.add_axes([0.5, 0.15, 0.5, 0.7])

        cb=plt.colorbar(sm,ax=ax.ravel().tolist(), ticks=np.arange(0, Nmax, 1))
        cb.set_label(label='radial order, n', size=16, weight='bold')
        plt.xlabel('r ',fontsize=18)

        fig.text(0.00, 0.5, 'Potential', va='center', rotation='vertical',fontsize=16)

        #plt.subplots_adjust(hspace=0.1)
        plt.savefig('plots/check63vsT_basis/Au%s_sn63%s_EMB.png'%(nhalo,ns))
        plt.close()
        #----------------------------------------------------------
        verb='Au-%s SN%s => Initialice Multipool '%(nhalo,ns)
        la.PrintPercent(ii,Lsnap.size,'fraper',text=verb)

        def GetFields(coords):
            dens0,potl0,dens,potl,fx,fy,fz = basis.getFields(x=coords[0],y=coords[1],z=coords[2])
            return dens0,dens,potl0,potl

        pool = MultiPool()
        Fields = np.array(list(pool.map(GetFields, pos)))
        pool.close()
        den0, den1,pot0,pot1 = np.array(Fields[:,0]),np.array(Fields[:,1]),np.array(Fields[:,2]),np.array(Fields[:,3])

        print(np.min(pot0),np.median(pot0),np.max(pot0),np.isnan(pot0).any())
        print(np.min(pot1),np.median(pot1),np.max(pot1),np.isnan(pot1).any())
        # Build Image
        Npx=100
        xb,yb=np.linspace(pos[:,0].min(), pos[:,0].max(),Npx),np.linspace(pos[:,1].min(), pos[:,1].max(),Npx)
        dx,dy= xb[1]-xb[0],yb[1]-yb[0]
        xgrid, ygrid = np.meshgrid(xb, yb)

        def BS2D(dat):
            grid, _x, _y, _ = binned_statistic_2d(pos[:,0], pos[:,1], dat, 'median', bins=[Npx,Npx])
            return grid
        dat_ = [potdm/G_,pot0,pot1]
        pool = MultiPool()
        DAT = list(pool.map(BS2D, dat_ ))
        Hpotmed_,pot_mon,pot_nonmon = DAT
        pool.close()

        mask1,mask2,mask3 = np.isnan(pot_mon),np.isnan(pot_nonmon),np.isnan(Hpotmed_)
        pot_mon[mask1],pot_nonmon[mask2],Hpotmed_[mask3]=0,0,0
        print('ALL nans ',mask1.all(),mask2.all(),mask3.all())
        print('ANY nans ',mask1.any(),mask2.any(),mask3.any())
        print('\n')

        mask=Hpotmed_/Hpotmed_
        CC = np.median(np.abs(Hpotmed_- (pot_mon + pot_nonmon)))

        Hpot_= np.log10(np.abs( Hpotmed_))#* (rb[-1]/(mbacc[-1]) )    ))

        #----
        fig, ax = plt.subplots(1, 3, figsize=(25,6))
        extent = xb.min(),xb.max(),yb.min(),yb.max()
        im = ax[0].imshow(Hpot_.T,extent=extent,aspect='auto',origin='lower',interpolation='none',cmap='gist_stern')
        #im = ax[0].contourf(xgrid, ygrid, Hpot_,100,cmap='gist_stern')
        plt.colorbar(im, ax=ax[0],label='Potential')
        ax[0].set_title('Input Data',fontsize=18)

        im2 = ax[1].imshow(np.log10(np.abs(pot_mon + pot_nonmon-(mask*CC))).T,extent=extent,aspect='auto',origin='lower',interpolation='none',cmap='gist_stern')
        plt.colorbar(im2, ax=ax[1],label='Potential')
        #im2.set_clim(im.get_clim()[0],im.get_clim()[1])
        ax[1].set_title('Potential Field Reconstruction',fontsize=18)


        potresidual = (Hpotmed_ - ((pot_mon + pot_nonmon)-(mask*CC)) )/(Hpotmed_)


        im3 = ax[2].contourf(xgrid, ygrid, potresidual.T,100,cmap='coolwarm')
        plt.colorbar(im3, ax=ax[2],label='Potential')
        im3.set_clim(-0.072,0.072 )
        for i_ in [0,1,2]:
            ax[i_].set_xlabel('x kpc',fontsize=15)
            if i_==0:ax[i_].set_ylabel('y kpc',fontsize=15)
        plt.subplots_adjust(wspace=0.1)
        ax[2].set_title('Residual',fontsize=18)
        plt.savefig('plots/check63vsT_basis/PotEXPMap_Au%s_SN%s.png'%(nhalo,ns))


        plt.close()

    break
    if ns==63: continue

    #--------------------------------------------
    '''
    R,D,M,P = makemodel_empirical(r, rho, "Au{}_Snap{}_table.txt".format(nhalo,ns))
    plt.figure(figsize=(12,5))

    plt.subplot(121)
    plt.xlabel('r',fontsize=14)
    plt.ylabel(r'$log\;\rho$',fontsize=14)
    plt.title('snapshot: %s'%63)
    plt.plot(R0,np.log10(D0))


    plt.subplot(122)
    plt.title('snapshot: %s'%ns)
    plt.xlabel('r',fontsize=14)
    plt.ylabel(r'$log\;\rho$',fontsize=14)
    plt.plot(R,np.log10(D))

    plt.savefig('plots/check63vsT_basis/Au%s_sn63%s_rho-r.png'%(nhalo,ns))
    plt.close()'''
    #--------------------------------------------
    #plt.plot(R,np.log10(D),color=colors[ii],zorder=ns)
    #plt.text(R[-1],np.log10(D)[-1],str(ns))
    #plt.savefig('plots/check63vsT_basis/Au%s_rho-r.png'%nhalo)
    #plt.close()
    
    pool = MultiPool()
    Fields = np.array(list(pool.map(GetFields, pos)))
    den0, den1,pot0,pot1 = np.array(Fields[:,0]),np.array(Fields[:,1]),np.array(Fields[:,2]),np.array(Fields[:,3])
    pool.close()

    # Build Image
    Npx=100
    xb,yb=np.linspace(pos[:,0].min(), pos[:,0].max(),Npx),np.linspace(pos[:,1].min(), pos[:,1].max(),Npx)
    dx,dy= xb[1]-xb[0],yb[1]-yb[0]
    xgrid, ygrid = np.meshgrid(xb, yb)

    dat = [potdm/G_,pot0,pot1]
    pool = MultiPool()
    DAT = list(pool.map(BS2D, dat ))
    Hpotmed_sn,pot_mon,pot_nonmon = DAT
    pool.close()

    mask1,mask2,mask3 = np.isnan(pot_mon),np.isnan(pot_nonmon),np.isnan(Hpotmed_)
    pot_mon[mask1],pot_nonmon[mask2],Hpotmed_[mask3]=0,0,0

    mask=Hpotmed_/Hpotmed_
    CC = np.median(np.abs(Hpotmed_sn- (pot_mon + pot_nonmon)))

    Hpot_= np.log10(np.abs( Hpotmed_))#* (rb[-1]/(mbacc[-1]) )    ))

    #----
    fig, ax = plt.subplots(1, 3, figsize=(25,6))
    extent = xb.min(),xb.max(),yb.min(),yb.max()
    im = ax[0].imshow(Hpot_.T,extent=extent,aspect='auto',origin='lower',interpolation='none',cmap='gist_stern')
    #im = ax[0].contourf(xgrid, ygrid, Hpot_,100,cmap='gist_stern')
    plt.colorbar(im, ax=ax[0],label='Potential')
    ax[0].set_title('Potential Field snap 63',fontsize=18)

    im2 = ax[1].imshow(np.log10(np.abs(pot_mon + pot_nonmon-(mask*CC))).T,extent=extent,aspect='auto',origin='lower',interpolation='none',cmap='gist_stern')
    plt.colorbar(im2, ax=ax[1],label='Potential')
    im2.set_clim(im.get_clim()[0],im.get_clim()[1])
    ax[1].set_title('Potential Field Reconstruction snap %s'%ns,fontsize=15)


    potresidual = (Hpotmed_ - ((pot_mon + pot_nonmon)-(mask*CC)) )/(Hpotmed_)


    im3 = ax[2].contourf(xgrid, ygrid, potresidual.T,100,cmap='coolwarm')
    plt.colorbar(im3, ax=ax[2],label='Potential')
    im3.set_clim(-0.072,0.072 )
    for i_ in [0,1,2]:
        ax[i_].set_xlabel('x kpc',fontsize=15)
        if i_==0:ax[i_].set_ylabel('y kpc',fontsize=15)
    plt.subplots_adjust(wspace=0.1)
    ax[2].set_title('Residual',fontsize=18)
    plt.savefig('plots/check63vsT_basis/PotEXPMap_Au%s_SN%s.png'%(nhalo,ns))
    plt.show()

    plt.close()
