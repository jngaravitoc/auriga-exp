import numpy as np
#import illustris_python as il
from scipy.stats import binned_statistic_2d, binned_statistic
import matplotlib.pyplot as plt
#from sklearn.neighbors import KernelDensity
import h5py
#from scipy import integrate
#from scipy.interpolate import interp1d,interp2d
#from loadmodules import *
from fortran import *

from const import *
from utilities import *
from gadget import *
from gadget_subfind import *


StypePart = {'gas':0,'dm1':1 ,'dm2':2,'stars':4,'bh':5}
L3MHDsam = [16,21,23,24,27,6]


class Reader_Au:
    def __init__(self,Nhalo,Nsnap):

        self.Nhalo=Nhalo
        self.Nsnap=Nsnap
        self.base = '/virgotng/mpa/Auriga/level3/Original/halo_%s/output/'%self.Nhalo
        self.sf = load_subfind(self.Nsnap, dir=self.base, loadonly=None)
        #  /virgo/simulations/Auriga/level3_MHD


    def Header(self):
         '''Muestra un diccionario con el header'''
         Lib = {'boxsize':      self.sf.boxsize,
                'hubbleparam':  self.sf.hubbleparam,
                'omega0':       self.sf.omega0,
                'omegalambda':  self.sf.omegalambda,
                'redshift':     self.sf.redshift,
                'time':         self.sf.time}
         return Lib


    def LoadHalo(self,fields=None):
        '''Carga toda la informacion del subhalo
        para todo el catalogo entero de un snapshot
        Ver en Especificaciones --> FoF Halos'''


    def LoadSubhalo(self,fields=None):
        '''Carga toda la informacion del subhalo
        para todo el catalogo entero de un snapshot
        Ver en Especificaciones --> Subfind'''
        #self.sf = load_subfind(self.Nsnap, dir=self.base, loadonly=fields)
        #self.sf = load_subfind(self.Nsnap, dir=self.base, loadonly=None)
        if fields==None: return self.sf.data
        else: return {n:self.sf.data[n] for n in fields}

    def Snapshot_Subhalo(self,idSubhalo,snapType='stars',fields=None, allPart = False):
        '''Carga toda las particulas de un tipo de un subhalo
        especifico
        Especificaciones --> tpart
        0=gas
        1=dm1
        2=dm2
        3=
        4=stars
        5=black holes'''
        tpart = StypePart[snapType]
        self.s = gadget_readsnap(self.Nsnap, snappath=self.base, hdf5=True, loadonlytype=tpart, loadonlyhalo=idSubhalo, subfind=self.sf, loadonly=fields)

        ## Only selected part
        ipart = (self.s.data['type'] == tpart)
        fielddist=['age','gsph']
        self.HaloPart = {}
        for l in fields:
            if l not in fielddist: self.HaloPart[l]=self.s.data[l][ipart]
            else: self.HaloPart[l]=self.s.data[l]
        if 'age' in fields:
            self.HaloPart['age'] = self.s.cosmology_get_lookback_time_from_a(self.HaloPart['age'], is_flat=True)
        if allPart: return self.HaloPart

        else:
            ## Only Subhalo N
            nstars = self.sf.data['slty'][idSubhalo,tpart]
            self.SubhaloPart = {l:self.HaloPart[l][:nstars] for l in fields}
            ''' Nota. Las particulas estan ordenadas de la mas cercana al centro host a la mas lejana.
            Por lo que la primera particula es usada para centrar. Las particulas del subhalo 0
            tendran particulas in-situ como acretadas. Para mejorar vision del Subhalo Host es
            necesario quitar las acretadas.
            '''
            return self.SubhaloPart


class ToolRot:
    def __init__(self,Data,param={},TransUnit=True,rotatewith='stars',agerotate=5,rotangle=None,
                 agelim=5,nint=4,rrot=[0,10]):
        self.Data = Data
        self.param = param
        self.cvel=self.param['svel']
        self.cpos = self.param['spos']
        if 'header' in self.param.keys():
            self.hd=self.param['header']
            self.Ol,self.Om = self.hd['omegalambda'],self.hd['omega0']
            self.sc = self.hd['time']
            self.h=self.hd['hubbleparam']
            self.h0 = 0.06777
            self.w_ = self.Om/(self.Ol*self.sc**3)
            self.Ht = self.h0*np.sqrt(self.Ol)*np.sqrt(1+self.w_)
        self.rotatewith=rotatewith
        self.agerotate=agerotate
        self.rotangle=rotangle
        self.TransUnit=TransUnit
        self.agelim=agelim
        self.nint=nint
        self.rrot=rrot

    def Centered(self): #POSITION
        k = self.Data.keys()
        F = 1000*self.sc/self.h if self.TransUnit else 1  #[kpc]

        if 'pos' in k:
            self.Data['pos']=self.Data['pos']-self.cpos
            self.Data['pos'] = self.Data['pos']*F
            self.Data['mass'] = self.Data['mass']*1e10/self.h
            return self.Data

        if 'stars' in k:
            for c in k:
                self.Data[c]['pos'] = self.Data[c]['pos']-self.cpos
                self.Data[c]['pos'] = self.Data[c]['pos']*F
                self.Data[c]['mass'] = self.Data[c]['mass']*1e10/self.h
            return self.Data

    def PhVel(self): #PHYSICAL VELOCITIES
        k = self.Data.keys()
        if 'vel' in k:
            self.Data['vel']=(self.Ht*self.Data['pos']) + self.sc*(self.Data['vel']-self.cvel)
            return self.Data

        if 'stars' in k:
            for c in k:
                self.Data[c]['vel']=(self.Ht*self.Data[c]['pos']) + self.sc*(self.Data[c]['vel']-self.cvel)
            return self.Data

    def Rotate(self):
        self.Data = self.Centered()
        self.Data = self.PhVel()
        k = self.Data.keys()
        DatStar = self.Data['stars'] if 'stars' in k else self.Data
        Datapt={}
        if 'stars' in self.Data.keys():
            for pt in self.Data.keys():
                if pt=='stars':continue
                Datapt[pt] = np.hstack([self.Data[pt]['pos'],self.Data[pt]['vel']])

        edad_ind = DatStar['age'] <= self.agelim
        rmin,rmax = self.rrot
        for i in range(self.nint):
            if i!=0:
                DatStar['pos'] = af[:,0:3]
                DatStar['vel'] = af[:,3:6]
            aedad={n:DatStar[n][edad_ind] for n in ['pos','vel']}
            apos,avel = aedad['pos'],aedad['vel']
            rstar_2d = np.sqrt(apos[:,0]**2 + apos[:,1]**2)
            #print('rstar_2d',len(rstar_2d))
            #Seleccionando en posicion y edad
            cutpos = (rstar_2d<=rmax)&(rstar_2d>=rmin)#&(abs(aedad[2])<=zlim)
            apos,avel = apos[cutpos],avel[cutpos]

            atemp = np.hstack([apos,avel])
            #print('atemp ',len(atemp[:,1]))
            #r X v
            Li = atemp[:,1]*atemp[:,5] - atemp[:,2]*atemp[:,4]
            Lj = atemp[:,2]*atemp[:,3] - atemp[:,0]*atemp[:,5]
            Lk = atemp[:,0]*atemp[:,4] - atemp[:,1]*atemp[:,3]

            Lx,Ly,Lz = np.sum(Li),np.sum(Lj),np.sum(Lk)
            L = np.sqrt((Lx**2) + (Ly**2) + (Lz**2))

            #Definicion de angulos
            self.theta = -np.rad2deg(np.arccos(Lz/L))
            #print(theta)

            if (Lx>0) and (Ly>0):
                self.phi = -np.rad2deg(np.arctan(Lx/Ly))
                #print('case1')

            if (Lx>0) and (Ly<0):
                self.phi = -(180. + np.rad2deg(np.arctan(Lx/Ly)))
                #print('case2')

            if (Lx<0) and (Ly<0):
                self.phi = -(180. + np.rad2deg(np.arctan(Lx/Ly)))
                #print('case3')

            if (Lx<0) and (Ly>0):
                self.phi = -(360. + np.rad2deg(np.arctan(Lx/Ly)))
                #print('case4')

            dat = np.hstack([DatStar['pos'],DatStar['vel']])
            af = np.copy(self.rotor(dat))#np.copy(as3)
            if 'stars' in self.Data.keys():
                for pt in self.Data.keys():
                    if pt=='stars':continue
                    datpt = Datapt[pt]
                    afpt = np.copy(self.rotor(datpt))
                    Datapt[pt]=afpt

            if i==0:
                rmax,rmin,zlim = 5,0,4
            elif i!=0:
                zlim-=1
                rmax-=1
        DatStar['pos'],DatStar['vel'] = af[:,0:3],af[:,3:6]

        if 'stars' in self.Data.keys():
            self.Data['stars']=DatStar
            for pt in self.Data.keys():
                if pt=='stars':continue
                self.Data[pt]['pos'] = Datapt[pt][:,0:3]
                self.Data[pt]['vel'] = Datapt[pt][:,3:6]
        #dat = af
        return self.Data

    def rotor(self,dat):
        as1 = np.copy(dat)
        as1[:,0] =  dat[:,0]*np.cos(np.deg2rad(self.phi)) + dat[:,1]*np.sin(np.deg2rad(self.phi))
        as1[:,1] = -dat[:,0]*np.sin(np.deg2rad(self.phi)) + dat[:,1]*np.cos(np.deg2rad(self.phi))
        as1[:,2] =  dat[:,2]
        as1[:,3] =  dat[:,3]*np.cos(np.deg2rad(self.phi)) + dat[:,4]*np.sin(np.deg2rad(self.phi))
        as1[:,4] = -dat[:,3]*np.sin(np.deg2rad(self.phi)) + dat[:,4]*np.cos(np.deg2rad(self.phi))
        as1[:,5] =  dat[:,5]


        as2 = np.copy(as1)
        as2[:,0] = as1[:,0]
        as2[:,1] = as1[:,2]*np.sin(np.deg2rad(self.theta)) + as1[:,1]*np.cos(np.deg2rad(self.theta))
        as2[:,2] = as1[:,2]*np.cos(np.deg2rad(self.theta)) - as1[:,1]*np.sin(np.deg2rad(self.theta))
        as2[:,3] = as1[:,3]
        as2[:,4] = as1[:,5]*np.sin(np.deg2rad(self.theta)) + as1[:,4]*np.cos(np.deg2rad(self.theta))
        as2[:,5] = as1[:,5]*np.cos(np.deg2rad(self.theta)) - as1[:,4]*np.sin(np.deg2rad(self.theta))

        #phi = -phi

        as3 = np.copy(as2)
        as3[:,0] =  as2[:,0]*np.cos(np.deg2rad(-self.phi)) + as2[:,1]*np.sin(np.deg2rad(-self.phi))
        as3[:,1] = -as2[:,0]*np.sin(np.deg2rad(-self.phi)) + as2[:,1]*np.cos(np.deg2rad(-self.phi))
        as3[:,2] =  as2[:,2]
        as3[:,3] =  as2[:,3]*np.cos(np.deg2rad(-self.phi)) + as2[:,4]*np.sin(np.deg2rad(-self.phi))
        as3[:,4] = -as2[:,3]*np.sin(np.deg2rad(-self.phi)) + as2[:,4]*np.cos(np.deg2rad(-self.phi))
        as3[:,5] =  as2[:,5]

        return as3










def StellarDensity2D(x,y,weights,minMax=None,statistic='sum',npix = [250,250],style='SB'):
    if minMax==None: minMax=[ min([min(x),min(y)]),max([max(x),max(y)]) ]

    grid, _x, _y, _ = binned_statistic_2d(x, y, weights, statistic, bins=npix)#, range=[minMax,minMax])
    box=np.abs(_x[1]-_x[0])#k['box']
    #minMax = [0, 1]

    if style=='normal': return grid,box
    if statistic=='sum':

        if style=='SB':
            #box_arcsec = np.rad2deg(np.arctan(box*1000/10))*(60*60)
            box_arcsec = (box*1000/10)*((180*60*60)/np.pi)
            return grid / box_arcsec**2,_x,_y,box
        if style=='SD':
            return grid / box**2,_x,_y,box
    if statistic!='sum':
        print('statistic i not sum')
        raise SystemExit

def StellarDensity1D(R,f,boxl=0.5,lmin=0,lmax=50):
    Flux = lambda m: 10.**(-0.4 * m)
    kpc2arcsec = lambda x: (x*1000/10)*((180*60*60)/np.pi)
    flux = Flux(f)

    rstar = R#np.sqrt(fstar[:,0]**2 + fstar[:,1]**2)
    #boxl,lmin,lmax = 0.5,0,50
    r = np.arange(lmin,lmax,boxl)

    box_arcsec = (boxl*1000/10)*((180*60*60)/np.pi)
    #print(box_arcsec)
    c = 0
    sb = np.zeros( (r.size,2) )
    # Perfil de brillo superficial
    for i in range(r.size-1):
        ind = (r[i]<=rstar)&(rstar<r[i+1])
        sb[c,0] = r[i]
        sb[c,1] = (np.sum(flux[ind]) ) / (np.pi*(r[i+1]**2 - r[i]**2) * kpc2arcsec(1)**2 ) #mag/arcsec^2
        c+=1

    mvs = -2.5*np.log10(sb[:,1])
    return sb[:,0],mvs


def Ropts(x,y,mag=26.5,):
    #x,y are numpy array
    ymag = np.abs(y-mag)
    pos = np.argmin(ymag)

    ropt = x[pos]
    return ropt

def MR5090(r,val,sel):
    #based-code in LightRadii.py
    # Calcula R90, R50
    '''
    r: array numpy radius proyected in the disc
    val: list of values to calculate de Param
    sel: float selecting stars by radius position
    '''

    sel = r<=sel

    r = r[sel] #selecciona
    sort = np.argsort(r) #ordena
    r = r[sort]
    R90,R50 = [],[]
    M90,M50 = [],[]
    ACC = []
    for v in val:
        v=v[sel][sort]

        acc=  np.add.accumulate(v)/np.sum(v)
        val05,val09 = np.argmin(np.abs(acc-0.5)),np.argmin(np.abs(acc-0.9))
        r90,r50 = r[val09],r[val05]
        m90,m50 = np.add.accumulate(v)[val09],np.add.accumulate(v)[val05]
        R50.append(r50)
        R90.append(r90)
        M50.append(m50)
        M90.append(m90)

        ACC.append([r,acc])
    if len(R90)==1: return R50[0],R90[0],M50[0],M90[0],ACC[0]
    else: return np.array(R50),np.array(R90),np.array(M50),np.array(M90),np.array(ACC)


class Asymmetry:
    def __init__(self,
                 stars_data,dm_data):
        self.fstar = stars_data
        self.fdm = dm_data
        '''
        fstar: [x,y,z,vx,vy,vz,m,age,idstar,Metal,U,B,V,K,g,r,i,z_]
        fdm [x,y,z,vx,vy,vz,iddm]
        '''

    def Mfourier(self,Ropt,r_min=0.4,r_max=1.2,r_int=0.1):

        #x,y,z,vx,vy,vz,m,age,idstar,insitu
        #0,1,2,3 ,4 ,5 ,6, 7 ,8,9

        ind = (np.abs(self.fstar[:,2])<=10) #seleccionando part con altura al disco <=10
        sel = self.fstar[ind]

        a = sel[:,:3] #Posiciones
        am = sel[:,6] #Masas
        #ais = sel[:,9]# insitu stars

        del sel,ind

        theta = np.arctan2(a[:,1],a[:,0]) #angulo/fase

        r2d = np.sqrt(a[:,0]**2 + a[:,1]**2) #distancia polar
        r2d = r2d/Ropt #Normalizacion por Ropt

        ## Modos de Fourier
        n_=round((r_max-r_min)/r_int)
        rint = r_int#0.1
        rmin = r_min#0.4
        rmax = rmin+rint

        B = [[],[],[],[],[],[],[]]
        Rbin = []
        for i in range(n_):

            ind = (rmin<r2d)&(r2d<=rmax)
            sel = a[ind]
            selm = am[ind]
            seltheta = theta[ind]
            #sis = ais[ind]==1 #estrella formada insitu
            for m in range(1,7):
                an = np.sum( selm * np.cos(m*seltheta) )
                bn = np.sum( selm * np.sin(m*seltheta) )
                B[m].append(np.sqrt(an**2 + bn**2) )

            B[0].append(np.sum(selm) )
            Rbin.append(rmin)
            rmax+=rint
            rmin+=rint
        B = np.array(B)
        Rbin = np.array(Rbin)

        An = B[1:]/B[0]

        Amean= np.array([np.mean(Ai) for Ai in An] )
        return Amean[0], (Rbin,An[0])


    def CMstar(self,Ropt,rmin=0.4,rmax=1.2,zmax=10):
        rmin,rmax=rmin*Ropt,rmax*Ropt
        r = np.sqrt(self.fstar[:,0]**2 + self.fstar[:,1]**2 )
        sel= (rmin<=r)&(r<=rmax) &(np.abs(self.fstar[:,2])<=zmax)
        #sis = fstar[:,9]==1 # insitu stars

        Mt = np.sum(self.fstar[:,6][sel])
        cmx = np.sum(self.fstar[:,0][sel]*self.fstar[:,6][sel])/Mt
        cmy = np.sum(self.fstar[:,1][sel]*self.fstar[:,6][sel])/Mt
        cmz = np.sum(self.fstar[:,2][sel]*self.fstar[:,6][sel])/Mt
        cm = np.sqrt(cmx**2 + cmy**2 )

        '''Mtis = np.sum(self.fstar[:,6][sel & sis])
        cmxis = np.sum(self.fstar[:,0][sel & sis]*fstar[:,6][sel & sis])/Mtis
        cmyis = np.sum(self.fstar[:,1][sel & sis]*fstar[:,6][sel & sis])/Mtis
        cmzis = np.sum(self.fstar[:,2][sel & sis]*fstar[:,6][sel & sis])/Mtis
        cmis = np.sqrt(cmxis**2 + cmyis**2 )'''
        return np.array([cmx,cmy,cmz])


    def Deltar(self,Ropt,DMmass,**kwargs):
        dmx,dmy,dmz = self.fdm[:,0],self.fdm[:,1],self.fdm[:,2]
        dmr = np.sqrt(dmx**2 + dmy**2 + dmz**2)

        cmxa,cmya,cmza = CM([dmx,dmy,dmz],DMmass)
        CMa = np.sqrt(cmxa**2 + cmya**2 + cmza**2)

        _1=(dmr<=3*Ropt)
        cmx1,cmy1,cmz1 = CM([dmx[_1],dmy[_1],dmz[_1]],DMmass)
        CM1 = np.sqrt(cmx1**2 + cmy1**2 + cmz1**2)

        _5=(dmr<=5*Ropt)
        cmx5,cmy5,cmz5 = CM([dmx[_5],dmy[_5],dmz[_5]],DMmass)
        CM5 = np.sqrt(cmx5**2 + cmy5**2 + cmz5**2)

        _50=(dmr<=50)
        cmx50,cmy50,cmz50 = CM([dmx[_50],dmy[_50],dmz[_50]],DMmass)
        CM50 = np.sqrt(cmx50**2 + cmy50**2 + cmz50**2)

        _100=(dmr<=100)
        cmx100,cmy100,cmz100 = CM([dmx[_100],dmy[_100],dmz[_100]],DMmass)
        CM100 = np.sqrt(cmx100**2 + cmy100**2 + cmz100**2)
        ret = [np.array([cmxa,cmya,cmza]),np.array([cmx1,cmy1,cmz1]),np.array([cmx5,cmy5,cmz5]),np.array([cmx50,cmy50,cmz50]),np.array([cmx100,cmy100,cmz100])]

        if 'R200' in kwargs.keys():
            r200 = kwargs['R200']
            _50o=(50<=dmr)&(dmr<=r200)
            cmx50o,cmy50o,cmz50o = CM([dmx[_50o],dmy[_50o],dmz[_50o]],DMmass)
            CM50o = np.sqrt(cmx50o**2 + cmy50o**2 + cmz50o**2)
            ret.append(np.array([cmx50o,cmy50o,cmz50o]))

            _10o=(100<=dmr)&(dmr<=r200)
            cmx10o,cmy10o,cmz10o = CM([dmx[_10o],dmy[_10o],dmz[_10o]],DMmass)
            CM10o = np.sqrt(cmx10o**2 + cmy10o**2 + cmz10o**2)
            ret.append(np.array([cmx10o,cmy10o,cmz10o]))
        return ret










### NOMENCLATURAS DE LA LIBRERIA
### PARA CARGAR DATOS GENERALES/CATALOGO
hdf5_name_conversion = {
        'GroupDensGasAngMomentum' : 'dgam',

        'SubhaloCM' : 'scmp',
        'SubhaloGrNr' : 'sgnr',
        'SubhaloHalfmassRad' : 'shmr',
        'SubhaloHalfmassRadType' : 'shmt',
        'SubhaloIDMostbound' : 'sidm',
        'SubhaloLen' : 'slen',
        'SubhaloLenType' : 'slty',
        'SubhaloMass' : 'smas',
        'SubhaloMassInRad' : 'smir',
        'SubhaloMassInRadType' : 'smit',
        'SubhaloMassType' : 'smty',
        'SubhaloParent' : 'sprt',
        'SubhaloPos' : 'spos',
        'SubhaloSFR' : 'ssfr',
        'SubhaloSFRinRad' : 'ssfi',
        'SubhaloSpin' : 'sspi',
        'SubhaloVel' : 'svel',
        'SubhaloVelDisp' : 'svdi',
        'SubhaloVmax' : 'svmx',
        'SubhaloVmaxRad' : 'svrx',
        'SubhaloStellarPhotometrics' : 'ssph',

        'GroupFirstSub' : 'ffsh',
        'GroupLen' : 'flen',
        'GroupLenType' : 'flty',
        'GroupMass' : 'fmas',
        'GroupMassType' : 'fmty',
        'GroupNsubs' : 'fnsh',
        'GroupPos' : 'fpos',
        'GroupSFR' : 'fsfr',
        'GroupVel' : 'fvel',
        'Group_M_Crit200' : 'fmc2',
        'Group_M_Mean200' : 'fmm2',
        'Group_M_TopHat200' : 'fmt2',
        'Group_R_Crit200' : 'frc2',
        'Group_R_Mean200' : 'frm2',
        'Group_R_TopHat200' : 'frt2',
# Illustris specific fields
        'GroupBHMass' : 'fbhm',
        'GroupBHMdot' : 'fbhd',
        'GroupCM' : 'fgcm',
        'GroupFuzzOffsetType' : 'ffot',
        'GroupGasMetallicity' : 'fgmt',
        'GroupGasMetalFractions' : 'fgmf',
        'GroupStarMetalFractions' : 'fsmf',
        'GroupStarMetallicity' : 'fsmt',
        'GroupWindMass' : 'gwim',
        'Group_M_Crit500' : 'fmc5',
        'Group_R_Crit500' : 'frc5',
        'SubhaloBHMass' : 'sbhm',
        'SubhaloBHMdot' : 'sbhd',
        'SubhaloGasMetalFractions' : 'sgmf',
        'SubhaloGasMetalFractionsHalfRad' : 'sgmh',
        'SubhaloGasMetalFractionsMaxRad' : 'sgmx',
        'SubhaloGasMetalFractionsSfr' : 'sgms',
        'SubhaloGasMetalFractionsSfrWeighted' : 'sgmw',
        'SubhaloGasMetallicity' : 'sgmt',
        'SubhaloGasMetallicityHalfRad' : 'sgth',
        'SubhaloGasMetallicityMaxRad' : 'sgtx',
        'SubhaloGasMetallicitySfr' : 'sgts',
        'SubhaloGasMetallicitySfrWeighted' : 'sgtw',
        'SubhaloMassInHalfRad' : 'smih',
        'SubhaloMassInHalfRadType' : 'smht',
        'SubhaloMassInMaxRad' : 'smim',
        'SubhaloMassInMaxRadType' : 'smmt',
        'SubhaloSFRinHalfRad' : 'sshr',
        'SubhaloSFRinMaxRad' : 'ssxr',
        'SubhaloStarMetalFractions' : 'ssmf',
        'SubhaloStarMetalFractionsHalfRad' : 'ssfh',
        'SubhaloStarMetalFractionsMaxRad' : 'ssfx',
        'SubhaloStarMetallicity' : 'ssmt',
        'SubhaloStarMetallicityHalfRad' : 'ssth',
        'SubhaloStarMetallicityMaxRad' : 'sstx',
        'SubhaloStellarPhotometricsMassInRad' : 'sspm',
        'SubhaloStellarPhotometricsRad' : 'sspr',
        'SubhaloWindMass' : 'swim',
        'SubhaloBfldHalo' : 'bfdh',
        'SubhaloBfldDisk' : 'bfdd',

        'Subhalo_Jstars' : 'shsa',
        'Subhalo_JstarsInHalfRad' : 'sjsh',
        'Subhalo_JstarsInRad' : 'sjsr'
}


### PARA CARGAR PARTICULAS

hdf5_name_conversion = {
        'Coordinates':'pos',
        'Velocities':'vel',
        'BirthPos':'bpos',
        'BirthVel':'bvel',
        'TracerID':'trid',
        'ParentID':'prid',
        'FluidQuantities':'fldq',
        'InternalEnergy':'u',
        'ParticleIDs' :'id',
        'Masses' : 'mass',
        'Density' :'rho',
        'Volume' : 'vol',
        'Pressure' : 'pres',
        'SmoothingLength' : 'hsml',
        'Nuclear Composition' : 'xnuc',
        'NuclearComposition' : 'xnuc',
        'Passive Scalars' : 'pass',
        'PassiveScalars' : 'pass',
        'StarFormationRate' :'sfr',
        'StellarFormationTime' : 'age',
        'FeedbackDone' : 'nfb',
        'GFM_StellarFormationTime' : 'age',
        'GFM_InitialMass' : 'gima',
        'GFM_Metallicity' : 'gz',
        'GFM_MassReleased' : 'gimr',
        'GFM_MetalReleased' : 'gimz',
        'GFM_MetalsReleased' : 'gmmz',
        'GFM_StellarPhotometrics' : 'gsph',
        'GFM_AGNRadiation' : 'agnr',
        'GFM_CoolingRate' : 'gcol',
        'GFM_Metallicity' : 'gz',
        'GFM_Metals' : 'gmet',
        'GFM_RProcess' : 'gmrp',
        'GFM_NSNS_Count' : 'gmrc',
        'GravPotential' : 'gpot',
        'Metallicity' : 'z',
        'Potential' :'pot',
        'GravPotential' : 'pot',
        'Acceleration' : 'acce',
        'TimeStep' : 'tstp',
        'MagneticField' : 'bfld',
        'MagneticFieldPsi' : 'psi',
        'DednerSpeed' : 'vded',
        'CurlB' : 'curb',
        'DivBCleening' : 'psi',
        'SmoothedMagneticField' : 'bfsm',
        'RateOfChangeOfMagneticField' : 'dbdt',
        'DivergenceOfMagneticField' : 'divb',
        'MagneticFieldDivergenceAlternative' : 'dvba',
        'MagneticFieldDivergence' : 'divb',
        'PressureGradient' : 'grap',
        'DensityGradient' : 'grar',
        'BfieldGradient' : 'grab',
        'VelocityGradient' : 'grav',
        'Center-of-Mass' : 'cmce',
        'CenterOfMass' : 'cmce',
        'Surface Area' : 'area',
        'Number of faces of cell' : 'nfac',
        'VertexVelocity' : 'veve',
        'Divergence of VertexVelocity' : 'divv',
        'VelocityDivergence' : 'divv',
        'Temperature' : 'temp',
        'Vorticity' : 'vort',
        'AllowRefinement' : 'ref',
        'ElectronAbundance' : 'ne',
        'HighResGasMass' : 'hrgm',
        'NeutralHydrogenAbundance': 'nh',
        'Star Index': 'sidx',
        'CellSpin': 'spin',
        'Spin Center': 'lpos',
        'Machnumber' :'mach',
        'EnergyDissipation' : 'edis',
        'CosmicRaySpecificEnergy': 'cren',
        'CosmicRayStreamingRegularization': 'chi',
        'CRPressureGradient': 'crpg',
        'BfieldGradient': 'bfgr',
        'Nucler energy generation rate': 'dedt',
        'NuclearEnergyGenerationRate': 'dedt',
        'MolecularWeight' : 'mu',
        'SoundSpeed' : 'csnd',
        'Jet_Tracer' : 'jetr',

        'Softenings': 'soft',
        'GravityInteractions' : 'gint',
        'SFProbability' : 'sfpr',
        'TurbulentEnergy' : 'utur',
        'AccretedFlag':'accf',
        'Erad' : 'erad',
        'Lambda' : 'lamd',
        'Kappa_P' : 'ka_p',
        'Kappa_R' : 'ka_r',
        'graderad' : 'grae',

        'BH_CumMassGrowth_QM' : 'bcmq',
        'BH_CumMassGrowth_RM' : 'bcmr',
        'BH_Hsml' : 'bhhs',
        'BH_Pressure' : 'bhpr',
        'BH_U' : 'bhu',
        'BH_Density' : 'bhro',
        'BH_Mdot' : 'bhmd',
        'BH_Mdot_Radio' : 'bhmr',
        'BH_Mdot_Quasar' : 'bhmq',
        'BH_Mass' : 'bhma',
        'BH_MassMetals' : 'bhmz',
        'BH_TimeStep' : 'bhts',
        'BH_MdotBondi' : 'bhbo',
        'BH_MdotEddington' : 'bhed',

        'ChemicalAbundances' : 'chma',
        ' Dust Temperature' : 'dtem',
        'CoolTime' : 'cltm',
        'MagneticVectorPotential' : 'afld',
        'MagneticAShiftX' : 'ashx',
        'MagneticAShiftY' : 'ashy',
        'MagneticAShiftZ' : 'ashz',

        'WindTrackerMassFraction' : 'gtwi',
        'SNTrackerMassFraction' : 'gtsn'}

hdf5_part_field = {
        'pos' : 'all',
        'vel' : 'all',
        'bpos' : 'stars',
        'bvel' : 'stars',
        'trid' : 'tracers',
        'prid' : 'tracers',
        'fldq' : 'tracers',
        'u' : 'gas',
        'id' : 'all',
        'mass' : 'all',
        'rho' : 'gas',
        'vol' : 'gas',
        'pres' : 'gas',
        'hsml' : 'gas',
        'xnuc' : 'gas',
        'pass' : 'gas',
        'sfr' : 'gas',
        'age' : 'stars',
        'nfb' : 'stars',  #CHECK!!!
        'gima' : 'stars',
        'gimr' : 'stars',
        'gimz' : 'stars',
        'gmmz' : 'stars',
        'accf' : 'stars',
        'gz' : 'baryons',
        'gsph' : 'stars',
        'gpot' : 'stars',
        'agnr' : 'gas',
        'gcol' : 'gas',
        'gmet' : 'baryons',
        'gmrp' : 'baryons',
        'gmrc' : 'baryons',
        'z' : 'baryons', #CHECK!!
        'pot' : 'all',
        'acce' : 'all',
        'tstp' : 'all',
        'bfld' : 'gas',
        'curb' : 'gas',
        'afld' : 'gas',
        'ashx' : 'gas',
        'ashy' : 'gas',
        'ashz' : 'gas',
        'psi' : 'gas',
        'vded' : 'gas',
        'bfsm' : 'gas',
        'dbdt' : 'gas',
        'divb' : 'gas',
        'dvba' : 'gas',
        'grap' : 'gas',
        'grar' : 'gas',
        'grab' : 'gas',
        'grav' : 'gas',
        'cmce' : 'gas',
        'area' : 'gas',
        'nfac' : 'gas',
        'veve' : 'gas',
        'divv' : 'gas',
        'divv' : 'gas',
        'temp' : 'gas',
        'vort' : 'gas',
        'ref' : 'gas',
        'xnuc' : 'gas',
        'ne' : 'gas',
        'hrgm' : 'gas',
        'nh' : 'gas',
        'sidx' : 'gas',
        'spin' : 'gas',
        'lpos' : 'gas',
        'mach' : 'gas',
        'edis' : 'gas',
        'cren' : 'gas',
        'chi'  : 'gas',
        'dedt' : 'gas',
        'bfgr' : 'gas',
        'crpg' : 'gas',
        'mu'   : 'gas',
        'csnd' : 'gas',
        'jetr' : 'gas',

        'soft' : 'all',
        'gint' : 'all',
        'sfpr' : 'gas',
        'utur' : 'gas',

        'bcmq' : 'bh',
        'bcmr' : 'bh',
        'bhhs' : 'bh',
        'bhpr' : 'bh',
        'bhu' : 'bh',
        'bhro' : 'bh',
        'bhmd' : 'bh',
        'bhmr' : 'bh',
        'bhmq' : 'bh',
        'bhma' : 'bh',
        'bhmz' : 'bh',
        'bhts' : 'bh',
        'bhbo' : 'bh',
        'bhed' : 'bh',

        'chma' : 'gas',
        'dtem' : 'gas',
        'cltm' : 'gas',

        'gtsn' : 'baryons',
        'gtwi' : 'baryons'
}

# Defino parámetros a utilizar ### POR AHORA
#h_0 = 0.06777
#omega_lambda= 0.693
#omega_m=0.307
#w = omega_m / (omega_lambda*(fac_esc**3))
#hubble_t = h_0 * np.sqrt(omega_lambda * np.sqrt(1+w))

# OTROS ______________________________________________________
class colors:
    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK  = '\33[30m'
    CRED    = '\33[31m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    CGREY    = '\33[90m'
    CRED2    = '\33[91m'
    CGREEN2  = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2   = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2  = '\33[96m'
    CWHITE2  = '\33[97m'

    CGREYBG    = '\33[100m'
    CREDBG2    = '\33[101m'
    CGREENBG2  = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2   = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2  = '\33[106m'
    CWHITEBG2  = '\33[107m'
#>>> print(colors.WARNING + "Warning: No active frommets remain. Continue?" + colors.ENDC)
#>>> print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")
#def tcolor
#    lib = {'header':colors.HEADER
#        }
    
     
def PrintPercent(i,T,typ='fraper',text=None):
    if typ=='fraction': print('==> '+str(i)+'/'+str(T),end='\r')
    if typ=='percent' : print('==> '+str(round(i*100/T,3))+'%' ,end='\r')
    if typ=='fraper'  :
        txt= '' if text==None else text
        nbar = 20
        f = round(nbar*(i/T))
        txtbar = colors.CWHITEBG+' '+colors.CEND+f"{colors.CGREENBG}"+' '*f +f"{colors.CEND}"+f"{colors.CBLACKBG}"+' '*(nbar-f) +f"{colors.CEND}"+ colors.CWHITEBG+' '+colors.CEND
        flecha=colors.CBOLD+colors.CRED+'-==> '+colors.CEND
        percent=colors.CBOLD+colors.CWHITE+str(round(i*100/T,1))+'%'+colors.CEND
        print(txtbar+' '+flecha + str(i)+'/'+str(T)+' - '+ percent+'-'+txt ,end='\r')   

