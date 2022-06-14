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
        self.base = '/virgo/simulations/Auriga/level3_MHD/halo_%s/output/'%self.Nhalo
        self.sf = load_subfind(self.Nsnap, dir=self.base, loadonly=None)


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
        fielddist=['age']
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

            #Seleccionando en posicion y edad
            cutpos = (rstar_2d<=rmax)&(rstar_2d>=rmin)#&(abs(aedad[2])<=zlim)
            apos,avel = apos[cutpos],avel[cutpos]

            atemp = np.hstack([apos,avel])

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
