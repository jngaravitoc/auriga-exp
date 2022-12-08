import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,leastsq


def PowerLawLn(r,hi,Io):#Io ya tiene log
    LnI = Io-(r/hi)
    return LnI

def BreakFunction(r,Rb,I0i,hi,ho):#I0o
    yi = lambda r: PowerLawLn(r,hi,I0i)
    yo = lambda r: PowerLawLn(Rb,hi,I0i)+((Rb-r)/ho)
    return np.piecewise(r,[r<=Rb,r>=Rb],[yi,yo])

def BreakLine(r,Rb,ai,bi,ao):
    yi = lambda r: (ai*r) + bi
    yo = lambda r: (ai*Rb) + bi + ao*(r-Rb) #(ao*r) + bo
    return np.piecewise(r,[r<=Rb,r>=Rb],[yi,yo])






class breaks:
    def __init__(self,radius,mass,ages):#[Mpc][Msun][Gyr]
        sort = np.argsort(radius)

        self.r = radius[sort]
        self.m = mass[sort]
        self.a = ages[sort]



    def DoubleExp(self,p0=None,bounds=None):
        '''Orden bounds:
        [Rb,I0i,I0o,hi,ho]  <--------<
        curve_fit al definir bounds usara Trust Region Reflexive como metodo de minimos cuadrados, es un metodo similar al levenberg-marquardt pero bien comportado para soluciones con restricciones, como en este caso. El uso de bounds es debido a que Breakfunction tiene 4 parametros, si todos los parametros son libres [-inf,inf] habran muchas soluciones para fittear un perfil y mucho de ellos son espurios, para evitar esto se le entrega restricciones para que el ajuste arroje parametros reales. Dejo esto como recordatorio en caso de escribir una publicacion.
        '''
        #bounds por defecto para este codigo
        if bounds==None: bounds=((self.r[0],0,0,0),(self.r[-1],np.inf,np.inf,np.inf))

        popt,pcov = curve_fit(BreakFunction,self.r,self.m,bounds=bounds)
        Rb,I0i,hi,ho = popt
        I0o=I0i - (1/hi - 1/ho)*Rb

        y_fit_inner = PowerLawLn(self.r,hi,I0i)
        y_fit_outer = PowerLawLn(self.r,ho,I0o)

        y_fit = BreakFunction(self.r,Rb,I0i,hi,ho)
        residuals = self.m - y_fit
        chi_sq = sum((residuals**2)/y_fit)

        Lib = {'h_i':hi,'h_o':ho,
               'I0i':I0i,'I0o':I0o,
               'Rbr':Rb,'LineFit':(self.r,y_fit),
               'LineFit_in':(self.r,y_fit_inner),
               'LineFit_out':(self.r,y_fit_outer),
               'ChiSquare':chi_sq, 'pcov':pcov}
        return Lib

    def PureExp(self,p0=None,bounds=((0,0),(np.inf,np.inf))):
        #Orden bounds
        #[hi,Io]

        popt,pcov = curve_fit(PowerLawLn,self.r,self.m,bounds=bounds)
        hi,Io = popt

        y_fit =PowerLawLn(self.r,hi,Io)
        residuals = self.m - y_fit
        chi_sq = sum((residuals**2)/y_fit)

        Lib = {'h_i':hi,'I0':Io,
               'LineFit':(self.r,y_fit),
               'ChiSquare':chi_sq, 'pcov':pcov}
        return Lib

    def ExponentialClassify(self): #Clasificacion por ChiSquare
        Lib2exp = self.DoubleExp()
        Lib1exp = self.PureExp()

        chisq1,chisq2 = Lib1exp['ChiSquare'],Lib2exp['ChiSquare']
        if chisq1-chisq2<=1:
            return 'TI', (Lib1exp,Lib2exp)
        else:
            beta = Lib2exp['h_i']-Lib2exp['h_o']
            if beta>0: return 'TII', (Lib1exp,Lib2exp)
            else:      return 'TIII', (Lib1exp,Lib2exp)

    def DoubleLine(self,p0=None,bounds=None):
        #Util Para ajustar Age Profiles o gradientes dobles
        '''Orden bounds:
        [Rb,ai,bi,ao]  <--------<
        '''
        #bounds por defecto para este codigo
        if bounds==None: bounds=((self.r[0],-np.inf,-np.inf,-np.inf),(self.r[-1],np.inf,np.inf,np.inf))

        popt,pcov = curve_fit(BreakLine,self.r,self.a,bounds=bounds)
        Rb,ai,bi,ao = popt
        bo=(ai-ao)*Rb + bi

        y_fit_inner = ai*self.r + bi
        y_fit_outer = ao*self.r + bo

        y_fit = BreakLine(self.r,Rb,ai,bi,ao)
        residuals = self.a - y_fit
        chi_sq = sum((residuals**2)/y_fit)

        Lib = {'ai':ai,'ao':ao,
               'bi':bi,'bo':bo,
               'Rb':Rb,'LineFit':(self.r,y_fit),
               'LineFit_in':(self.r,y_fit_inner),
               'LineFit_out':(self.r,y_fit_outer),
               'ChiSquare':chi_sq, 'pcov':pcov}
        return Lib






#=====================================================================================
# OTROS ==============================================================================



def ClassifyVL21(beta,crit=0.5): #Criteria Type surface density Varela-Lavin et al.(2021, prep.)
    beta = np.array(beta)
    selTI = np.abs(beta)<=crit #kpc #Criterio Type I
    selTII= crit < beta # criterio Type II
    selTIII= beta < -1*crit #criterio Type III
    types = np.ones_like(beta,dtype=str)
    types[selTI]='TI'
    types[selTII]='TII'
    types[selTIII]='TIII'
    return types


def SDBinMovil(r,m,n):#[kpc][Msun]
    r,m=np.array(r),np.array(m)
    Sigma = []
    sor=np.argsort(r)
    r,m=r[sor],m[sor]
    #r = r*1e3 #kpc
    for i in range(len(r)):
         if n-1 < i < len(r)-n:
             SumM = np.sum(m[i-n:i+n])
             S = np.pi*((r[i+n]**2) - (r[i-n]**2)) #kpc**2
             Sigma.append(SumM/S)
    return r[n:-n],np.log(np.array(Sigma))

































