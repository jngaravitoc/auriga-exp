import numpy as np
from numpy import Inf
##  exp
from spherical_basis_builder import *
import simpleSL
 



if __name__ == "__main__":

    #*************************  Compute BFE: *****************************
    pos = part_rot[not_in_subs]
    mass =  part['dark']['mass'][not_in_subs]
    print(np.sum(mass), mass[0], len(pos), len(mass))
    M_tot = np.sum(mass)

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
        plt.plot(xvals, E1[0][n],color='black')

    plt.savefig('./testfig.png')
        plt.close()

    print(np.shape(coefficients))

