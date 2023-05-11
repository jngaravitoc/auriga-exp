import os, sys, pickle, pyEXP
import numpy as np

def make_config(basis_id, numr, rmin, rmax, lmax, nmax, scale, 
                modelname='', cachename='.slgrid_sph_cache'):
    """
    Creates a configuration file required to build a basis model.

    Args:
    basis_id (str): The identity of the basis model.
    numr (int): The number of radial grid points.
    rmin (float): The minimum radius value.
    rmax (float): The maximum radius value.
    lmax (int): The maximum l value of the basis.
    nmax (int): The maximum n value of the basis.
    scale (float): Scaling factor for the basis.
    modelname (str, optional): Name of the model. Default is an empty string.
    cachename (str, optional): Name of the cache file. Default is '.slgrid_sph_cache'.

    Returns:
    str: A string representation of the configuration file.

    Raises:
    None
    """
    
    config = '\n---\nid: {:s}\n'.format(basis_id)
    config += 'parameters:\n'
    config += '  numr: {:d}\n'.format(numr)
    config += '  rmin: {:.7f}\n'.format(rmin)
    config += '  rmax: {:.3f}\n'.format(rmax)
    config += '  Lmax: {:d}\n'.format(lmax)
    config += '  nmax: {:d}\n'.format(nmax)
    config += '  scale: {:.3f}\n'.format(scale)
    config += '  modelname: {}\n'.format(modelname)
    config += '  cachename: {}\n'.format(cachename)
    config += '...\n'
    return config

def empirical_density_profile(pos, mass, nbins=500, rmin=0, rmax=600, log_space=False):
    """
    Computes the number density radial profile assuming all particles have the same mass.

    Args:
        pos (ndarray): array of particle positions in cartesian coordinates with shape (n,3).
        mass (ndarray): array of particle masses with shape (n,).
        nbins (int, optional): number of bins in the radial profile. Default is 500.
        rmin (float, optional): minimum radius of the radial profile. Default is 0.
        rmax (float, optional): maximum radius of the radial profile. Default is 600.
        log_space (bool, optional): whether to use logarithmic binning. Default is False.

    Returns:
        tuple: a tuple containing the arrays of radius and density with shapes (nbins,) and (nbins,), respectively.

    Raises:
        ValueError: if pos and mass arrays have different lengths or if nbins is not a positive integer.
    """
    if len(pos) != len(mass):
        raise ValueError("pos and mass arrays must have the same length")
    if not isinstance(nbins, int) or nbins <= 0:
        raise ValueError("nbins must be a positive integer")

    # Compute radial distances
    r_p = np.sqrt(np.sum(pos**2, axis=1))

    # Compute bin edges and shell volumes
    if log_space:
        bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    else:
        bins = np.linspace(rmin, rmax, nbins+1)
    V_shells = 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3)

    # Compute density profile
    density, _ = np.histogram(r_p, bins=bins, weights=mass)
    density /= V_shells

    # Compute bin centers and return profile
    radius = 0.5 * (bins[1:] + bins[:-1])
    return radius, density

def make_exp_basis_table(radius, density, outfile='', label='', 
                          verbose=True, physical_units=True, return_values=False):
    
    """
    Create a table of basis functions for an exponential density profile, compatible with EXP.

    Parameters:
    -----------
    radius : array-like
        Array of sampled radius points.
    density : array-like
        Array of density values at radius points.
    outfile : str, optional
        The name of the output file. If not provided, the file will not be written.
    label : str, optional
        A comment string to add to the output file.
    verbose : bool, optional
        Whether to print scaling factors and other details during execution.
    physical_units : bool, optional
        Whether to use physical units (True) or normalized units (False) in the output file.
    return_values : bool, optional
        Whether to return the radius, density, mass, and potential arrays (True) or not (False).

    Returns:
    --------
    If return_values is True:
        radius : array-like
            The radius values.
        density : array-like
            The density values.
        mass : array-like
            The mass enclosed at each radius.
        potential : array-like
            The potential energy at each radius.
    If return_values is False (default):
        None

    Notes:
    ------
    This function assumes an exponential density profile:

        rho(r) = rho_0 * exp(-r/a)

    where rho_0 and a are constants.

    The table of basis functions is used by EXP to create a density profile.

    Reference:
    ----------
    https://gist.github.com/michael-petersen/ec4f20641eedac8f63ec409c9cc65ed7
    """    
    
    M = 1.
    R = np.nanmax(rvals)

    # make the mass and potential arrays
    rvals, dvals = radius, density 
    mvals = np.zeros(density.size)
    pvals = np.zeros(density.size)
    pwvals = np.zeros(density.size)

    # initialise the mass enclosed an potential energy
    mvals[0] = 1.e-15
    pwvals[0] = 0.

    # evaluate mass enclosed and potential energy by recursion
    for indx in range(1, dvals.size):
        mvals[indx] = mvals[indx-1] +\
          2.0*np.pi*(rvals[indx-1]*rvals[indx-1]*dvals[indx-1] +\
                 rvals[indx]*rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);
        pwvals[indx] = pwvals[indx-1] + \
          2.0*np.pi*(rvals[indx-1]*dvals[indx-1] + rvals[indx]*dvals[indx])*(rvals[indx] - rvals[indx-1]);

    # evaluate potential (see theory document)
    pvals = -mvals/(rvals+1.e-10) - (pwvals[dvals.size-1] - pwvals)

    # get the maximum mass and maximum radius
    M0 = mvals[dvals.size-1]
    R0 = rvals[dvals.size-1]

    # compute scaling factors
    Beta = (M/M0) * (R0/R)
    Gamma = np.sqrt( (M0*R0)/(M*R) ) * (R0/R)
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
        print('! ', plabel, file=f)
        print('! R    D    M    P',file=f)

        print(rvals.size,file=f)
        
        if unit_phy:
            for indx in range(0,rvals.size):
                print('{0} {1} {2} {3}'.format(rvals[indx],\
                                              dvals[indx],\
                                              mvals[indx],\
                                              pvals[indx]),file=f)
        else:
            for indx in range(0,rvals.size):
                print('{0} {1} {2} {3}'.format(rfac*rvals[indx],\
                  dfac*dvals[indx],\
                  mfac*mvals[indx],\
                  pfac*pvals[indx]),file=f)

        f.close()
    
    if return_val:
        if unit_phy:
            return rvals, dvals, mvals, pvals

        return rvals*rfac, dfac*dvals, mfac*mvals, pfac*pvals

    
def make_a_BFE(pos, mass, config=None, basis_id='sphereSL', time=0, 
               numr=200, rmin=0.61, rmax=599, lmax=4, nmax=20, scale=22.5, 
               modelname='dens_table.txt', cachename='.slgrid_sph_cache', save_dir=''):
    """
    Create a BFE expansion for a given set of particle positions and masses.
    
    Parameters:
    pos (numpy.ndarray): The positions of particles. Each row represents one particle, 
                         and each column represents the coordinate of that particle.
    mass (numpy.ndarray): The masses of particles. The length of this array should be the same 
                          as the number of particles.
    config (pyEXP.config.Config, optional): A configuration object that specifies the basis set. 
                                             If not provided, an empirical density profile will be computed 
                                             and a configuration object will be created automatically.
    basis_id (str, optional): The type of basis set to be used. Default is 'sphereSL'.
    time (float, optional): The time at which the expansion is being computed. Default is 0.
    numr (int, optional): The number of radial grid points in the basis set. Default is 200.
    rmin (float, optional): The minimum radius of the basis set. Default is 0.61.
    rmax (float, optional): The maximum radius of the basis set. Default is 599.
    lmax (int, optional): The maximum harmonic order in the basis set. Default is 4.
    nmax (int, optional): The maximum number of polynomials in the basis set. Default is 20.
    scale (float, optional): The scale of the basis set. Default is 22.5.
    modelname (str, optional): The name of the file containing the density profile model. 
                               Default is 'dens_table.txt'.
    cachename (str, optional): The name of the file that will be used to cache the basis set. 
                               Default is '.slgrid_sph_cache'.
    save_dir (str, optional): The name of the file if provided that will be used to save the coef files as .h5.
                              Default is ''. 
    Returns:
    tuple: A tuple containing the basis and the coefficients of the expansion.
           The basis is an instance of pyEXP.basis.Basis, and the coefficients are 
           an instance of pyEXP.coefs.Coefs.
    """
    
    # check if config file is passed
    if config is None:
        print('No config file provided.')
        print(f'Computing empirical density')
        rad, rho = empirical_density_profile(pos, mass, nbins=500)
        makemodel_empirical(r_exact, rho, pfile=modelname)
        config = make_config(basis_id, numr, rmin, rmax, lmax, nmax, scale, 
                             modelname, cachename)

    # Construct the basis instances
    basis = pyEXP.basis.Basis.factory(config)

    # Prints info from Cache
    basis.cacheInfo(cachename)
    
    #compute coefficients
    coef = basis.createFromArray(mass, pos, time=time)
    coefs = pyEXP.coefs.Coefs.makecoefs(coef, 'dark halo')
    coefs.add(coef)
    
    if len(save_dir) > 0 :
      coefs.WriteH5Coefs(save_dir)
    
    return basis, coefs

###Field computations for plotting###
def make_basis_plot(basis, savefile=None, nsnap='mean', y=0.92, dpi=200):
    """
    Plots the potential of the basis functions for different values of l and n.

    Args:
    basis (obj): object containing the basis functions for the simulation
    savefile (str, optional): name of the file to save the plot as
    nsnap (str, optional): description of the snapshot being plotted
    y (float, optional): vertical position of the main title
    dpi (int, optional): resolution of the plot in dots per inch

    Returns:
    None

    """
    # Set up grid for plotting potential
    lrmin, lrmax, rnum = 0.5, 2.7, 100
    halo_grid = basis.getBasis(lrmin, lrmax, rnum)
    r = np.linspace(lrmin, lrmax, rnum)
    r = np.power(10.0, r)

    # Create subplots and plot potential for each l and n
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 5, figsize=(6,6), dpi=dpi, 
                            sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0, hspace=0)
    ax = ax.flatten()

    for l in range(len(ax)):
        ax[l].set_title(f"$\ell = {l}$", y=0.8, fontsize=6)    
        for n in range(20):
            ax[l].semilogx(r, halo_grid[l][n]['potential'], '-', label="n={}".format(n), lw=0.5)

    # Add labels and main title
    fig.supylabel('Potential', weight='bold', x=-0.02)
    fig.supxlabel('Radius', weight='bold', y=0.02)
    fig.suptitle(f'nsnap = {nsnap}', 
                 fontsize=12, 
                 weight='bold', 
                 y=y,
                )
    
    # Save plot if a filename was provided
    if savefile:
        plt.savefig(f'{savefile}', bbox_inches='tight')

def find_field(basis, coefficients, time=0, xyz=(0, 0, 0), property='dens', include_monopole=True):
    """
    Finds the value of the specified property of the field at the given position.

    Args:
    basis (obj): Object containing the basis functions for the simulation.
    coefficients (obj): Object containing the coefficients for the simulation.
    time (float, optional): The time at which to evaluate the field. Default is 0.
    xyz (tuple or list, optional): The (x, y, z) position at which to evaluate the field. Default is (0, 0, 0).
    property (str, optional): The property of the field to evaluate. Can be 'dens', 'pot', or 'force'. Default is 'dens'.
    include_monopole (bool, optional): Whether to return the monopole contribution to the property only. Default is True.

    Returns:
    float or list: The value of the specified property of the field at the given position. If property is 'force', a list of
    three values is returned representing the force vector in (x, y, z) directions.

    Raises:
    ValueError: If the property argument is not 'dens', 'pot', or 'force'.
    """

    coefficients.set_coefs(coefficients.getCoefStruct(time))
    dens0, pot0, dens, pot, fx, fy, fz = basis.getFields(xyz[0], xyz[1], xyz[2])
    
    if property == 'dens':
        if include_monopole:
            return dens0
        return dens + dens0

    elif property == 'pot':
        if include_monopole:
            return pot0
        return pot + pot0

    elif property == 'force':
        return [fx, fy, fz]

    else:
        raise ValueError("Invalid property specified. Possible values are 'dens', 'pot', and 'force'.")


def spherical_avg_prop(basis, coefficients, time=0, radius=np.linspace(0.1, 600, 100), property='dens'):
    """
    Computes the spherically averaged value of the specified property of the field over the given radii.

    Args:
    basis (obj): Object containing the basis functions for the simulation.
    coefficients (obj): Object containing the coefficients for the simulation.
    time (float, optional): The time at which to evaluate the field. Default is 0.
    radius (ndarray, optional): An array of radii over which to compute the spherically averaged property. Default is an
        array of 100 values logarithmically spaced between 0.1 and 600.
    property (str, optional): The property of the field to evaluate. Can be 'dens', 'pot', or 'force'. Default is 'dens'.

    Returns:
    ndarray: An array of spherically averaged values of the specified property over the given radii.

    Raises:
    ValueError: If the property argument is not 'dens', 'pot', or 'force'.
    """

    coefficients.set_coefs(coefficients.getCoefStruct(time))
    field = [find_field(basis, np.hstack([[rad], [0], [0]]), property=property, include_monopole=True) for rad in radius]

    if property == 'force':
        return np.vstack(field), radius

    return np.array(field), radius


def slice_fields(basis, coefficients, time=0, 
                 projection='XY', proj_plane=0, npoints=300, 
                 grid_limits=(-300, 300), prop='dens', monopole_only=False):
    """
    Plots a slice projection of the fields of a simulation.

    Args:
    basis (obj): object containing the basis functions for the simulation
    coefficients (obj): object containing the coefficients for the simulation
    time (float): the time at which to plot the fields
    projection (str): the slice projection to plot. Can be 'XY', 'XZ', or 'YZ'.
    proj_plane (float, optional): the value of the coordinate that is held constant in the slice projection
    npoints (int, optional): the number of grid points in each dimension
    grid_limits (tuple, optional): the limits of the grid in the x and y dimensions, in the form (x_min, x_max)
    prop (str, optional): the property to return. Can be 'dens' (density), 'pot' (potential), or 'force' (force).
    monopole_only (bool, optional): whether to return the monopole component in the returned property value.

    Returns:
    array or list: the property specified by `prop`. If `prop` is 'force', a list of the x, y, and z components of the force is returned.
                    Also returns the grid used to compute slice fields. 
    """
    x = np.linspace(grid_limits[0], grid_limits[1], npoints)
    xgrid = np.meshgrid(x, x)
    xg = xgrid[0].flatten()
    yg = xgrid[1].flatten()

    
    if projection not in ['XY', 'XZ', 'YZ']:
        raise ValueError("Invalid projection specified. Possible values are 'XY', 'XZ', and 'YZ'.")

    N = len(xg)
    rho0 = np.zeros_like(xg)
    pot0 = np.zeros_like(xg)
    rho = np.zeros_like(xg)
    pot = np.zeros_like(xg)
    basis.set_coefs(coefficients.getCoefStruct(time))

    for k in range(0, N):
        if projection == 'XY':
            rho0[k], pot0[k], rho[k], pot[k], fx, fy, fz = basis.getFields(xg[k], yg[k], proj_plane)
        elif projection == 'XZ':
            rho0[k], pot0[k], rho[k], pot[k], fx, fy, fz = basis.getFields(xg[k], proj_plane, yg[k])
        elif projection == 'YZ':
            rho0[k], pot0[k], rho[k], pot[k], fx, fy, fz = basis.getFields(proj_plane, xg[k], yg[k])
    
    dens = rho.reshape(npoints, npoints)
    pot = pot.reshape(npoints, npoints)
    dens0 = rho0.reshape(npoints, npoints)
    pot0 = pot0.reshape(npoints, npoints)

    if prop == 'dens':
        if monopole_only:
            return dens0
        return dens0, dens, xgrid

    if prop == 'pot':
        if monopole_only:
            return pot0
        return pot0, pot, xgrid

    if prop == 'force':
        return [fx.reshape(npoints, npoints), fy.reshape(npoints, npoints), fz.reshape(npoints, npoints)], xgrid
    


