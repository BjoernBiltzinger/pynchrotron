import numba as nb
import numpy as np

from pynchrotron.synchrotron_kernel import (
    synchrotron_kernel,
    compute_synchtron_matrix,
    compute_synchtron_matrix2
)


@nb.njit(fastmath=True, parallel=False)
def cool_and_radiate(
    energy,
    n_photon_energies,
    ne,
    B,
    bulk_lorentz_factor,
    gamma_min,
    gamma_max,
    index,
    DT,
    n_grid_points,
    steps,
):

    # allocate needed grids
    
    gamma = np.zeros(n_grid_points)
    gamma2 = np.zeros(n_grid_points)
    fgamma = np.zeros(n_grid_points)
    G = np.zeros(n_grid_points + 1)
    source_eval = np.zeros(n_grid_points)
    emission = np.zeros(n_photon_energies)

    # precompute some variables
    
    pm1 = 1 - index
    denom = np.power(gamma_max, pm1) - np.power(gamma_min, pm1)
    cool = 1.29234e-9 * B * B

    # define the step size such that we have a grid slightly
    # out gamma max to avoid boundary issues

    step = np.exp(1.0 / n_grid_points * np.log(gamma_max * 1.1))

    # now compute the grid and the source input
    # which is a power law
    
    for j in range(n_grid_points):

        gamma[j] = np.power(step, j)
        gamma2[j] = gamma[j] * gamma[j]

        if j < n_grid_points - 1:

            G[j] = 0.5 * (gamma[j] + gamma[j] * step)

        else:

            G[n_grid_points - 1] = 0.5 * (gamma[j] + gamma[j] * step)

        if (gamma[j] > gamma_min) and (gamma[j] < gamma_max):
            source_eval[j] = ne * np.power(gamma[j], -index) * (pm1) * 1.0 / denom

    # precompute a matrix for each photon energy and electron grid point
    # takes ~ 500 mus for 100 input energies times 300 gamma2
    synchrotron_matrix = compute_synchtron_matrix(
        energy, gamma2, B, bulk_lorentz_factor, n_photon_energies, n_grid_points
    )


    # allocate the Chang and Cooper arrays
    
    V3 = np.zeros(n_grid_points)
    V2 = np.zeros(n_grid_points)
    delta_gamma1 = G[n_grid_points - 1] - G[n_grid_points - 2]
    delta_gamma2 = G[1] - G[0]


    # these values will remain constant throughout the cooling
    # so just compute them one
    
    for j in range(n_grid_points - 2, 0, -1):

        delta_gamma = 0.5 * (G[j] - G[j - 1])  # Half steps are at j+.5 and j-.5

        gdotp = cool * gamma2[j + 1]  # Forward  step cooling
        gdotm = cool * gamma2[j]  # Backward step cooling

        V3[j] = (DT * gdotp) / delta_gamma  # Tridiagonal coeff.
        V2[j] = 1.0 + (DT * gdotm) / delta_gamma  # Tridiagonal coeff.

    # now compute the cooling and synchrotron emission
        
    #fgammatp1 = np.zeros(n_grid_points)
    val = np.zeros(n_grid_points)

    precalc1 = DT * cool * gamma[1] * gamma[1]/delta_gamma2
    precalc2 =  (DT * cool * gamma[n_grid_points - 1] * gamma[n_grid_points - 1])/ delta_gamma1+1

    # this loop is the bootleneck for gamma_max>>gamma_cool (when the calc. is slow)
    # for gamma_max/gamma_cool = 1e5 takes about 99.6 % of the runtime
    for _ in range(0, steps + 1):

        # set the end point
        
        fgamma[n_grid_points - 1] /= precalc2

        # back sweep through the grid
        
        for j in range(n_grid_points - 2, 0, -1):

            fgamma[j] = (fgamma[j] + source_eval[j] + V3[j] * fgamma[j + 1]) / V2[j]

        # set the end point
            
        fgamma[0] = (
            fgamma[0] + (precalc1 * fgamma[1])
        )

        val += fgamma

    val = val[1:]
    val*=(gamma[1:]-gamma[:-1])

    emission = np.dot(synchrotron_matrix[:,1:], val)/(2.0*energy)

    return emission
