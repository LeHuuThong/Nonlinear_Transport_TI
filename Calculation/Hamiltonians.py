
import numpy as np
import tinyarray
# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_0 = np.array([[1, 0], [0, 1]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
tau_0 = np.array([[1, 0], [0, 1]])
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

def Two_surface_model(vf, m0, B, g, Mt, Mb, Es, V0):
    """
    Construct the Hamiltonian for a two-surface model as in the Notes
    """
    # parameter mapping
    M_plus = (Mt + Mb) / 2
    M_minus = (Mt - Mb) / 2
    # onsite term
    H0 =    np.kron(tau_0, sigma_0) * Es \
        +   np.kron(tau_0, sigma_z) * (m0 + 4*B) \
        +   np.kron(tau_z, sigma_z) * g*M_plus \
        -   np.kron(tau_y, sigma_y) * g*M_minus \
        +   np.kron(tau_x, sigma_x) * V0
    # hopping in x direction from site (i, j) to (i + 1, j)
    Hx = -  np.kron(tau_0, sigma_z) * B  \
         -  np.kron(tau_z, sigma_y) * 1j * vf/2
    # hopping in y direction from site (i, j) to (i, j + 1)
    Hy = -  np.kron(tau_0, sigma_z) * B  \
         +  np.kron(tau_0, sigma_x) * 1j * vf/2
    
    H0, Hx, Hy = tinyarray.array(H0), tinyarray.array(Hx), tinyarray.array(Hy)
    return H0, Hx, Hy

def Two_surface_continuum(kx, ky, vf, m0, B, g, Mt, Mb, Es, V0):
    # parameter mapping
    M_plus = (Mt + Mb) / 2
    M_minus = (Mt - Mb) / 2
    
    Ham =   np.kron(tau_0, sigma_0) * Es \
        +   np.kron(tau_0, sigma_x) * vf*ky\
        -   np.kron(tau_z, sigma_y) * vf*kx\
        +   np.kron(tau_0, sigma_z) * (m0 + B*(kx**2+ky**2))\
        +   np.kron(tau_z, sigma_z) * g*M_plus\
        -   np.kron(tau_y, sigma_y) * g*M_minus\
        +   np.kron(tau_x, sigma_x) * V0

    Ham = tinyarray.array(Ham)
    return Ham

def Two_surface_and_bulk(vf, m0, B, g, Mt, Mb, Es, V0, Eb, B1):
    H0_surf, Hx_surf, Hy_surf = Two_surface_model(vf, m0, B, g, Mt, Mb, Es, V0)
    def bulk_model(Eb, B1):
        H0 = Eb + 4*B1
        Hx = -B1
        Hy = -B1
        return H0, Hx, Hy
    
    H0_bulk, Hx_bulk, Hy_bulk = bulk_model(Eb, B1)

    H0 = np.block([[H0_surf, np.zeros((4, 1))],
                   [np.zeros((1, 4)), H0_bulk]])
    Hx = np.block([[Hx_surf, np.zeros((4, 1))],
                   [np.zeros((1, 4)), Hx_bulk]])
    Hy = np.block([[Hy_surf, np.zeros((4, 1))],
                   [np.zeros((1, 4)), Hy_bulk]])
    return H0, Hx, Hy

