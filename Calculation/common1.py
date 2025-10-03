import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dask import delayed, compute

# Pauli matrices
sigma_0 = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
Sigma = [sigma_0, sigma_x, sigma_y, sigma_z]
tau_0 = sigma_0
tau_x = sigma_x
tau_y = sigma_y
tau_z = sigma_z


Umatrix = 1/np.sqrt(2)*np.array([[1,0,0,1],
                                 [0,1,1,0],
                                 [1,0,0,-1],
                                 [0,-1,1,0]])
Projector_topsurface = 0.5* np.array([[1,0,0,1],
                                      [0,1,1,0],
                                      [0,1,1,0],
                                      [1,0,0,1]])
Projector_bottomsurface = 0.5* np.array([[1,0,0,-1],
                                         [0,1,-1,0],
                                         [0,-1,1,0],
                                         [-1,0,0,1]])

def Skronecker(i,j):
    return np.kron(Sigma[i], Sigma[j])

def RuiChenModel(M,B,A): 
    M0 = (M-6*B)*Skronecker(3,0)
    Txyz = [(B * Skronecker(3,0) - 1j* (A/2.) * Skronecker(1,i)) for i in range(1,4)]
    return M0, Txyz[0], Txyz[1], Txyz[2]

def RuiChenBulkModel(kvec, M, B, A):
    k_abs = np.linalg.norm(kvec)

    Term_A = sum(kvec[i] * Sigma[i + 1] for i in range(3))
    Term_A = A*np.kron(Term_A, sigma_x)
    Term_B = (M+B*k_abs**2)*Skronecker(0,3)
    H_k = Term_A + Term_B
    return H_k

def TwoSurfaceModel(vf, m0, B, g, Mt, Mb, Es, V0):
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
    
    Hz = np.zeros((4,4), dtype=complex)
    
    return H0, Hy, Hx, Hz # Switching Hx and Hy to match the kwant code

def TwoSurfaceModel_originalbasis(vf, m0, B, g, Mt, Mb, Es, V0):
    M_plus = (Mt + Mb) / 2
    M_minus = (Mt - Mb) / 2
    # onsite term
    #Testing tauxtauy m0 term
    H0 =   +np.kron(tau_0, sigma_0) * Es \
           +np.kron(tau_x, sigma_0) * (m0 + 4*B) \
           +np.kron(tau_0, sigma_z) * g*M_plus \
           +np.kron(tau_z, sigma_z) * g*M_minus \
           +np.kron(tau_z, sigma_0) * V0
    # hopping in x direction from site (i, j) to (i + 1, j)
    Hx =   -np.kron(tau_x, sigma_0) * B  \
           -np.kron(tau_z, sigma_y) * 1j * vf/2
    # hopping in y direction from site (i, j) to (i, j + 1)
    Hy =   -np.kron(tau_x, sigma_0) * B  \
           +np.kron(tau_z, sigma_x) * 1j * vf/2
    
    Hz = np.zeros((4,4), dtype=complex)
    
    return H0, Hx, Hy, Hz

def TwoSurfaceWithBulk(vf, m0, B, g, Mt, Mb, Es, V0, Eb, B1):
    H0_surf, Hx_surf, Hy_surf, _ = TwoSurfaceModel_originalbasis(vf, m0, B, g, Mt, Mb, Es, V0)
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
    Hz = np.zeros((5,5), dtype=complex)
    return H0, Hy, Hx, Hz

def MBTModel(m0, M1, M2, A0, B0, C0, D1, D2):
    M0 = (C0 +2*D1 + 4*D2) * Skronecker(0,0) + (m0+ 2*M1 + 4*M2) * Skronecker(3,0)
    Tx = -D2 * Skronecker(0,0) - M2 * Skronecker(3,0) + 1j*A0/2.*Skronecker(1,2)
    Ty = -D2 * Skronecker(0,0) - M2 * Skronecker(3,0) - 1j*A0/2.*Skronecker(1,1)
    Tz = -D1 * Skronecker(0,0) - M1 * Skronecker(3,0) - 1j*B0/2.*Skronecker(2,0)
    return M0, Tx, Ty, Tz

def MBTBulkModel(kvec, m0, M1, M2, A0, B0, C0, D1, D2):
    kx, ky, kz = kvec
    # The order of tau and sigma is opposite compared the the function of RuiChenBulkModel
    H_k  = (C0 + D1*kz**2 + D2* (kx**2 + ky**2)) * Skronecker(0,0)
    H_k += (m0 + M1*kz**2 + M2* (kx**2 + ky**2)) * Skronecker(3,0)
    H_k += A0 * (ky*Skronecker(1,1)- kx*Skronecker(2,1)) + B0*kz*Skronecker(2,0)
    return H_k

def EmptyModel():
    M0 = 0*np.eye(2)
    Txyz = [-1j*sigma_y for i in range(1,4)]
    return M0, Txyz[0], Txyz[1], Txyz[2]

def create_hamiltonian_cube(onsite, tx, ty, tz, Nx, Ny, Nz):
    """
    Constructs the tight-binding Hamiltonian for a cubic lattice with open boundary conditions.
    
    Parameters:
    onsite : ndarray
        Onsite energy matrix of shape (n_orb, n_orb)
    tx : ndarray
        Hopping matrix along x-direction of shape (n_orb, n_orb)
    ty : ndarray
        Hopping matrix along y-direction of shape (n_orb, n_orb)
    tz : ndarray
        Hopping matrix along z-direction of shape (n_orb, n_orb)
    Nx, Ny, Nz : int
        Number of sites in x, y, and z directions
    
    Returns:
    H : sparse matrix
        The Hamiltonian in sparse matrix format
    """
    n_orb = onsite.shape[0]  # Number of orbitals per site
    N = Nx * Ny * Nz * n_orb  # Total number of states
    H = sp.lil_matrix((N, N), dtype=complex)
    # H = np.zeros((N, N), dtype=complex)
    
    def idx(ix, iy, iz, orb):
        return ((iz * Ny + iy) * Nx + ix) * n_orb + orb
    
    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                for orb1 in range(n_orb):
                    i1 = idx(ix, iy, iz, orb1)
                    for orb2 in range(n_orb):
                        i2 = idx(ix, iy, iz, orb2)
                        H[i1, i2] = onsite[orb1, orb2]
                        if ix < Nx - 1:
                            i2 = idx(ix + 1, iy, iz, orb2)
                            H[i1, i2] = tx[orb1, orb2]
                            H[i2, i1] = np.conj(tx[orb1, orb2])
                        if iy < Ny - 1:
                            i2 = idx(ix, iy + 1, iz, orb2)
                            H[i1, i2] = ty[orb1, orb2]
                            H[i2, i1] = np.conj(ty[orb1, orb2])
                        if iz < Nz - 1:
                            i2 = idx(ix, iy, iz + 1, orb2)
                            H[i1, i2] = tz[orb1, orb2]
                            H[i2, i1] = np.conj(tz[orb1, orb2])

    return H

def create_hamiltonian_cube_better(onsite, tx, ty, tz, Nx, Ny, Nz):
    """
    Alternative construction of the Hamiltonian without flattening the indices immediately,
    improving readability at the cost of memory efficiency.
    """
    n_orb = onsite.shape[0]
    H = np.zeros((Nx, Ny, Nz, n_orb, Nx, Ny, Nz, n_orb), dtype=complex)
    
    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                # Onsite term
                H[ix, iy, iz, :, ix, iy, iz, :] = onsite
                # Hopping in x direction
                if ix < Nx - 1:
                    H[ix, iy, iz, :, ix + 1, iy, iz, :] = tx
                    H[ix + 1, iy, iz, :, ix, iy, iz, :] = np.conj(tx.T)
                # Hopping in y direction
                if iy < Ny - 1:
                    H[ix, iy, iz, :, ix, iy + 1, iz, :] = ty
                    H[ix, iy + 1, iz, :, ix, iy, iz, :] = np.conj(ty.T)
                # Hopping in z direction
                if iz < Nz - 1:
                    H[ix, iy, iz, :, ix, iy, iz + 1, :] = tz
                    H[ix, iy, iz + 1, :, ix, iy, iz, :] = np.conj(tz.T)
    
    # Flatten to final sparse matrix
    H_sparse = sp.csr_matrix(H.reshape(Nx * Ny * Nz * n_orb, Nx * Ny * Nz * n_orb))
    return H_sparse

def Bandstructure_ky(ky, Nx, Nz, Hoppings, mt, mb, Magnetization = True):
    # Create Hamiltonian
    H0_new = Hoppings[0] + Hoppings[2] * np.exp(1j * ky)  + Hoppings[2].T.conj() * np.exp(-1j * ky)
    Hoppings_new = (H0_new, Hoppings[1], Hoppings[2], Hoppings[3])
    Ny = 1
    H_ky = create_hamiltonian_cube(*Hoppings_new, Nx, Ny, Nz)
    
    # Adding Magnatization term on the top and bottom boundaries in the z-direction
    if Magnetization:
        n_orb = Hoppings[0].shape[0]
        def idx(ix, iy, iz, orb):
            return ((iz * Ny + iy) * Nx + ix) * n_orb + orb
        Norb = Hoppings[0].shape[0]
        iy=0
        for ix in range(Nx):
            for n1 in range(Norb):
                for n2 in range(Norb):
                    for iz in range(1):
                        # Bottom surface
                        id1 = idx(ix, iy, iz, n1); id2 = idx(ix, iy, iz, n2)
                        H_ky[id1, id2] += mb[n1,n2]
                        H_ky[id2, id1] += np.conj(mb[n1,n2])
                        # Top surface
                        id1 = idx(ix, iy, -1-iz, n1); id2 = idx(ix, iy, -1-iz, n2)
                        H_ky[id1, id2] += mt[n1,n2]
                        H_ky[id2, id1] += np.conj(mt[n1,n2])
    
    # Computation
    eigenvalues, eigenvectors = compute_eigenvalues_and_vectors(H_ky, k=None)

    return eigenvalues, eigenvectors

def Bandstructure_kx(kx, Ny, Nz, Hoppings, mt, mb, Magnetization = True):
    # Create Hamiltonian
    H0_new = Hoppings[0] + Hoppings[1] * np.exp(1j * kx)  + Hoppings[1].T.conj() * np.exp(-1j * kx)
    Hoppings_new = (H0_new, np.zeros_like(Hoppings[1]), Hoppings[2], Hoppings[3])
    Nx = 1
    H_kx = create_hamiltonian_cube(*Hoppings_new, Nx, Ny, Nz)
    
    # Adding Magnatization term on the top and bottom boundaries in the z-direction
    if Magnetization:
        n_orb = Hoppings[0].shape[0]
        def idx(ix, iy, iz, orb):
            return ((iz * Ny + iy) * Ny + ix) * n_orb + orb
        Norb = Hoppings[0].shape[0]
        iy=0
        for ix in range(Ny):
            for n1 in range(Norb):
                for n2 in range(Norb):
                    for iz in range(1):
                        # Bottom surface
                        id1 = idx(ix, iy, iz, n1); id2 = idx(ix, iy, iz, n2)
                        H_kx[id1, id2] += mb[n1,n2]
                        H_kx[id2, id1] += np.conj(mb[n1,n2])
                        # Top surface
                        id1 = idx(ix, iy, -1-iz, n1); id2 = idx(ix, iy, -1-iz, n2)
                        H_kx[id1, id2] += mt[n1,n2]
                        H_kx[id2, id1] += np.conj(mt[n1,n2])
    
    # Computation
    eigenvalues, eigenvectors = compute_eigenvalues_and_vectors(H_kx, k=None)

    return eigenvalues, eigenvectors

def compute_eigenvalues_and_vectors(H, k=10):
    """
    Compute the lowest k eigenvalues and eigenvectors of the Hamiltonian H.
    
    Parameters:
    H : sparse matrix
        The Hamiltonian matrix
    k : int, optional
        Number of eigenvalues and eigenvectors to compute (default is 10)
    
    Returns:
    eigenvalues : ndarray
        The lowest k eigenvalues
    eigenvectors : ndarray
        Corresponding eigenvectors
    """

    if k is None:
        eigenvalues, eigenvectors = scipy.linalg.eigh(H.toarray())
    else:
        eigenvalues, eigenvectors = spla.eigsh(H, k=k, which='SM')
    sorted_indices = np.argsort(eigenvalues)  # Get indices that would sort the eigenvalues
    eigenvalues = eigenvalues[sorted_indices]  # Sort eigenvalues
    eigenvectors = eigenvectors[:, sorted_indices]  # Reorder eigenvectors to match sorted eigenvalues
    return eigenvalues, eigenvectors.T

def plot_eigen_spectrum(eigenvalues):
    """
    Plots the energy eigenvalue spectrum.
    
    Parameters:
    eigenvalues : ndarray
        Computed eigenvalues to be plotted.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(eigenvalues)), eigenvalues, 'bo', markersize=4)
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Energy")
    plt.title("Energy Eigenvalue Spectrum")
    plt.grid()
    plt.show()

def plot_eigenvector_xz_plane(eigenvector, Nx, Nz, n_orb, ax, cmap='Greens'):
    """
    Plot the spatial distribution of a selected eigenvector in the xz plane.
    
    Parameters:
    eigenvectors : ndarray
        Matrix where each column is an eigenvector
    eigen_index : int
        Index of the eigenvector to visualize
    Nx, Nz : int
        System dimensions
    n_orb : int
        Number of orbitals per site
    """
    eigenvector = eigenvector.reshape((Nx, 1, Nz, n_orb))
    prob_density = np.abs(eigenvector) ** 2
    prob_density = np.sum(prob_density, axis=(1,3))
    
    # plt.figure(figsize=(8, 6))
    # plt.imshow(prob_density.T, origin='lower', cmap='inferno', aspect='auto')
    # plt.colorbar(label="Probability Density")
    # plt.xlabel("X Index")
    # plt.ylabel("Z Index")
    # plt.show()

    z, x = np.meshgrid(range(Nz), range(Nx), indexing='ij')
    ax.scatter(z, x, s=prob_density * 5e3, c=prob_density, cmap=cmap, edgecolors=None)
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    return ax

def Wfc_to_Density(Wfc, Nz, Ny, Nx, norb):
    Wfc = Wfc.reshape((Nz, Ny, Nx, norb))
    prob_density = np.abs(Wfc) ** 2
    prob_density = np.sum(prob_density, axis=(1,3))
    prob_density = prob_density / np.sum(prob_density, axis=(0,1))
    return prob_density

def Wfc_to_Density_1Dx(Wfc, Nz, Ny, Nx, norb):
    Wfc = Wfc.reshape((Nz, Ny, Nx, norb))
    prob_density = np.abs(Wfc) ** 2
    prob_density = np.sum(prob_density, axis=(0,1,3))
    return prob_density
def Wfc_to_Density_1Dy(Wfc, Nz, Ny, Nx, norb):
    Wfc = Wfc.reshape((Nz, Ny, Nx, norb))
    prob_density = np.abs(Wfc) ** 2
    prob_density = np.sum(prob_density, axis=(0,2,3))
    return prob_density

def plot_density_xz_plane(prob_density, Nz, Nx, ax, markerscale=1e3,  cmap='Greens'):
    """
    Plot the spatial distribution of a selected eigenvector in the xz plane.
    
    Parameters:
    eigenvectors : ndarray
        Matrix where each column is an eigenvector
    eigen_index : int
        Index of the eigenvector to visualize
    Nx, Nz : int
        System dimensions
    n_orb : int
        Number of orbitals per site
    """
    x, z = np.meshgrid(range(Nx), range(Nz), indexing='ij')
    plot = ax.scatter(x, z, s=prob_density.T * markerscale, c=prob_density.T/2, cmap=cmap, edgecolors=None)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    fig = ax.figure
    cbar = fig.colorbar(plot ,ax=ax, orientation='horizontal', pad=0.1)
    return ax

def func1(Model, Para, Projector_topsurface, Projector_bottomsurface, ky=0.5, Bandindex=0, Nband=1, Names=['top', 'bottom']):
    Hoppings = Model(*Para.values())
    Nx, Ny, Nz = 20, 1, 1
    norb = Hoppings[0].shape[0]
    Nvec = Nz * Ny * Nx * norb
    Nbands = Nvec
    kylist = ky*np.array([-1, 0., 1])
    Nk = len(kylist)
    Elist = np.zeros((Nk, Nbands))
    StateList = np.zeros((Nk, Nbands, Nvec), dtype=complex)
    cmaps = ['Blues', 'Greens', 'Reds']
    # Parallel computation of bandstructure
    delayed_calculations = [delayed(Bandstructure_ky)(ky, Nx, Nz, Hoppings, None, None, Magnetization=False) for ky in kylist]
    results = compute(*delayed_calculations)
    # Collect results into Elist
    for i, result in enumerate(results):
        Elist[i] = result[0]
        StateList[i] = result[1]

    Evalues = Elist[:,Nbands//2+Bandindex:Nbands//2+Bandindex+Nband] # Doubly degenerate bands
    print(Evalues)
    Wfcs = StateList[:,Nbands//2+Bandindex:Nbands//2+Bandindex+Nband]
    # Normalization wrt unit occupation in each surface
    Wfcs = np.array([[Wfcs[i,j]/np.linalg.norm(Wfcs[i,j]) for j in range(Nband)]for i in range(Nk)])
    # print(np.array([[np.linalg.norm(Wfcs[i,j]) for j in range(Nband)]for i in range(Nk)]))

    # Plot both surfaces' density
    Density = np.array([[Wfc_to_Density_1Dx(Wfcs[i,j], Nz, Ny, Nx, norb) for j in range(Nband)]for i in range(Nk)])
    # Sum over the two bands
    Density = Density.sum(axis=1)

    Wfcs_top = np.array([[np.einsum('ab,cdeb->cdea',Projector_topsurface, Wfcs[i,j].reshape(Nz,Ny,Nx,norb)).reshape(Nz*Ny*Nx*norb) 
                        for j in range(Nband)]
                        for i in range(Nk)])
    Density_top = np.array([[Wfc_to_Density_1Dx(Wfcs_top[i,j], Nz, Ny, Nx, norb) for j in range(Nband)]for i in range(Nk)])
    Density_top = Density_top.sum(axis=1)

    Wfcs_bottom = np.array([[np.einsum('ab,cdeb->cdea'
                                    ,Projector_bottomsurface, Wfcs[i,j].reshape(Nz,Ny,Nx,norb)).reshape(Nz*Ny*Nx*norb)
                            for j in range(Nband)]
                            for i in range(Nk)])
    Density_bottom = np.array([[Wfc_to_Density_1Dx(Wfcs_bottom[i,j], Nz, Ny, Nx, norb) for j in range(Nband)]for i in range(Nk)])
    Density_bottom = Density_bottom.sum(axis=1)


    fig = plt.figure(figsize=(9, 3))
    selected_indices = [0, 2]  # or whatever indices you want for the pairs
    outer_gs = gridspec.GridSpec(len(selected_indices), 1, hspace=0.2)  # spacing between pairs
    colors = ['b','g', 'r']
    markerscale = 2e3
    x = np.arange(Nx)

    for pair_idx, i in enumerate(selected_indices):
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[pair_idx], hspace=0)  # no spacing within pair
        # Top surface
        ax_top = fig.add_subplot(inner_gs[0])
        ax_top.scatter(x, np.zeros(Nx), s=Density_top[i] * markerscale, c=colors[i], edgecolors=None, label='Top')
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_ylabel(Names[0])
        # ax_top.set_ylim([-0.5, 1.5])
        ax_top.set_title(f'ky = {kylist[i]:.1f}')
        ax_top.set_aspect('auto')

        # Bottom surface
        ax_bottom = fig.add_subplot(inner_gs[1])
        ax_bottom.scatter(x, np.ones(Nx), s=Density_bottom[i] * markerscale, c=colors[i], edgecolors=None, label='Bottom')
        ax_bottom.set_xticks([])
        ax_bottom.set_yticks([])
        ax_bottom.set_ylabel(Names[1])
        # ax_bottom.set_ylim([-0.5, 1.5])
        ax_bottom.set_aspect('auto')

    fig.tight_layout()
    return fig

def Plot_TI_Structure(ax, Mtop=1, MBottom=-1, text=None):
    import matplotlib.patches as patches
    # Tunable parameters
    layer_length = 4.0
    layer_thickness = 0.5
    ti_thickness = 0.7
    margin = 0.5
    gap = 0.0  # No space between layers

    # Derived positions
    bottom_y = 0
    ti_y = bottom_y + layer_thickness + gap
    top_y = ti_y + ti_thickness + gap
    fig_height = top_y + layer_thickness
    center_x = margin + layer_length / 2
    center_y = fig_height / 2
    # Draw layers
    bottom_mag = patches.Rectangle((margin, bottom_y), layer_length, layer_thickness, linewidth=0, edgecolor='k', facecolor='pink', alpha=1)
    ax.add_patch(bottom_mag)
    ti = patches.Rectangle((margin, ti_y), layer_length, ti_thickness, linewidth=0, edgecolor='k', facecolor='lightgray')
    ax.add_patch(ti)
    ax.text(center_x, center_y, text, ha='center', va='center', fontsize=16, color='k')
    top_mag = patches.Rectangle((margin, top_y), layer_length, layer_thickness, linewidth=0, edgecolor='k', facecolor='lightblue', alpha=1)
    ax.add_patch(top_mag)

    # Arrow parameters
    arrow_length = layer_thickness * 0.7
    head_width = 0.08
    head_length = 0.08

    # Bottom layer arrows
    x_arrows = np.linspace(margin + 0.2, margin + layer_length - 0.2, 5)
    y_bottom_center = bottom_y + layer_thickness / 2
    for x in x_arrows:
        y_start = y_bottom_center - 0.5 * arrow_length * MBottom
        ax.arrow(
            x, y_start,
            0, MBottom * arrow_length,
            head_width=head_width, head_length=head_length,
            fc='k', ec='k', linewidth=1., length_includes_head=True
        )

    # Top layer arrows
    y_top_center = top_y + layer_thickness / 2
    for x in x_arrows:
        y_start = y_top_center - 0.5 * arrow_length * Mtop
        ax.arrow(
            x, y_start,
            0, Mtop * arrow_length,
            head_width=head_width, head_length=head_length,
            fc='k', ec='k', linewidth=1., length_includes_head=True
        )

    ax.set_xlim(0, margin * 2 + layer_length)
    ax.set_ylim(0, fig_height)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.axis('off')
    # ax.set_aspect('equal')
    return ax