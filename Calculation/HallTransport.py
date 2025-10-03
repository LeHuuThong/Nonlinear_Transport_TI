# For calculations
import kwant
import tinyarray
import numpy as np
from scipy.stats import norm
import numpy as np
import copy
import itertools
import json

# For plotting
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe


# For parallelization
import dask
from dask import delayed, compute
from dask.distributed import Client
from dask import config
eye = np.eye(6)
delta_ki = eye[np.newaxis, :, :]  # shape (1, Nleads, Nleads)
delta_kj = eye[:, np.newaxis, :]  # shape (Nleads, 1, Nleads)
mask = delta_ki + delta_kj        # shape (Nleads, Nleads, Nleads)

class HamiltonianBuilder:
    # Constructor
    def __init__(self, Hoppings, L, W, nW, nL, eU_values, lead_pos=None, Hoppings_leads=None):
        """
        Constructing system with width W, length L, 
        nW blocks in the width direction, nL blocks in the length direction, 
        and electrostatic potential eU_values.
        """
        self.Hoppings = Hoppings
        self.dimH = Hoppings[0].shape[0]
        self.Nleads = None
        self.L = L
        self.W = W
        self.nW = nW
        self.nL = nL
        self.eU_values = eU_values
        self.eU = None
        self.eV = None
        self.system = None  
        self.leads = None
        self.lattice = None
        self.Elist = None
        self.lead_pos = lead_pos
        self.Hoppings_leads = Hoppings_leads
    # Methods
    def create_system(self):
        self.Nleads = 6
        if self.eU is None:
            self.eU = self.Constructing_eU(self.W, self.L, self.nW, self.nL, self.eU_values)
        if self.eV is None:
            self.eV = np.zeros(self.Nleads)
        self.system, self.leads, self.lattice = self.Build_system(self. W, self. L, *self.Hoppings, self.eU, self.eV, self.lead_pos, self.Hoppings_leads) 
    @staticmethod
    def Build_system(W: int, L: int, H0, Hx, Hy, eU, eV, lead_pos=None, Hoppings_leads=None):
        """
        Build a system with a Hall bar geometry and six leads.
        W: int, length of the Hall bar
        L: int, width of the Hall bar
        H0: array, dim = #orbitals, onsite energy
        eU: array, dim = L.W, electrostatic potentials
        Hx: array, dim = #orbitals, hopping energy in x direction
        Hy: array, dim = #orbitals, hopping energy in y direction
        eV: array, dim = #leads = 6, leads' chemical potentials
        Return: kwant.builder.Builder, list of kwant.builder.Builder, kwant.lattice.TranslationalSymmetry
        """
        if Hoppings_leads is None:
            H0_leads = H0
            Hx_leads = Hx
            Hy_leads = Hy
        else:
            H0_leads, Hx_leads, Hy_leads = Hoppings_leads
        # spin and orbital degree of freedom
        id = np.eye(H0.shape[0])

        def onsite_energy(site, H0, eU):
            x, y = site.pos # Our notation: x goes from 0 to L, y goes from 0 to W
            return H0 + eU[int(x),int(y)]* id

        a = 1 # lattice constant
        xvec = np.array([a, 0])
        yvec = np.array([0, a])

        lattice = kwant.lattice.general([xvec,yvec], norbs=H0.shape[0]) # square lattice, one orbital per site
        system = kwant.Builder() # system object
        
        # Hamiltonian in the scattering region
        # Kwant will automatically add the reverse hoppings to make the Hamiltonian hermitian
        system[(lattice(x, y) for x in range(L) for y in range(W))] = lambda site: onsite_energy(site, H0, eU)
        system[kwant.builder.HoppingKind((1,0), lattice, lattice)] = Hx
        system[kwant.builder.HoppingKind((0,1), lattice, lattice)] = Hy

        # Hamiltonian in the leads
        # Six-terminal Hall bar with lead-n in clockwise order


        LeadsVectors = [-xvec, yvec, yvec, xvec, -yvec, -yvec]
        if lead_pos is None:
            lead_pos = [[0, W], [1*L//5, 2*L//5], [3*L//5, 4*L//5], [0, W], [3*L//5, 4*L//5], [1*L//5, 2*L//5]]
        LeadsPositions = [(lattice(0, j) for j in range(lead_pos[0][0],lead_pos[0][1])),
                          (lattice(i, 0) for i in range(lead_pos[1][0],lead_pos[1][1])),
                          (lattice(i, 0) for i in range(lead_pos[2][0],lead_pos[2][1])),
                          (lattice(0, j) for j in range(lead_pos[3][0],lead_pos[3][1])),
                          (lattice(i, 0) for i in range(lead_pos[4][0],lead_pos[4][1])),
                          (lattice(i, 0) for i in range(lead_pos[5][0],lead_pos[5][1])),
                          ]
        Leads = [0 for _ in range(6)]
        for i in range(6):
            lead = kwant.Builder(kwant.TranslationalSymmetry(lattice.vec(LeadsVectors[i])))
            lead[LeadsPositions[i]] = H0_leads + eV[i] * id
            lead[kwant.builder.HoppingKind(xvec, lattice, lattice)] = Hx_leads
            lead[kwant.builder.HoppingKind(yvec, lattice, lattice)] = Hy_leads
            system.attach_lead(lead)
            Leads[i] = lead

        return system, Leads, lattice
    @staticmethod
    def Constructing_eU(W, L, nW, nL, eU_values):
        """
        Construct the electrostatic potential matrix eU from eU_block. 
        W: int, width
        L: int, length
        nW: int, number of blocks in the width direction
        nL: int, number of blocks in the length direction
        eU_block: array, dim = nW.nL, electrostatic potential in each block
        """
        eU = np.zeros((W, L))
        block_width = W // nW
        block_length = L // nL

        if eU_values.shape != (nW, nL):
            raise ValueError(f"Dimension of eU_block {eU_values.shape} does not match the expected dimensions ({nW}, {nL})")

        for i in range(nW):
            for j in range(nL):
                eU[i*block_width:(i+1)*block_width, j*block_length:(j+1)*block_length] = eU_values[i,j]

        # Handle the remaining rows if W is not divisible by nW
        if W % nW != 0:
            for j in range(nL):
                eU[nW*block_width:, j*block_length:(j+1)*block_length] = eU_values[-1,j]

        # Handle the remaining columns if L is not divisible by nL
        if L % nL != 0:
            for i in range(nW):
                eU[i*block_width:(i+1)*block_width, nL*block_length:] = eU_values[i,-1]

        # Handle the remaining block if both W and L are not divisible by nW and nL
        if W % nW != 0 and L % nL != 0:
            eU[nW*block_width:, nL*block_length:] = eU_values[-1,-1]

        return eU.T
    @staticmethod
    def Plotsystem(system, asymetricband=None):
        kwant.plot(system)
        # check edge states
        lead = system.leads[0].finalized()
        Energy_bands = kwant.physics.Bands(lead)
        momenta = np.linspace(-np.pi, np.pi, 401)
        energies = [Energy_bands(k) for k in momenta]       
        # Calculate bandgap
        E0 = Energy_bands(0)
        Nband = E0.shape[0]
        E0 = Energy_bands(0)
        if asymetricband is None:
            Bandgap = E0[Nband//2] - E0[Nband//2 - 1]
            print(f"Bandgap = {Bandgap:.4f}")
        if asymetricband is not None:
            Edifference = [E0[i+1] - E0[i] for i in range(Nband- 1)]
            gapindex, Bandgap = min(enumerate(Edifference), key = lambda x: x[1])
            print(f"Bandgap = {Bandgap:.4f}")
            print(E0[gapindex-5:gapindex+5])

        fig, ax = plt.subplots()
        ax.plot(momenta, energies)
        ax.set_xlabel("momentum")
        ax.set_ylabel("energy")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-1, 1)
        ax.grid()
        return fig, ax
    @staticmethod
    def DensityPlot(data):
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap='viridis',vmin=0, vmax=np.max(data), origin='lower')
        ax.set_xlim(-1, data.shape[1]+1)
        ax.set_ylim(-1, data.shape[0]+1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.1, pad=0.1,orientation='horizontal',use_gridspec=True,aspect=30)
        return fig, ax, im, cbar

class HallResistance_1st:
    # Constructor
    def __init__(self, Hamiltonian, energy, det_threshold=1e-9):
        # Instance attributes
        self.Hamiltonian = Hamiltonian
        self.energy = energy
        self.current = [1, 0, 0, -1, 0, 0]
        self.det_threshold = det_threshold
        # First order
        self.smatrix = None
        self.transmission = None # T_{ij} is the transmission from lead j to lead i. This is the tranposed of the one in the Notes
        self.G1 = None # I = G V. This is \tilde G in the Notes
        self.G1_det = None
        self.idrop = None
        self.R1 = None #idrop dependent
        self.voltages = None # depend on i_drop
        self.Hall_resistance = None
        self.Hall_resistance_avg = None

    # Methods
    # Final Evaluation
    def evaluate_smatrix(self):
        Nleads = self.Hamiltonian.Nleads
        system = self.Hamiltonian.system.finalized()

        self.smatrix = kwant.smatrix(system, self.energy)

        # self.transmission = np.array([[self.smatrix.transmission(j,i) * (1 if j!=i else 0) #
        #                                for i in range(Nleads)] for j in range(Nleads)])
        self.transmission = self.smatrix_to_transmission(self.smatrix, Nleads)                     
        
        # self.G1 = self.smatrix.conductance_matrix() # gives the same result as the following line
        self.G1 = self.conductance_from_transmission(self.transmission)

        self.idrop = 0
        G1_reduced = np.delete(np.delete(self.G1, self.idrop, axis=0), self.idrop, axis=1)
        G1_det = np.linalg.det(G1_reduced)
        if abs(G1_det) > self.det_threshold:
            self.G1_det = G1_det
            self.R1 = np.linalg.inv(G1_reduced)

    def evaluate_Hall_resistance(self):
        if self.R1 is None:
            self.evaluate_smatrix()
        if self.G1_det is not None:
            voltage = HallResistance_1st.calculate_voltage(self.R1, self.current, self.idrop)
            Hall_resistance = HallResistance_1st.calculate_Hall_resistance(voltage, self.current)
            self.voltages = voltage
            self.Hall_resistance = Hall_resistance
    # Testing methods
    def test_Conductance_1st(self):
        self.test_Tmatrix_currentconservation(self.transmission)
        print("The transmission matrix satisfies current conservation.")
        self.test_Gmatrix_currentconservation(self.G1)
        print("All sums over each index of G1 are zero.")
    @staticmethod
    def test_Tmatrix_currentconservation(transmission):
        sum0 = np.sum(transmission, axis=0)
        sum1 = np.sum(transmission, axis=1)
        assert np.allclose(sum0, sum1, rtol=1e-4), "The transmission matrix does not satisfy current conservation."  
        return True
    @staticmethod
    def test_Gmatrix_currentconservation(G1):
        sum0 = np.sum(G1, axis=0)
        sum1 = np.sum(G1, axis=1)
        assert np.allclose(sum0, 0, rtol=1e-4), "The sum over the first index of the conductance matrix is not zero."
        assert np.allclose(sum1, 0, rtol=1e-4), "The sum over the second index of the conductance matrix is not zero." 
        return True
    
    def test_idrop(self):
        """
        Test the the Hall resistances remain the same when changing the i_drop index.
        """
        Nleads = self.Hamiltonian.Nleads
        for i_drop in range(Nleads):
            R1 = self.invert_conductance(self.G1, i_drop)
            voltage = self.calculate_voltage(R1, self.current, i_drop)
            Hall_resistance = self.calculate_Hall_resistance(voltage, self.current)
            
            assert self.dict_allclose(Hall_resistance, self.Hall_resistance), \
            f"The Hall resistance changes when changing the i_drop index from {self.idrop} to {i_drop}."
        
        print("The Hall resistances remain the same when changing the i_drop index.")
    # Calculations methods
    @staticmethod
    def smatrix_to_transmission(smatrix, Nleads):
        T = np.array([[smatrix.transmission(j,i) * (1 if j!=i else 0) #
                                       for i in range(Nleads)] for j in range(Nleads)])
        return T
    @staticmethod
    def dict_allclose(dict1, dict2, rtol=1e-4):
        """
        Compare two dictionaries to check if their values are close within a tolerance.

        Parameters:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.

        Returns:
        bool: True if all values are close within the specified tolerance, False otherwise.
        """
        if dict1.keys() != dict2.keys():
            return False

        for key in dict1:
            if not np.allclose(dict1[key], dict2[key], rtol=rtol):
                return False

        return True
    @staticmethod
    def invert_conductance(conductance, i_drop):
        """ 
        Invert the conductance matrix to get the resistance matrix
        One needs to drop the i_drop index before inversion
        The relation is $I = G V$ and
        $$\tilde V = R \tilde I$$
        where $\tilde I$ has the I[i_drop] dropped
        while $\tilde V$ has the V[i_drop] dropped and other components is subtracted by V[i_drop]
        """
        # delete row and column of lead i_drop in conductance matrix
        conductance_reduced = np.delete(np.delete(conductance, i_drop, axis=0), i_drop, axis=1)
        # inversion
        resistance = np.linalg.inv(conductance_reduced)

        return resistance
    @staticmethod
    def calculate_voltage(resistance, current, i_drop):
        """
        Calculate the voltage differences
        """
        # delete row i_drop in current
        current_reduced = np.delete(current, i_drop, axis=0)
        # calculate the voltage drop
        voltage_difference = np.dot(resistance, current_reduced)
        voltage = np.insert(voltage_difference, i_drop, 0, axis=0)

        return voltage
    @staticmethod
    def calculate_Hall_resistance(voltage, current):
        """
        Calculate the conductances in the Hall bar
        The lead index follows the clockwise order, starting from the left lead where the driving current is injected
        """
        
        if current[0] == 0:
            raise ValueError("The current at lead 0 cannot be zero.")
        
        Hall_conductances = {
            # Longitudinal conductances on the top and bottom edges
            # Related by mirror y
            "R12": (voltage[1] - voltage[2]) / current[0],
            "R54": (voltage[5] - voltage[4]) / current[0],
            
            # Hall conductances on the left and right sides of the sample
            # Related by mirror x
            "R15": (voltage[1] - voltage[5]) / current[0],
            "R24": (voltage[2] - voltage[4]) / current[0]
        }
        return Hall_conductances
    @staticmethod
    def conductance_from_transmission(transmission):
        """ 
        Converting transmission matrix to conductance matrix
        transmission = T_{ij} is the transmission from lead j to lead i
        At first order, zero temperature:
        I_i = T_{ij} (V_i - V_j)
        In terms of conductance:
        I_i = C_{ij} V_j
        So the relation between conductance and transmission is:
        C_{ij} = \delta_{ij} \sum_k T_{ik} - T_{ij}
        """
        shape = transmission.shape
        conductance= np.zeros(shape)
        sum = np.sum(transmission, axis=1)
        for i, j in itertools.product(range(shape[0]), range(shape[1])):
                conductance[i][j] = - transmission[i][j] + sum[i] * (1 if i==j else 0)
                # if i==j: conductance[i,j] += np.sum(transmission[i,:])
        return conductance

class HallResistance_2nd(HallResistance_1st):
    # Constructor
    def __init__(self, Hamiltonian, energy, dE=1e-6):
        # Instance attributes
        super().__init__(Hamiltonian, energy)
        # Second order
        self.wavefunctions = None # the first index (via method) is the lead index, then the mode index, then the site index, then the orbital index (after reshaping)
        self.chara_potential = None   # the first index is the lead index, then the site index
        self.injectivity_sum = None     #
        self.ldos = None
        self.dE = dE
        self.transmission_dE = None   # All transmission matrices and derivatives are tranposed compared to the Notes
        self.transmission_deV = None 
        self.transmission_dE_deV_error = None
        self.transmission_derivative_test = None
        self.G2 = None # symmetrized G2 # This is \tilde G2 in the Notes
        self.G2a = None
        self.G2b = None
        self.G2_raw = None
        # self.G2_timereversed = None
        self.R2 = None
        self.voltages_2nd = None # depend on i_drop
        self.Hall_resistance_2nd = None
        # local currents
        self.localcurrents = None
        self.localcurrents_topsurface = None
        self.localcurrents_bottomsurface = None
        self.localcurrent_deV = None
        self.localcurrent_1st = None

    # Methods
    # Final evaluations
    def evaluate_wavefunctions(self):
        dimH = self.Hamiltonian.dimH
        Nleads = self.Hamiltonian.Nleads
        W = self.Hamiltonian.W
        L = self.Hamiltonian.L
        system = self.Hamiltonian.system.finalized()
        self.wavefunctions = kwant.wave_function(system, self.energy)
        self.chara_potential, self.injectivity_sum = self.calculate_chara_potential(self.wavefunctions, W, L, dimH, Nleads)
    
    def evaluate_transmission_derivatives(self):
        self.transmission_dE = self.calculate_transmission_dE(self.transmission, self.Hamiltonian, self.energy, self.dE)
        self.transmission_deV = self.calculate_transmission_deV_parallel(self.transmission, self.Hamiltonian, self.chara_potential, self.energy, self.dE)
        self.transmission_dE_deV_error = self.calculate_relative_error(self.transmission_dE, self.transmission_deV)
    
    def evaluate_HallResistance_2nd(self):
        if self.Hall_resistance is None:
            self.evaluate_Hall_resistance()
        if self.wavefunctions is None:
            self.evaluate_wavefunctions()
        if self.transmission_deV is None:
            self.evaluate_transmission_derivatives()
        self.G2, self.G2a, self.G2b, self.G2_raw = self.calculate_G2(self.transmission_deV, self.Hamiltonian.Nleads)
        # _, _, _, self.G2_timereversed = self.calculate_G2(np.transpose(self.transmission_deV, axes=(0,2,1)), self.Hamiltonian.Nleads)
        self.R2 = self.calculate_R2(self.R1, self.G2, self.idrop)
        self.voltages_2nd = self.calculate_voltage_2nd(self.R2, self.current, self.idrop)
        self.Hall_resistance_2nd = self.calculate_Hall_resistance_2nd(self.voltages_2nd, self.current)

    def evaluate_localcurrent(self):
        if self.wavefunctions is None:
            self.evaluate_wavefunctions()

        Projector_topsurface = 0.5* np.array([[1,0,0,1],
                                             [0,1,1,0],
                                             [0,1,1,0],
                                             [1,0,0,1]])
        Projector_bottomsurface = 0.5* np.array([[1,0,0,-1],
                                                [0,1,-1,0],
                                                [0,-1,1,0],
                                                [-1,0,0,1]])
        self.localcurrents = self.calculate_localcurrent(self.Hamiltonian, self.wavefunctions)
        self.localcurrents_topsurface = self.calculate_localcurrent(self.Hamiltonian, self.wavefunctions, operator=Projector_topsurface)
        self.localcurrents_bottomsurface = self.calculate_localcurrent(self.Hamiltonian, self.wavefunctions, operator=Projector_bottomsurface)

    def evaluate_localcurrent_1st(self):
        self.localcurrent_deV = self.calculate_localcurrent_deV_1st(self.localcurrent_sum, self.Hamiltonian, self.chara_potential, self.energy, self.dE)
        self.localcurrent_1st = np.sum([self.localcurrent_deV[i] * self.voltages[i] for i in range(self.Hamiltonian.Nleads)])
    @staticmethod
    def calculate_localcurrent(Hamiltonian, wavefunctions, operator = np.eye(4)):
        dimH = Hamiltonian.dimH
        system = Hamiltonian.system.finalized()
        Nleads = Hamiltonian.Nleads

        current_calculator = kwant.operator.Current(system, operator)
        current_leads = [None for _ in range(Nleads)]
        for i in range(Nleads):
            wavefunction_leadi = wavefunctions(i)
            N_modes = len(wavefunction_leadi)
            current_leads[i] = np.sum([current_calculator(wavefunction_leadi[i]) for i in range(N_modes)], axis=0)
        return current_leads
    @staticmethod
    def calculate_localcurrent_deV_1st(localcurrent, Hamiltonian, chara_potential, E, dE):
        """
        Calculate the derivative of localcurrent w.r.t. eV using Dask for parallelization
        """
        Nleads = Hamiltonian.Nleads
        localcurrent_deV_1st = np.zeros((Nleads, len(localcurrent)))
        def calculate(localcurrent, Hamiltonian, chara_potential, E, dE, i0):
            Hamiltonian1 = copy.deepcopy(Hamiltonian)
            Hamiltonian1.eV = dE * np.array([1 if j == i0 else 0 for j in range(Hamiltonian.Nleads)])
            Hamiltonian1.eU = dE * chara_potential[i0].reshape(Hamiltonian1.L, Hamiltonian1.W)
            Hamiltonian1.create_system()
            system = Hamiltonian1.system.finalized()
            wavefunctions1 = kwant.wave_function(system, E)
            localcurrent1 = HallResistance_2nd.calculate_localcurrent(Hamiltonian1, wavefunctions1)
            return (localcurrent1 - localcurrent) / dE
        delayed_results = [delayed(calculate)(localcurrent, Hamiltonian, chara_potential, E, dE, i) for i in range(Nleads)]
        results = compute(*delayed_results)

        for i, result in enumerate(results):
            localcurrent_deV_1st[i] = result

        return localcurrent_deV_1st
    @staticmethod
    def plot_current(system, localcurrent):

        fig, ax = plt.subplots()
        kwant.plotter.current(system, localcurrent, ax=ax, show=True)

        return fig, ax
    # Testings
    def test_chara_potential(self):
        L = self.Hamiltonian.L
        W = self.Hamiltonian.W
        if self.wavefunctions is None:
            self.evaluate_wavefunctions()
        if self.ldos is None:
            self.ldos = kwant.ldos(self.Hamiltonian.system.finalized(), self.energy)

        # HamiltonianBuilder.DensityPlot(np.sum([self.chara_potential[i].reshape(L,W).T for i in range(6)],axis=0))

        # print("Plotting the injectivity sum (multiplied by 2$\pi$)")
        # HamiltonianBuilder.DensityPlot(self.injectivity_sum.reshape(L,W).T/2/np.pi)
        
        ldos_sum = np.sum(self.ldos.reshape(-1, self.Hamiltonian.dimH), axis=1)
        print("Plotting the LDOS sum")
        HamiltonianBuilder.DensityPlot(ldos_sum.reshape(L,W).T)
        pass

    def test_transmission_derivatives_error(self, dE_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]):
        self.transmission_derivative_test = {}
        self.transmission_derivative_test['dE'] = dE_list
        if self.smatrix is None:
            self.evaluate_smatrix()
        if self.wavefunctions is None:
            self.evaluate_wavefunctions()
        
        def calculate_error(transmission, Hamiltonian, chara_potential, E, dE):
            transmission_dE = HallResistance_2nd.calculate_transmission_dE(transmission, Hamiltonian, E, dE)
            transmission_deV = HallResistance_2nd.calculate_transmission_deV_parallel(transmission, Hamiltonian, chara_potential, E,dE)
            error = HallResistance_2nd.calculate_relative_error(transmission_dE, transmission_deV)
            return np.abs(error)
        
        delayed_results = [delayed(calculate_error)(self.transmission, self.Hamiltonian, self.chara_potential, self.energy, dE) \
                            for dE in dE_list]
        error_list = compute(*delayed_results)
        
        self.transmission_derivative_test['error'] = error_list

        print("Plotting the relative error of transmission derivatives")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.loglog(dE_list,self.transmission_derivative_test['error'], marker = 'o')
        ax.set_xlabel(r'$\delta E$')
        ax.set_ylabel(r'Relative error $\varepsilon_T$')
        ax.grid()
        return fig, ax
    
    def test_transmission_derivatives_currentconservation(self):
        self.test_Tmatrix_currentconservation(self.transmission_dE)
        print("The transmission_dE satisfies current conservation.")
        for k in range(self.Hamiltonian.Nleads):
            self.test_Tmatrix_currentconservation(self.transmission_deV[k,:,:])
        print("The transmission_deV (for all leads) satisfy current conservation.")

    def test_conductance_2nd(self):
        self.test_G2(self.G2)
        print("All sums over each index of G2 are zero.")
    
    def test_idrop_2nd(self):
        """
        Test the the Hall resistances remain the same when changing the i_drop index.
        """
        Nleads = self.Hamiltonian.Nleads
        for i_drop in range(Nleads):
            R1 = self.invert_conductance(self.G1, i_drop)
            R2 = self.calculate_R2(R1, self.G2, i_drop)
            voltage = self.calculate_voltage_2nd(R2, self.current, i_drop)
            Hall_resistance = self.calculate_Hall_resistance_2nd(voltage, self.current)

            assert self.dict_allclose(Hall_resistance, self.Hall_resistance_2nd), \
            f"The Hall resistance changes when changing the i_drop index from {self.idrop} to {i_drop}."

        print("The 2nd-order Hall resistances remain the same when changing the i_drop index.")
    @staticmethod
    def test_G2(G2):
        """
        Test that the sums over each index of G2 equal zero.
        """
        sum_over_k = np.sum(G2, axis=0)
        sum_over_i = np.sum(G2, axis=1)
        sum_over_j = np.sum(G2, axis=2)

        # print("Sum over 0 index of G2:", sum_over_k)
        # print("Sum over 1 index of G2:", sum_over_i)    
        # print("Sum over 2 index of G2:", sum_over_j)
        
        assert np.allclose(sum_over_k, 0, atol=1e-4), "Sum over 0 index of G2 is not zero."
        assert np.allclose(sum_over_i, 0, atol=1e-4), "Sum over 1 index of G2 is not zero."
        assert np.allclose(sum_over_j, 0, atol=1e-4), "Sum over 2 index of G2 is not zero."

        return True

    # Static-like methods
    # Second order
    @staticmethod
    def calculate_chara_potential(wavefunctions, W, L, dimH, Nleads):
        injectivity = np.zeros((Nleads,L*W))
        characteristic_potential = np.zeros((Nleads,L*W))


        for lead_index in range(Nleads):
            # Wavefunctions in the main region
            wf = wavefunctions(lead_index)
            wf = wf.reshape(wf.shape[0], -1, dimH)
            #Calculate the injectivity
            injectivity[lead_index] = np.sum(np.abs(wf)**2, axis=(0,2))

            
        injectivity_sum = np.sum(injectivity, axis=0)
        # Calculate injectivity of carriers from different leads
        # Use the wave functions and their incoming velocities
        for i in range(Nleads):
            characteristic_potential[i] = injectivity[i]/injectivity_sum
        return characteristic_potential , injectivity_sum
    @staticmethod
    def calculate_transmission_dE(transmission, Hamiltonian, E, dE):
        """
        Calculate the derivative of transmission matrix w.r.t. energy
        """
        Nleads = Hamiltonian.Nleads
        smatrix1 = kwant.smatrix(Hamiltonian.system.finalized(), E + dE)
        # transmission1 = np.array([[smatrix1.transmission(j,i) for i in range(Nleads)] for j in range(Nleads)])
        transmission1 = HallResistance_1st.smatrix_to_transmission(smatrix1, Nleads)                     
        transmission_dE = (transmission1 - transmission) / dE
        return transmission_dE
    @staticmethod
    def calculate_transmission_deV_parallel(transmission, Hamiltonian, chara_potential, E, dE):
        """
        Calculate the derivative of transmission matrix w.r.t. eV using Dask for parallelization
        """
        Nleads = Hamiltonian.Nleads
        transmission_deV = np.zeros((Nleads, Nleads, Nleads))

        def calculate_transmission_for_lead(Hamiltonian, chara_potential, transmission, E, dE, i0):
            Hamiltonian1 = copy.deepcopy(Hamiltonian)
            Hamiltonian1.eV += dE * np.array([1 if j == i0 else 0 for j in range(Hamiltonian.Nleads)])
            Hamiltonian1.eU += dE * chara_potential[i0].reshape(Hamiltonian1.L, Hamiltonian1.W)
            Hamiltonian1.create_system()
            smatrix1 = kwant.smatrix(Hamiltonian1.system.finalized(), E)
            # transmission1 = np.array([[smatrix1.transmission(j, i) for i in range(Hamiltonian.Nleads)] for j in range(Hamiltonian.Nleads)])
            transmission1 = HallResistance_1st.smatrix_to_transmission(smatrix1, Nleads)                     
            return (transmission1 - transmission) / dE

        delayed_results = [delayed(calculate_transmission_for_lead)(Hamiltonian, chara_potential, transmission, E, dE, i) for i in range(Nleads)]
        results = compute(*delayed_results)

        for i, result in enumerate(results):
            transmission_deV[i] = result

        return transmission_deV
    @staticmethod
    def symmetrize_tensor(T, axis1, axis2):
        return 0.5 * (T + np.swapaxes(T, axis1, axis2))
    @staticmethod
    def calculate_G2(transmission_deV, Nleads):
        """
        Calculate the second order conductance tensor
        $ I^{(2)}_i = G_{ijk} V_k (V_i-V_j) $
        $ I^{(2)}_i = \tilde G_{ijk} V_j V_k $
        $ G_{ijk} = 1/2 \partial_E T_{ij} (\delta_ki + \delta_kj) + \partial_eV_k T_{ij} $
        """
        G2 = np.zeros((Nleads, Nleads, Nleads))
        G2a = np.zeros((Nleads, Nleads, Nleads))
        G2b = np.zeros((Nleads, Nleads, Nleads))

        transmission_deV_sum = np.sum(transmission_deV, axis=0)
        # Corrected
        for k in range(Nleads):
            for i in range(Nleads):
                for j in range(Nleads):
                    G2a[i, j, k] = - 0.5 * transmission_deV_sum[i, j] * ((1 if k == i else 0) + (1 if k == j else 0))
                    # G2[i, j, k] = - 0.5 * transmission_deV_sum[i, j] * ((1 if k == i else 0) + (1 if k == j else 0)) \
                    #                + transmission_deV[k, i, j]
        G2b = np.transpose(transmission_deV, (1, 2, 0))
        G2 = G2a + G2b

        G2 = HallResistance_2nd.conductance_from_transmission(G2)
        G2 = HallResistance_2nd.symmetrize_tensor(G2, 1, 2)

        G2a = HallResistance_2nd.conductance_from_transmission(G2a)
        G2a = HallResistance_2nd.symmetrize_tensor(G2a, 1, 2)

        G2b = HallResistance_2nd.conductance_from_transmission(G2b)
        G2b = HallResistance_2nd.symmetrize_tensor(G2b, 1, 2)

        G2_raw = None
        return G2, G2a, G2b, G2_raw
    @staticmethod
    def calculate_R2(R1, G2, i_drop):
        """
        Calculate the 2nd order resistance tensor
        $ $
        """
        G2_reduced = np.delete(np.delete(np.delete(G2, i_drop, axis=0), i_drop, axis=1), i_drop, axis=2)
        R2 = -np.einsum('ab, bcd, ce, df -> aef', R1, G2_reduced, R1, R1)
        return R2
    @staticmethod
    def calculate_voltage_2nd(R2, current, i_drop):
        """
        Calculate the voltage differences
        """
        # delete row i_drop in current
        current_reduced = np.delete(current, i_drop, axis=0)
        # calculate the voltage drop
        voltage_difference = np.einsum('abc, b, c -> a', R2, current_reduced,current_reduced)
        voltage = np.insert(voltage_difference, i_drop, 0, axis=0)

        return voltage
    @staticmethod
    def calculate_Hall_resistance_2nd(voltage, current):
        """
        Calculate the conductances in the Hall bar
        The lead index follows the clockwise order, starting from the left lead where the driving current is injected
        """
        
        if current[0] == 0:
            raise ValueError("The current at lead 0 cannot be zero.")
        current2 = (current[0]**2)
        
        Hall_resistances = {
            # Longitudinal conductances on the top and bottom edges
            # Related by mirror y
            "R12": (voltage[1] - voltage[2]) / current2,
            "R54": (voltage[5] - voltage[4]) / current2,
            
            # Hall conductances on the left and right sides of the sample
            # Related by mirror x
            "R15": (voltage[1] - voltage[5]) / current2,
            "R24": (voltage[2] - voltage[4]) / current2
        }
        return Hall_resistances
    @staticmethod
    def calculate_relative_error(transmission_dE, transmission_deV):
        """
        Calculate the relative error between transmission_dE and the sum of transmission_deV.
        """
        error = (transmission_dE + np.sum(transmission_deV, axis=0))
        error_relative = np.linalg.norm(error, 'fro')/np.linalg.norm(transmission_dE, 'fro')
        return error_relative

class HallResistanceData:
    # Constructor
    def __init__(self, Hamiltonian, Elist, det_threshold=1e-9):    
        # Instance attributes
        self.Hamiltonian = Hamiltonian
        self.current = [1, 0, 0, -1, 0, 0]
        self.Elist = Elist
        self.det_threshold = det_threshold
        self.Elist_nonsingular = None
        self.gapindex = None
        self.dE_avg = None

        self.G1 = None
        self.G1_det = None
        self.Hall_resistance = None
        self.G1_avg = None
        self.Hall_resistance_avg = None
        self.voltages = None
        self.voltages_avg = None
    
    def evaluate_Hallresistance(self):
        def calculate_resistance(Hamiltonian, E):
            calculator = HallResistance_1st(Hamiltonian, E, det_threshold=self.det_threshold)
            calculator.evaluate_smatrix()
            calculator.evaluate_Hall_resistance()
            return calculator.Hall_resistance, calculator.G1, calculator.G1_det, calculator.voltages        

        Hall_resistancelist = []
        G1 = []
        G1_det = []
        Elist_nonsingular = []
        voltages = []

        delayed_calculations = [delayed(calculate_resistance)(self.Hamiltonian,E) for E in self.Elist]
        results = compute(*delayed_calculations)
        for i, result in enumerate(results):
            if result[2] is not None:
                Elist_nonsingular.append(self.Elist[i])
                Hall_resistancelist.append(result[0])
                G1.append(result[1])
                G1_det.append(result[2])
                voltages.append(result[3])
            else: 
                if self.gapindex is None: 
                    self.gapindex = i-1                

        G1_det = np.array(G1_det)
        self.G1_det = G1_det
        self.G1 = G1
        self.Hall_resistance = self.combine_dicts(Hall_resistancelist)
        self.Elist_nonsingular = np.array(Elist_nonsingular)
        self.Elist = self.Elist_nonsingular
        self.voltages = np.array(voltages)
        
    def evaluate_Hallresistance_avg(self, std_dev):
        self.dE_avg = std_dev
        N_energy = len(self.Elist_nonsingular)
        G1list_avg = HallResistanceData.smoothing(self.Elist_nonsingular, self.G1, std_dev)
        Hall_resistancelist_avg = [None for _ in range(N_energy)]
        voltages_avg = [None for _ in range(N_energy)]
        idrop = 0
        for i in range(N_energy):
            R1 = HallResistance_1st.invert_conductance(G1list_avg[i], 0)
            voltages_avg[i] = HallResistance_1st.calculate_voltage(R1, self.current, idrop)
            Hall_resistancelist_avg[i] = HallResistance_1st.calculate_Hall_resistance(voltages_avg[i], self.current)
        self.G1_avg = G1list_avg
        self.Hall_resistance_avg = self.combine_dicts(Hall_resistancelist_avg)
        self.voltages_avg = np.array(voltages_avg)
    
    # Static-like methods
    @staticmethod
    def get_gapindex(Elist_nonsingular):
        differences = np.diff(Elist_nonsingular)
        max_diff_index = np.argmax(differences)
        return max_diff_index
    @staticmethod
    def smoothing(Elist, Data, std_dev):
        Data_avg = np.zeros_like(Data)
        def calculate(E, Elist, Data):
            NE = len(Elist)
            Data_sum = 0.
            norm_sum = 0.
            for i1, E1 in enumerate(Elist):
                if i1==0: dE = Elist[1] - Elist[0]
                elif i1==NE-1: dE = Elist[-1] - Elist[-2]
                else: dE = (Elist[i1+1] - Elist[i1-1])/2
                f_delta = norm.pdf(E1, E, std_dev)
                Data_sum += Data[i1] * f_delta * dE
                norm_sum += f_delta * dE 
            return Data_sum / norm_sum    

        # # Serial computation
        # Data_avg = map(lambda E: calculate(E, Elist, Data), Elist)
        # Data_avg = list(Data_avg)

        # # Parallel computation
        delayed_calculations = [delayed(calculate)(E, Elist, Data) for E in Elist]
        results = compute(*delayed_calculations)
        Data_avg = results

        # # Average out using Gaussian distribution
        # for i0, e0 in enumerate(Elist):
        #     Data_sum = np.zeros_like(Data[0])
        #     norm_sum = 0.
        #     for i1, e1 in enumerate(Elist):
        #         if i1==0: dE = Elist[1] - Elist[0]
        #         elif i1==len(Elist)-1: dE = Elist[-1] - Elist[-2]
        #         else: dE = (Elist[i1+1] - Elist[i1-1])/2

        #         f_delta = norm.pdf(e1, e0, std_dev)
        #         Data_sum += Data[i1] * f_delta * dE
        #         norm_sum += f_delta * dE 
        #     Data_avg[i0] = Data_sum / norm_sum    

        return Data_avg
    @staticmethod
    def PlotHallConductance(Elist, Hall_resistance, gapindex = None, xmax=None):
        fig, ax = plt.subplots(figsize=(5,4))
        linestyles = ['-', '-', '-', '-']
        markers = ['o', 'o', 'o', 'o']
        colors = ['tab:blue', 'tab:blue', 'r', 'r']
        linewidths = [1, 0, 0, 1]
        markersizes = [0, 3, 3, 0]
        zorders = [1,2,2,1]
        labels = [r'$R^{\omega}_{xx,L}$', r'$R^{\omega}_{xx,R}$', r"$R'^{\omega}_{xy}$", r"$R^{\omega}_{xy}$"]
        for key, value in Hall_resistance.items():
            if gapindex is None:
                ax.plot(Elist, value, 
                        label=labels.pop(0),
                        linestyle=linestyles.pop(0),
                        marker = markers.pop(0),
                        color = colors.pop(0),
                        markersize=markersizes.pop(0),
                        linewidth=linewidths.pop(0),
                        zorder=zorders.pop(0)) 
            else:
                lines = ax.plot(Elist[:gapindex+1], value[:gapindex+1],
                            Elist[gapindex+1:], value[gapindex+1:],
                            label=labels.pop(0),
                            linestyle=linestyles.pop(0),
                            marker = markers.pop(0),
                            color = colors.pop(0),
                            markersize=markersizes.pop(0),
                            linewidth=linewidths.pop(0),
                            zorder=zorders.pop(0)) 
                ax.axvspan(Elist[gapindex], Elist[gapindex+1], color='lightgray', alpha=0.5, label=None)
                lines[1].set_label(None)
                
        ax.grid(True, linestyle=':', linewidth=1., zorder=-1)
        ax.set_xlabel(r'$E_F$')
        ax.set_ylabel('Resistance')
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        if xmax is not None:
            ax.set_xlim(-xmax, xmax)

        return fig, ax
    @staticmethod
    def combine_dicts(dict_list):
        combined_dict = {}
        for d in dict_list:
            for key, value in d.items():
                if key not in combined_dict:
                    combined_dict[key] = []
                combined_dict[key].append(value)
        return combined_dict
    @staticmethod
    def smoothing_nonuniform(x, y, std_dev):
        # from scipy.interpolate import interp1d
        # from scipy.ndimage import gaussian_filter1d
        # # Check if y is multi-dimensional
        # if y.ndim == 1:
        #     y = y[:, np.newaxis]
        
        # # Step 1: Interpolate to a uniform grid
        # x_uniform = np.linspace(np.min(x), np.max(x), len(x))
        # y_smooth = np.zeros_like(y)
        
        # for i in range(y.shape[1]):
        #     interp_func = interp1d(x, y[:, i], kind='linear', fill_value="extrapolate")
        #     y_uniform = interp_func(x_uniform)
            
        #     # Step 2: Apply Gaussian smoothing
        #     y_smooth_uniform = gaussian_filter1d(y_uniform, std_dev)
            
        #     # Step 3: Interpolate back to the original non-uniform grid
        #     interp_func_smooth = interp1d(x_uniform, y_smooth_uniform, kind='linear', fill_value="extrapolate")
        #     y_smooth[:, i] = interp_func_smooth(x)
        
        # return y_smooth.squeeze()
        pass
    @staticmethod
    def Plot_voltages(dE_list, voltages):
        fig, ax = plt.subplots()
        linestyles = ['--','-','-', '-', None, None]
        linewidths = np.array([1, 1, 1, 1, 0, 0])
        markers = [None, 'o', 'o', None,'o', 'o']
        markersizes = np.array([1,1,1,1,1,1])*2
        for i in range(1, voltages.shape[1]):
            ax.plot(dE_list, voltages[:,i], label=f'V{i}', 
                    alpha=1, 
                    linestyle = linestyles[i], 
                    linewidth=linewidths[i],
                    marker = markers[i],
                    markersize = markersizes[i])
        ax.set_xlabel('Chemical potential $\mu$')
        ax.set_ylabel('Voltage')
        ax.legend()
        ax.grid(True)
        return fig, ax

class HallResistanceData_2nd(HallResistanceData):
    # Constructor
    def __init__(self, Hamiltonian, Elist, dE=1e-6, ab_separation=False):
        super().__init__(Hamiltonian, Elist)
        self.dE = dE
        self.G2 = None
        self.G2_avg = None
        self.ab_separation = ab_separation
        self.G2a = None
        self.G2a_avg = None
        self.G2b = None
        self.G2b_avg = None

        self.voltages_2nd = None
        self.voltages_2nd_avg = None
        self.Hall_resistance_2nd = None
        self.Hall_resistance_2nd_avg = None
        self.Hall_resistance_a_2nd = None
        self.Hall_resistance_a_2nd_avg = None
        self.Hall_resistance_b_2nd = None
        self.Hall_resistance_b_2nd_avg = None
        self.transmission_dE_deV_error = None
        self.transmission_dE_norm = None
        self.transmission_deV_norm = None
        self.transmission_dE = None
        self.transmission_dE_avg = None
        self.transmission_dE_avg_norm = None
    
    def evaluate_Hallresistance_2nd(self):
        # Confirm the first order calculations were done
        if self.Hall_resistance is None:
            self.evaluate_Hallresistance()
        
        # Second order calculations
        def calculate_resistance_2nd(Hamiltonian, E):
            calculator = HallResistance_2nd(Hamiltonian, E , self.dE)
            calculator.evaluate_Hall_resistance()
            calculator.evaluate_HallResistance_2nd()
            return calculator.voltages_2nd,\
                    calculator.Hall_resistance_2nd,\
                    calculator.G2, \
                    calculator.G2a, \
                    calculator.G2b, \
                    calculator.transmission_dE_deV_error, \
                    calculator.transmission_deV, \
                    calculator.transmission_dE
        
        # Define quantities
        N_energy = len(self.Elist)
        Nleads = self.Hamiltonian.Nleads
        voltages = np.zeros((N_energy, Nleads))
        Hall_resistancelist_2nd = [None for _ in range(N_energy)]
        G2list = np.zeros((N_energy,Nleads,Nleads,Nleads))
        G2alist = np.zeros((N_energy,Nleads,Nleads,Nleads))
        G2blist = np.zeros((N_energy,Nleads,Nleads,Nleads))
        transmission_dE = np.zeros((N_energy, Nleads, Nleads))
        transmission_dE_deV_error = np.zeros(len(self.Elist))
        transmission_dE_norm = np.zeros(len(self.Elist))
        transmission_deV_norm = np.zeros(len(self.Elist))
        
        # Parallel computation
        delayed_calculations = [delayed(calculate_resistance_2nd)(self.Hamiltonian, E) for E in self.Elist]
        result_list = compute(*delayed_calculations)

        # Collecting results
        for i, result in enumerate(result_list):
            voltages[i,:], Hall_resistancelist_2nd[i], G2list[i], G2alist[i], G2blist[i], transmission_dE_deV_error[i], T_deV, T_dE = result
            transmission_deV_norm[i] = np.linalg.norm(np.sum(T_deV, axis=0), 'fro')
            transmission_dE_norm[i] = np.linalg.norm(T_dE, 'fro')
            transmission_dE[i] = T_dE

        # Storing outputs
        self.voltages_2nd = voltages
        self.G2 = G2list
        if self.ab_separation:
            self.G2a = G2alist
            self.G2b = G2blist
        self.Hall_resistance_2nd = self.combine_dicts(Hall_resistancelist_2nd)
        self.transmission_dE = transmission_dE
        self.transmission_dE_deV_error = transmission_dE_deV_error
        self.transmission_deV_norm = transmission_deV_norm
        self.transmission_dE_norm = transmission_dE_norm
    
    def evaluate_Hallresistance_2nd_avg(self, std_dev):
        # Make sure the first order calculations were done with the same std_dev
        self.evaluate_Hallresistance_avg(std_dev)
        def helper(G2list_avg):
            # Define quantities
            N_energy = len(self.Elist)
            voltages = np.zeros((N_energy, self.Hamiltonian.Nleads))
            Hall_resistancelist_2nd_avg = [None for _ in range(N_energy)]
            idrop = 0
            # Parallel computation
            def calculator(G1_avg, G2_avg, current):
                R1 = HallResistance_1st.invert_conductance(G1_avg, idrop)
                R2 = HallResistance_2nd.calculate_R2(R1, G2_avg, idrop)
                voltage = HallResistance_2nd.calculate_voltage_2nd(R2, current, idrop)
                Hall_resistance = HallResistance_2nd.calculate_Hall_resistance_2nd(voltage, current)
                return voltage, Hall_resistance
            arg_list = [(self.G1_avg[i], G2list_avg[i], self.current) for i in range(N_energy)]
            delayed_calculations = [delayed(calculator)(*arg) for arg in arg_list]
            result_list = compute(*delayed_calculations)
            for i, R in enumerate(result_list):
                voltages[i] = R[0]
                Hall_resistancelist_2nd_avg[i] = R[1]
                # print(type(self.combine_dicts(Hall_resistancelist_2nd_avg)))
            return voltages, self.combine_dicts(Hall_resistancelist_2nd_avg)

        self.G2_avg = HallResistanceData.smoothing(self.Elist, self.G2, std_dev)
        self.voltages_2nd_avg, self.Hall_resistance_2nd_avg = helper(self.G2_avg)

        if self.ab_separation:
            self.G2a_avg = HallResistanceData.smoothing(self.Elist, self.G2a, std_dev)
            _, self.Hall_resistance_a_2nd = helper(self.G2a)
            _, self.Hall_resistance_a_2nd_avg = helper(self.G2a_avg)

            self.G2b_avg = HallResistanceData.smoothing(self.Elist, self.G2b, std_dev)
            _, self.Hall_resistance_b_2nd = helper(self.G2b)
            _, self.Hall_resistance_b_2nd_avg = helper(self.G2b_avg)

        self.transmission_dE_avg = HallResistanceData.smoothing(self.Elist, self.transmission_dE, std_dev)
        self.transmission_dE_avg_norm = np.linalg.norm(self.transmission_dE_avg, 'fro', axis=(1,2))

    @staticmethod
    def PlotHallConductance_2nd(Elist, Hall_resistance_2nd, gapindex = None):
        fig, ax = plt.subplots(figsize=(5, 4))
        linestyles = ['-', '-', '-', '-']
        markers = ['o', 'o', 'o', 'o']
        colors = ['tab:blue', 'tab:blue', 'r', 'r']
        linewidths = [1, 0, 0, 1]
        markersizes = [0, 3, 3, 0]
        zorders = [1,2,2,1]
        labels = [r'$R^{2\omega}_{xx,L}$', r'$R^{2\omega}_{xx,R}$', r"$R'^{2\omega}_{xy}$", r"$R^{2\omega}_{xy}$"]
        for key, value in Hall_resistance_2nd.items():
            if gapindex is None:
                ax.plot(Elist, value, 
                        label=labels.pop(0),
                        linestyle=linestyles.pop(0),
                        marker = markers.pop(0),
                        color = colors.pop(0),
                        markersize=markersizes.pop(0),
                        linewidth=linewidths.pop(0),
                        zorder=zorders.pop(0)) 
            else:
                lines = ax.plot(Elist[:gapindex+1], value[:gapindex+1],
                            Elist[gapindex+1:], value[gapindex+1:],
                            label=labels.pop(0),
                            linestyle=linestyles.pop(0),
                            marker = markers.pop(0),
                            color = colors.pop(0),
                            markersize=markersizes.pop(0),
                            linewidth=linewidths.pop(0),
                            zorder=zorders.pop(0))
                ax.axvspan(Elist[gapindex], Elist[gapindex+1], color='lightgray', alpha=0.5, label=None)
                lines[1].set_label(None)
        ax.grid(True, linestyle=':', linewidth=1., zorder=-1)
        ax.set_xlabel('Chemical potential $\mu$')
        ax.set_ylabel('Second-order Hall Resistance')
        ax.legend()
        return fig, ax
    @staticmethod
    def PlotHallConductance_2nd_abseparation(Elist, Hall_resistance_2nd, Hall_resistance_a_2nd, Hall_resistance_b_2nd, gapindex = None):
        fig, ax = plt.subplots(figsize=(5, 4))
        linestyles = ['-', '-', '-', '-']
        markers = ['o', 'o', 'o', 'o']
        colors = ['tab:blue', 'gold', 'darkorange', 'r']
        linewidths = [2, 1.5, 1.5, 1]
        markersizes = [0, 0, 0, 0]
        zorders = [3,2,2,1]
        labels = [r'$R^{2\omega}_{xx}$', r'$R^{2\omega, (a)}_{xx}$', r'$R^{2\omega, (b)}_{xx}$']
        for Resistance in [Hall_resistance_2nd, Hall_resistance_a_2nd, Hall_resistance_b_2nd]:
            value = Resistance['R54']
            if gapindex is None:
                ax.plot(Elist, value, 
                        label=labels.pop(0),
                        linestyle=linestyles.pop(0),
                        marker = markers.pop(0),
                        color = colors.pop(0),
                        markersize=markersizes.pop(0),
                        linewidth=linewidths.pop(0),
                        zorder=zorders.pop(0),
                        path_effects=[pe.withStroke(linewidth=1, foreground="white")]) 
            else:
                lines = ax.plot(Elist[:gapindex+1], value[:gapindex+1],
                            Elist[gapindex+1:], value[gapindex+1:],
                            label=labels.pop(0),
                            linestyle=linestyles.pop(0),
                            marker = markers.pop(0),
                            color = colors.pop(0),
                            markersize=markersizes.pop(0),
                            linewidth=linewidths.pop(0),
                            zorder=zorders.pop(0),
                            path_effects=[pe.withStroke(linewidth=1, foreground="white")]) 
                lines[1].set_label(None)
                ax.axvspan(Elist[gapindex], Elist[gapindex+1], color='lightgray', alpha=0.5, label=None)
        ax.grid(True, linestyle=':', linewidth=1., zorder=-1)
        ax.set_xlabel('Chemical potential $\mu$')
        ax.set_ylabel('Hall Resistance')
        ax.legend()
        return fig, ax
    @staticmethod
    def PlotHallConductance_2nd_v1(Elist, Hall_resistance_2nd, ax = None, gapindex = None, downsample=1, 
                                    ylabel=None, xlabel=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 2.5))
        else:
            fig = ax.figure
        linestyles = ['-', '-', '-', '-']
        markers = ['o', 'o', 'o', 'o']
        colors = ['r', 'b', 'b', 'y']
        linewidths = [1.5, 1.5, 0, 0]
        markersizes = [0, 0, 0, 0]
        labels = ['$V^{2\omega}_{xx}$', '$V^{2\omega}_{xy}$', 'Hall R15', 'Hall R24']
        for key, value in Hall_resistance_2nd.items():
            if key=='R54' or key=='R24':
                if gapindex is None:
                    ax.plot(Elist[::downsample], value[::downsample], 
                            label=labels.pop(0),
                            linestyle=linestyles.pop(0),
                            marker = markers.pop(0),
                            color = colors.pop(0),
                            markersize=markersizes.pop(0),
                            linewidth=linewidths.pop(0)) 

                else:
                    lines = ax.plot(Elist[:gapindex+1:downsample], value[:gapindex+1:downsample],
                                Elist[gapindex+1::downsample], value[gapindex+1::downsample],
                                label=labels.pop(0),
                                linestyle=linestyles.pop(0),
                                marker = markers.pop(0),
                                color = colors.pop(0),
                                markersize=markersizes.pop(0),
                                linewidth=linewidths.pop(0)) 
                    lines[1].set_label(None)
                    
                    ax.axvspan(Elist[gapindex], Elist[gapindex+1], color='lightgray', alpha=0.5, label=None)

        ymax = np.max(np.abs(Hall_resistance_2nd['R54']))
        ax.set_ylim(-1.2*ymax, 1.2* ymax)
        ax.set_xlabel(xlabel)
        if ylabel is None: ylabel = "arb. unit"
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', linewidth=1., zorder=-1)
        return fig, ax
    @staticmethod
    def Plot_relative_error_with_norms(dE_list, error_list, transmission_dE_norm, transmission_deV_norm):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # Create a second y-axis to plot the norms
        ax1.plot(dE_list, transmission_dE_norm, marker='s', markersize=3, label='Transmission dE Norm', color='r', zorder = 1)
        ax1.plot(dE_list, transmission_deV_norm, marker='^', markersize=3, label='Transmission deV Norm', color='g', zorder = 1)
        ax1.set_ylabel('Norms', color='r')
        ax1.tick_params(axis='y', labelcolor='r')

        # Plot relative error on the left y-axis
        ax2.plot(dE_list, error_list, marker='o', markersize=3, label='Relative Error', color='b', zorder = 100)
        ax2.set_xlabel('Chemical potential $\mu$')
        ax2.set_ylabel('Relative error of transmission derivatives', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.grid(True)

        # Add legends separately
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        return fig, ax1, ax2