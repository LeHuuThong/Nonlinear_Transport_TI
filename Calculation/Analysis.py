# For calculations
import kwant
import tinyarray
import numpy as np
from scipy.stats import norm
import numpy as np
import copy
import itertools
import HallTransport as HallTransport
import Hamiltonians
from datetime import datetime

# For plotting
import matplotlib
from matplotlib import pyplot as plt

def SystemCreator(para, Model, system_size, eU = None, eV = None, lead_pos = None, asymmetricband = None, plotting=True, para_leads=None):
    # Create the system
    print("Creating the system")
    W, L = system_size
    Hoppings = Model(*para.values())
    if para_leads is not None:
        Hoppings_leads = Model(*para_leads.values())
    else:
        Hoppings_leads = None
    Hamiltonian = HallTransport.HamiltonianBuilder(Hoppings, W=W, L=L, nW=2, nL=5, eU_values=np.zeros((2,5)), lead_pos=lead_pos, Hoppings_leads=Hoppings_leads)
    Hamiltonian.eU = eU
    Hamiltonian.eV = eV
    Hamiltonian.create_system()
    # Plotting lattice and band structure
    if plotting is True:
        Hamiltonian.Plotsystem(Hamiltonian.system, asymmetricband)
    
    return Hamiltonian

def Testing_1st(Hamiltonian, E_check=-0.5):
    # Testings
    print("Testing the first order calculations")
    FirstOrder = HallTransport.HallResistance_1st(Hamiltonian, energy=E_check)
    FirstOrder.evaluate_Hall_resistance()
    FirstOrder.test_Conductance_1st()
    FirstOrder.test_idrop()

    return FirstOrder

def Testing_2nd(Hamiltonian, E_check=-0.5, dE_derivative = 1e-8):
    # Test and print out ldos for at E_check
    print("Testing the second order calculations")
    System_2nd = HallTransport.HallResistance_2nd(Hamiltonian, energy=E_check, dE=dE_derivative)
    System_2nd.evaluate_smatrix()
    System_2nd.evaluate_wavefunctions()
    System_2nd.evaluate_transmission_derivatives()
    System_2nd.evaluate_HallResistance_2nd()
    System_2nd.test_chara_potential()
    plt.show()
    System_2nd.test_transmission_derivatives_error()
    plt.show()
    System_2nd.test_conductance_2nd()
    System_2nd.test_idrop_2nd()
    
    return System_2nd

def Analysis_1st(Hamiltonian, Elist, dE_avg = 0.2, det_threshold=1e-9, plotting=True):
    # First order resistance
    print("Calculating first order resistance as a function of Fermi energy")
    HallData_1st = HallTransport.HallResistanceData(Hamiltonian, Elist, det_threshold)
    HallData_1st.evaluate_Hallresistance() # Main computation
    HallData_1st.evaluate_Hallresistance_avg(dE_avg)
    if plotting is True:
        fig, ax = HallTransport.HallResistanceData.PlotHallConductance(HallData_1st.Elist_nonsingular, HallData_1st.Hall_resistance, HallData_1st.gapindex)
        fig, ax = HallTransport.HallResistanceData.PlotHallConductance(HallData_1st.Elist_nonsingular, HallData_1st.Hall_resistance_avg, HallData_1st.gapindex)
    return HallData_1st

def Analysis_2nd(Hamiltonian, Elist, dE_derivative = 1e-8, dE_avg = 0.2, gapindex = None, plotting=True, ab_separation=False):
    # Second order resistance
    print("Calculating second order resistance as a function of Fermi energy")
    HallData_2nd = HallTransport.HallResistanceData_2nd(Hamiltonian, Elist, dE=dE_derivative,  ab_separation=ab_separation)
    Elist = HallData_2nd.Elist
    if HallData_2nd.gapindex is None: HallData_2nd.gapindex = gapindex

    HallData_2nd.evaluate_Hallresistance_2nd() # Main computation
    HallData_2nd.evaluate_Hallresistance_2nd_avg(dE_avg)
    if plotting is True:
        # Plotting Hall resistance
        print("Plotting Hall resistance")
        fig, ax = HallTransport.HallResistanceData_2nd.PlotHallConductance_2nd(HallData_2nd.Elist, HallData_2nd.Hall_resistance_2nd, HallData_2nd.gapindex)
        plt.show(fig)

        # Plotting Hall resistance averaged
        print("Plotting Hall resistance averaged")
        HallData_2nd.PlotHallConductance_2nd(HallData_2nd.Elist, HallData_2nd.Hall_resistance_2nd_avg, HallData_2nd.gapindex)
        # HallData_2nd.Plot_voltages(HallData_2nd.Elist, HallData_2nd.voltages_2nd_avg)
        plt.show(fig)

        # Plotting error
        print("Plotting error")
        fig, ax1, ax2 = HallData_2nd.Plot_relative_error_with_norms(\
                    HallData_2nd.Elist, HallData_2nd.transmission_dE_deV_error,\
                    HallData_2nd.transmission_dE_norm, HallData_2nd.transmission_deV_norm)
        ax2.set_ylim(0, 0.1)
        ax1.set_ylim(bottom=0)

        if ab_separation:
            fig, ax = HallData_2nd.PlotHallConductance_2nd_abseparation(HallData_2nd.Elist, HallData_2nd.Hall_resistance_2nd_avg, HallData_2nd.Hall_resistance_a_2nd_avg, HallData_2nd.Hall_resistance_b_2nd_avg, HallData_2nd.gapindex)

    
    return HallData_2nd