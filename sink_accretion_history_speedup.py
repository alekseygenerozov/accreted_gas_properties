#!/usr/bin/env python3

import glob
import os
import time

import h5py
import numpy as np
import pandas as pd
import pytreegrav as pg

GAS_DATA_DTYPE = [
    ("sink_id", np.int64),
    ("M_tot", np.float64),
    ("x_peak_1", np.float64),
    ("x_peak_2", np.float64),
    ("x_peak_3", np.float64),
    ("max_dist_peak", np.float64),
    ("x_cm_1", np.float64),
    ("x_cm_2", np.float64),
    ("x_cm_3", np.float64),
    ("v_cm_1", np.float64),
    ("v_cm_2", np.float64),
    ("v_cm_3", np.float64),
    ("L_vec_1", np.float64),
    ("L_vec_2", np.float64),
    ("L_vec_3", np.float64),
    ("R_eff", np.float64),
    ("R_p_1", np.float64),
    ("R_p_2", np.float64),
    ("R_p_3", np.float64),
    ("T", np.float64),
    ("T_std", np.float64),
    ("T_16", np.float64),
    ("T_med", np.float64),
    ("T_84", np.float64),
    ("B", np.float64),
    ("B_std", np.float64),
    ("B_16", np.float64),
    ("B_med", np.float64),
    ("B_84", np.float64),
    ("Ne", np.float64),
    ("Ne_std", np.float64),
    ("Ne_16", np.float64),
    ("Ne_med", np.float64),
    ("Ne_84", np.float64),
    ("rho", np.float64),
    ("rho_std", np.float64),
    ("rho_16", np.float64),
    ("rho_med", np.float64),
    ("rho_84", np.float64),
    ("sig_3D", np.float64),
    ("E_grav", np.float64),
    ("E_kin", np.float64),
    ("E_mag", np.float64),
    ("E_int", np.float64),
    ("N_inc", np.int64),
    ("N_fb", np.int64),
    ("PE_global", np.float64),
    ("num_stars", np.int64),
]


def weighted_percentile(data, weights, percentile):
    """Calculates the weighted percentile of a 1D array."""
    if weights is None:
        return np.percentile(data, percentile)

    # Sort data and weights synchronously
    ix = np.argsort(data)
    data, weights = data[ix], weights[ix]

    # Compute the cumulative distribution of weights -- following numpy percentile
    cum_weights = np.cumsum(weights)
    cum_weights /= np.sum(weights)
    values_count = len(data)
    if np.any(cum_weights[0, ...] == 0):
        cum_weights[cum_weights == 0] = -1

    def find_cdf_1d(arr, cdf):
        indices = np.searchsorted(cdf, percentile / 100, side="left")
        # We might have reached the maximum with i = len(arr), e.g. for
        # percentile = 1, and need to cut it to len(arr) - 1.
        indices = np.minimum(indices, values_count - 1)
        result = np.take(arr, indices, axis=0)
        return result

    # Interpolate to find the data value at the desired percentile
    return find_cdf_1d(data, cum_weights)


class SinkAccretionHistory:
    """
    Class for getting particle IDs and accretion times for all gas particles to
    be accreted onto a sink particle for all sink particles formed in a
    STARFORGE simulation.
    Needs bhswallow_%d.txt, bhformation_%d.txt files in blackhole_details.
    Adapted from A. Kaalva.
    """

    # Initialize with path to blackhole_details directory (bhdir).
    def __init__(self, bhdir, fname_gas=None, fname_sink=None):
        self.bhdir = bhdir
        self.accretion_dict = self.accretion_history_to_dict(
            fname_gas=fname_gas, fname_sink=fname_sink
        )

    def get_accretion_history(self, fname=None, save_txt=True, verbose=True):

        # List of bhswallow_%d.txt filenames:
        bhswallow_files = glob.glob(self.bhdir + "bhswallow*.txt")

        # Read all lines from all bhswallow files:
        bhswallow_total = []
        for i in range(len(bhswallow_files)):
            if verbose:
                print("Opening bhswallow file {0:d}...".format(i), flush=True)
            with open(bhswallow_files[i]) as f:
                data_list = f.read().splitlines()
                for j in range(len(data_list)):
                    # Ignore lines beginning with # symbol.
                    if data_list[j][0:1] == "#":
                        continue
                    new_data = data_list[j].split()
                    bhswallow_total.append(new_data)

        # Lists of sink IDs, gas IDs, and accretion times from bhswallow_data:
        sink_ID_list = []
        gas_ID_list = []
        time_list = []
        for i in range(len(bhswallow_total)):
            time_list.append(float(bhswallow_total[i][0]))
            sink_ID_list.append(int(bhswallow_total[i][1]))
            gas_ID_list.append(int(bhswallow_total[i][6]))
        sink_IDs = np.asarray(sink_ID_list, dtype=int)
        gas_IDs = np.asarray(gas_ID_list, dtype=int)
        times = np.asarray(time_list, dtype=float)

        # Sort by sink ID to group accretion history for each sink particle.
        if verbose:
            print("Sorting data by sink ID...", flush=True)
        idx_sink_sort = np.argsort(sink_IDs)
        sink_IDs_grouped = sink_IDs[idx_sink_sort]
        gas_IDs_grouped = gas_IDs[idx_sink_sort]
        times_grouped = times[idx_sink_sort]

        # Get first occurrence of each unique sink ID in grouped list:
        unique_sinks, first_occurrence = np.unique(sink_IDs_grouped, return_index=True)

        # Within each sink particle group, sort accreted gas particles by accretion time.
        if verbose:
            print("Sorting data by accretion time...", flush=True)
        idx_times_sort = []
        for i in range(len(first_occurrence) - 1):
            imin = first_occurrence[i]
            imax = first_occurrence[i + 1]
            # Sort times[imin:imax]
            idx_t = np.argsort(times_grouped[imin:imax]) + imin
            idx_times_sort.extend(idx_t.tolist())
        idx_t = np.argsort(times_grouped[imax:]) + imax
        idx_times_sort.extend(idx_t.tolist())

        sink_IDs_sort = sink_IDs_grouped[idx_times_sort]
        gas_IDs_sort = gas_IDs_grouped[idx_times_sort]
        times_sort = times_grouped[idx_times_sort]

        # Stack into (N_gas_accreted, 3) array to save to text file.
        accretion_data = np.vstack((sink_IDs_sort, gas_IDs_sort, times_sort)).T

        # (Optionally) save as text file.
        if save_txt:
            if verbose:
                print("Saving as text file...", flush=True)
            # If no filename, save in blackhole_details directory.
            if fname is None:
                fname = os.path.join(self.bhdir, "sink_accretion_data.txt")
            fmt = "%d", "%d", "%.8f"
            np.savetxt(fname, accretion_data, fmt=fmt)

        return accretion_data

    # Get sink particle properties at formation time.
    def get_sink_formation_details(self, fname=None, save_txt=False):

        # List of bhformation_%d.txt filenames:
        bhformation_files = glob.glob(self.bhdir + "bhformation*.txt")

        # Read all lines from all bhswallow files:
        bhformation_total = []
        for i in range(len(bhformation_files)):
            with open(bhformation_files[i]) as f:
                data_list = f.read().splitlines()
                for j in range(len(data_list)):
                    # Skip lines beginning with # symbol.
                    if data_list[j][0:1] == "#":
                        continue
                    new_data = data_list[j].split()
                    bhformation_total.append(new_data)

        # Lists of sink IDs, formation time, mass, position, velocity
        sink_ID_list = []
        m_list, t_list = [], []
        x_list, y_list, z_list = [], [], []
        u_list, v_list, w_list = [], [], []

        for i in range(len(bhformation_total)):
            sink_ID_list.append(int(bhformation_total[i][1]))
            t_list.append(float(bhformation_total[i][0]))
            m_list.append(float(bhformation_total[i][2]))
            x_list.append(float(bhformation_total[i][3]))
            y_list.append(float(bhformation_total[i][4]))
            z_list.append(float(bhformation_total[i][5]))
            u_list.append(float(bhformation_total[i][6]))
            v_list.append(float(bhformation_total[i][7]))
            w_list.append(float(bhformation_total[i][8]))

        sink_IDs = np.asarray(sink_ID_list, dtype=int)
        t = np.asarray(t_list, dtype=float)
        m = np.asarray(m_list, dtype=float)
        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        z = np.asarray(z_list, dtype=float)
        u = np.asarray(u_list, dtype=float)
        v = np.asarray(v_list, dtype=float)
        w = np.asarray(w_list, dtype=float)

        sink_formation_data = np.vstack((t, m, x, y, z, u, v, w)).T

        # (Optionally) save as text file.
        if save_txt:
            # If no filename, save in blackhole_details directory.
            if fname is None:
                fname = os.path.join(self.bhdir, "sink_formation_data.txt")
            fmt = "%d", "%.8g", "%.8g", "%.8g", "%.8g", "%.8g", "%.8g", "%.8g", "%.8g"
            np.savetxt(fname, np.vstack((sink_IDs, t, m, x, y, z, u, v, w)).T, fmt=fmt)

        return sink_IDs, sink_formation_data

    # Parse accretion_data, sink_formation_data to return dict containing
    # gas IDs, accretion times, formation properties for each sink particle.
    # dict keys: sink IDs (int_arr)
    # dict values:
    #   - accretion_dict[sink_ID][0] = [accreted_gas_ids (int_arr), accretion_times (float_arr),
    #                                   (non-feedback) ids, (non-feedback) counts,
    #                                   (feedback) ids, (feedback) counts]
    #   - accretion_dict[sink_ID][1] = sink [t, m, x, y, z, vx, vy, vz] at formation.
    def accretion_history_to_dict(self, fname_gas=None, fname_sink=None):

        # Read accretion_data from stored .txt file.
        if fname_gas is not None:
            sink_IDs = np.loadtxt(fname_gas, dtype=int, usecols=0)
            gas_IDs = np.loadtxt(fname_gas, dtype=int, usecols=1)
            acc_times = np.loadtxt(fname_gas, dtype=float, usecols=2)
        else:
            accretion_data = self.get_accretion_history()
            sink_IDs = accretion_data[:, 0]
            gas_IDs = accretion_data[:, 1]
            acc_times = accretion_data[:, 2]

        unique_sinks, first_occurrence = np.unique(sink_IDs, return_index=True)

        # Split gas IDs, accretion times into list of subarrays per sink particle.
        gas_IDs_split = np.split(gas_IDs, first_occurrence[1:])
        acc_times_split = np.split(acc_times, first_occurrence[1:])

        accretion_dict = dict.fromkeys(unique_sinks.astype(np.int64))
        for i in range(len(unique_sinks)):

            sink_accretion_dict = {
                "all_gas_ids": [],
                "all_acc_times": [],
                "non_fb_ids": [],
                "non_fb_counts": [],
                "fb_ids": [],
                "fb_counts": [],
                "formation_details": [],
            }

            # All accreted gas IDs for this sink particle.
            gas_ids_all, acc_times_all = (
                gas_IDs_split[i].astype(np.int64),
                acc_times_split[i],
            )

            # Find unique gas IDs among list accreted gas IDs. Use to classify gas particles as
            # "feedback" (multiple occurrences of particle ID) vs. "non-feedback" (appears only
            # once in list of accreted gas IDs).
            u, c = np.unique(gas_ids_all, return_counts=True)
            mask = c == 1
            u_n, c_n = (
                u[mask],
                c[mask],
            )  # Particle IDs, counts of non-feedback particles.
            u_f, c_f = u[~mask], c[~mask]  # Particle IDs, counts of feedback particles.

            # Use field 'u_n' (particle IDs of non-feedback gas) to analyze accreted gas properties.
            # u_n = accretion_dict[sink_ID][0][2]
            # u_f = accretion_dict[sink_ID][0][4]
            # accretion_dict[unique_sinks[i]] = [gas_ids_all, acc_times_all, u_n, c_n, u_f, c_f]
            sink_accretion_dict["all_gas_ids"] = gas_ids_all
            sink_accretion_dict["all_acc_times"] = acc_times_all
            sink_accretion_dict["non_fb_ids"] = u_n
            sink_accretion_dict["non_fb_counts"] = c_n
            sink_accretion_dict["fb_ids"] = u_f
            sink_accretion_dict["fb_counts"] = c_f
            accretion_dict[unique_sinks[i]] = sink_accretion_dict
        accretion_dict["sink_IDs"] = unique_sinks.astype(np.int64)

        # Add sink particle properties at formation time to dict.
        if fname_sink is not None:
            s_IDs = np.loadtxt(fname_sink, dtype=int, usecols=0)
            s_data = np.loadtxt(
                fname_sink, dtype=float, usecols=(1, 2, 3, 4, 5, 6, 7, 8)
            )
        else:
            s_IDs, s_data = self.get_sink_formation_details()

        for i, sink_ID in enumerate(s_IDs):
            if sink_ID in accretion_dict:
                sink_accretion_dict = accretion_dict[sink_ID]
                sink_accretion_dict["formation_details"] = s_data[i]
                accretion_dict[sink_ID] = sink_accretion_dict

        return accretion_dict


# TO-DO: want to sort sink particles in snapshot into singles, binaries, higher-order
# systems - add "system type" to above accretion_dict?


class Cloud:
    """
    Class for calculating bulk cloud properties.
    Parameters:
        - M0: initial cloud mass [code_mass].
        - R0: initial cloud radius [code_length].
        - alpha0: initial cloud turbulent virial parameter.
        - G_code: gravitational constant in code units
        (default: 4300.17 in [pc * Msun^-1 * (m/s)^2]).
    """

    def __init__(self, M, R, alpha, G_code=4300.71, verbose=False):

        # Initial cloud mass, radius, and turbulent virial parameter.
        # G_code: gravitational constant in code units [default: 4300.71].
        self.M = M
        self.R = R
        self.L = (4.0 * np.pi * self.R**3 / 3.0) ** (1.0 / 3.0)
        self.alpha = alpha
        self.G_code = G_code

        self.rho = self.get_initial_density(verbose=verbose)
        self.Sigma = self.get_initial_surface_density(verbose=verbose)
        self.vrms = self.get_initial_sigma_3D(verbose=verbose)
        self.t_cross = self.get_initial_t_cross(verbose=verbose)
        self.t_ff = self.get_initial_t_ff(verbose=verbose)

    # ----------------------------- FUNCTIONS ---------------------------------

    # FROM INITIAL CLOUD PARAMETERS: surface density, R, vrms, Mach number.
    def get_initial_density(self, verbose=False):
        """
        Calculate the initial cloud density [code_mass/code_length**3].
        """
        rho = (3.0 * self.M) / (4.0 * np.pi * self.R**3)
        if verbose:
            print("Density: {0:.2f} Msun pc^-2".format(rho))
        return rho

    def get_initial_surface_density(self, verbose=False):
        """
        Calculate the initial cloud surface density [code_mass/code_length**2].
        """
        Sigma = self.M / (np.pi * self.R**2)
        if verbose:
            print("Surface density: {0:.2f} Msun pc^-2".format(Sigma))
        return Sigma

    # Initial 3D rms velocity.
    def get_initial_sigma_3D(self, verbose=False):
        """
        Calculate the initial 3D rms velocity [code_velocity].
        """
        sig_3D = np.sqrt((3.0 * self.alpha * self.G_code * self.M) / (5.0 * self.R))
        if verbose:
            print("sigma_3D = {0:.3f} m s^-1".format(sig_3D))
        return sig_3D

    # Initial cloud trubulent crossing time.
    def get_initial_t_cross(self, verbose=False):
        """
        Calculate the initial turbulent crossing time as R/v_rms [code_time].
        """
        sig_3D = self.get_initial_sigma_3D()
        t_cross = self.R / sig_3D
        if verbose:
            print("t_cross = {0:.3g} [code]".format(t_cross))
        return t_cross

    # Initial cloud gravitational free-fall time.
    def get_initial_t_ff(self, verbose=False):
        """
        Calculate the initial freefall time [code_time].
        """
        rho = (3.0 * self.M) / (4.0 * np.pi * self.R**3)
        t_ff = np.sqrt((3.0 * np.pi) / (32.0 * self.G_code * rho))
        if verbose:
            print("t_ff = {0:.3g} [code]".format(t_ff))
        return t_ff


# Speedup by pre-computing idx_g and pasing m, v, etc. as
# arguments to functions for calculating gas properties.
class SnapshotGasProperties:
    """
    Read gas particle data from HDF5 snapshot.
    """

    def __init__(self, fname, cloud, B_unit=1e4):

        # Physical constants.
        self.PROTONMASS_CGS = 1.6726e-24
        self.ELECTRONMASS_CGS = 9.10953e-28
        self.BOLTZMANN_CGS = 1.38066e-16
        self.HYDROGEN_MASSFRAC = 0.76
        self.ELECTRONCHARGE_CGS = 4.8032e-10
        self.C_LIGHT_CGS = 2.9979e10

        # Initial cloud parameters.
        self.fname = fname
        self.snapdir = self.get_snapdir()
        self.Cloud = cloud
        self.M0 = cloud.M  # Initial cloud mass, radius.
        self.R0 = cloud.R
        self.L0 = cloud.L  # Volume-equivalent length.
        self.alpha0 = cloud.alpha  # Initial virial parameter.

        # OLD: Resolution limit for masking feedback cells.
        # self.res_limit = res_limit (1e-3)
        # NEW: mask feedback cells based on frequency of appearance in list
        # of all accreted gas IDs (1 = non-feedback; >1 = feedback).
        # PROBLEM - some feedback particles not captured by uniqueness
        # check in acc_dict[sink_ID]['non_fb_ids'] - need an additional
        # uniqueness check in list of matching particle IDs in snapshot.

        # Open HDF5 file.
        with h5py.File(fname, "r") as ff:
            header = ff["Header"]
            p0 = ff["PartType0"]  # Particle type 0 (gas).

            # Header attributes.
            self.box_size = header.attrs["BoxSize"]
            self.num_p0 = header.attrs["NumPart_Total"][0]
            self.t = header.attrs["Time"]

            # Unit conversions to cgs; note typo in header for G_code.
            self.G_code = header.attrs["Gravitational_Constant_In_Code_Inits"]
            if "Internal_UnitB_In_Gauss" in header.attrs:
                self.B_code = header.attrs[
                    "Internal_UnitB_In_Gauss"
                ]  # sqrt(4*pi*unit_pressure_cgs)
            else:
                self.B_code = 2.916731267922059e-09
            self.l_unit = header.attrs["UnitLength_In_CGS"]
            self.m_unit = header.attrs["UnitMass_In_CGS"]
            self.v_unit = header.attrs["UnitVelocity_In_CGS"]
            self.B_unit = B_unit  # Magnetic field unit in Gauss (default: 1e4).
            self.t_unit = self.l_unit / self.v_unit
            self.t_unit_myr = self.t_unit / (3600.0 * 24.0 * 365.0 * 1e6)
            self.rho_unit = self.m_unit / self.l_unit**3
            self.nH_unit = self.rho_unit / self.PROTONMASS_CGS
            self.P_unit = self.m_unit / self.l_unit / self.t_unit**2
            self.spec_L_unit = self.l_unit * self.v_unit  # Specific angular momentum.
            self.L_unit = self.spec_L_unit * self.m_unit  # Angular momentum.
            self.E_unit = self.l_unit**2 / self.t_unit**2  # Energy per mass.
            # Convert internal energy to temperature units.
            self.u_to_temp_units = (
                self.PROTONMASS_CGS / self.BOLTZMANN_CGS
            ) * self.E_unit

            # Other useful conversion factors.
            self.cm_to_AU = 6.6845871226706e-14
            self.cm_to_pc = 3.2407792896664e-19

            # PartType0 data.
            self.p0_ids = p0["ParticleIDs"][()]  # Particle IDs.
            self.p0_m = p0["Masses"][()].astype(float)  # Masses.
            self.p0_rho = p0["Density"][()].astype(float)  # Density.
            self.p0_hsml = p0["SmoothingLength"][()].astype(
                float
            )  # Particle smoothing length.
            self.p0_E_int = p0["InternalEnergy"][()].astype(float)  # Internal energy.
            self.p0_P = p0["Pressure"][()].astype(float)  # Pressure.
            self.p0_cs = p0["SoundSpeed"][()].astype(float)  # Sound speed.
            self.p0_x = p0["Coordinates"][()][:, 0].astype(float)  # Coordinates.
            self.p0_y = p0["Coordinates"][()][:, 1].astype(float)
            self.p0_z = p0["Coordinates"][()][:, 2].astype(float)
            self.p0_u = p0["Velocities"][()][:, 0].astype(float)  # Velocities.
            self.p0_v = p0["Velocities"][()][:, 1].astype(float)
            self.p0_w = p0["Velocities"][()][:, 2].astype(float)
            self.p0_Ne = p0["ElectronAbundance"][()]  # Electron abundance.
            if "MagneticField" in p0.keys():  # Magnetic field.
                self.p0_Bx = p0["MagneticField"][()][:, 0].astype(float)
                self.p0_By = p0["MagneticField"][()][:, 1].astype(float)
                self.p0_Bz = p0["MagneticField"][()][:, 2].astype(float)
                self.p0_B_mag = np.sqrt(self.p0_Bx**2 + self.p0_By**2 + self.p0_Bz**2)
            else:
                self.p0_Bx = np.zeros(len(self.p0_ids))
                self.p0_By = np.zeros(len(self.p0_ids))
                self.p0_Bz = np.zeros(len(self.p0_ids))
                self.p0_B_mag = np.zeros(len(self.p0_ids))

            # Hydrogen number density and total metallicity.
            self.p0_n_H = (1.0 / self.PROTONMASS_CGS) * np.multiply(
                self.p0_rho * self.rho_unit,
                1.0 - p0["Metallicity"][()][:, 0].astype(float),
            )
            self.p0_total_metallicity = p0["Metallicity"][()][:, 0].astype(float)
            # Calculate mean molecular weight.
            self.p0_mean_molecular_weight = self.get_mean_molecular_weight(self.p0_ids)

            # Neutral hydrogen abundance, molecular mass fraction.
            self.p0_neutral_H_abundance = p0["NeutralHydrogenAbundance"][()].astype(
                float
            )
            self.p0_molecular_mass_frac = p0["MolecularMassFraction"][()].astype(float)

            # Calculate gas adiabatic index and temperature.
            fH, f, xe = self.HYDROGEN_MASSFRAC, self.p0_molecular_mass_frac, self.p0_Ne
            f_mono, f_di = fH * (xe + 1.0 - f) + (1.0 - fH) / 4.0, fH * f / 2.0
            gamma_mono, gamma_di = 5.0 / 3.0, 7.0 / 5.0
            gamma = 1.0 + (f_mono + f_di) / (
                f_mono / (gamma_mono - 1.0) + f_di / (gamma_di - 1.0)
            )
            self.p0_temperature = (
                (gamma - 1.0)
                * self.p0_mean_molecular_weight
                * self.u_to_temp_units
                * self.p0_E_int
            )
            # Dust temperature.
            if "Dust_Temperature" in p0.keys():
                self.p0_dust_temp = p0["Dust_Temperature"][()]

            # Simulation timestep.
            if "TimeStep" in p0.keys():
                self.p0_timestep = p0["TimeStep"][()]

            # For convenience, stack coordinates/velocities/B_field in a (n_gas, 3) array.
            self.p0_coord = np.vstack((self.p0_x, self.p0_y, self.p0_z)).T
            self.p0_vel = np.vstack((self.p0_u, self.p0_v, self.p0_w)).T
            self.p0_mag = np.vstack((self.p0_Bx, self.p0_By, self.p0_Bz)).T
            # Create a global mapping of Particle ID -> Snapshot Index ONCE in __init__
            # Find the maximum ID value to size our lookup array
            max_id = int(np.max(self.p0_ids))
            # Initialize with -1 (meaning "not found in this snapshot")
            self.id_to_index = np.full(max_id + 1, -1, dtype=np.int64)
            # Assign snapshot indices to their respective particle IDs
            self.id_to_index[self.p0_ids] = np.arange(len(self.p0_ids))
            # 3. Compute unique IDs and counts using NumPy
            unique_ids, counts = np.unique(self.p0_ids, return_counts=True)
            # 4. Map the counts directly to their ID slots
            self.id_counts = np.zeros(max_id + 1, dtype=np.int64)
            self.id_counts[unique_ids] = counts
            self.star_soft = 8.73e-5

            self.stars_x = np.zeros(0)
            self.stars_y = np.zeros(0)
            self.stars_z = np.zeros(0)
            self.stars_m = np.zeros(0)
            self.stars_coord = np.vstack((self.stars_x, self.stars_y, self.stars_z)).T
            self.stars_hsml = np.ones(0) * self.star_soft
            if "PartType5" in ff:
                stars = ff["PartType5"]  # Particle type 5 (sink/star).
                self.stars_x = stars["Coordinates"][()][:, 0].astype(
                    float
                )  # star Coordinates.

                self.stars_y = stars["Coordinates"][()][:, 1].astype(float)
                self.stars_z = stars["Coordinates"][()][:, 2].astype(float)
                self.stars_coord = np.vstack(
                    (self.stars_x, self.stars_y, self.stars_z)
                ).T
                self.stars_m = stars["Masses"][()].astype(float)  # star Masses.
                # self.stars_hsml = stars["SinkRadius"][()].astype(
                #     float
                # )  # star Smoothing length.
                self.stars_hsml = np.ones(len(self.stars_m)) * self.star_soft

                pos_all = np.vstack((self.p0_coord, self.stars_coord))
                mass_all = np.concatenate((self.p0_m, self.stars_m))
                soft_all = np.concatenate((self.p0_hsml, self.stars_hsml))
            else:
                pos_all = self.p0_coord
                mass_all = self.p0_m
                soft_all = self.p0_hsml

            start = time.time()
            print(
                "Computing tree",
            )
            self.tree = pg.ConstructTree(pos_all, mass_all, softening=soft_all)
            print("Tree done", f"{time.time() - start:.2g}")

    # Try to get snapshot number from filename.
    def get_i(self):
        return int(self.fname.split("snapshot_")[1].split(".hdf5")[0])

    # Try to get snapshot datadir from filename.
    def get_snapdir(self):
        return self.fname.split("snapshot_")[0]

    # Calculate gas mean molecular weight.
    def get_mean_molecular_weight(self, gas_ids):
        T_eff_atomic = 1.23 * (5.0 / 3.0 - 1.0) * self.u_to_temp_units * self.p0_E_int
        nH_cgs = self.p0_rho * self.nH_unit
        T_transition = self._DMIN(8000.0, nH_cgs)
        f_mol = 1.0 / (1.0 + T_eff_atomic**2 / T_transition**2)
        return 4.0 / (
            1.0 + (3.0 + 4.0 * self.p0_Ne - 2.0 * f_mol) * self.HYDROGEN_MASSFRAC
        )

    # Return only those particle IDs in gas_ids which occur only once in
    # list of all matching snapshot particle IDs.
    def get_unique_ids(self, gas_ids):
        valid_ids = gas_ids[gas_ids <= (len(self.id_to_index) - 1)]
        gas_ids_index = self.id_to_index[valid_ids]
        ##Filter out particles that don't exist in this snapshot (value is -1)
        valid_ids = valid_ids[gas_ids_index >= 0]
        gas_ids_index = gas_ids_index[gas_ids_index >= 0]
        ##Getting rid of feedback cells
        gas_ids_index_counts = self.id_counts[valid_ids]
        gas_ids_index_nf = gas_ids_index[gas_ids_index_counts == 1]

        return gas_ids_index_nf, len(gas_ids_index) - len(gas_ids_index_nf)

    # Get mask based on gas_ids (include only unique IDs).
    def get_mask(self, gas_ids, verbose=False):
        num_excluded, unique_ids = self.get_unique_ids(gas_ids)
        mask = np.isin(self.p0_ids, unique_ids)
        return mask

    # Check that mask based on gas_ids is non-empty.
    def check_gas_ids(self, gas_ids, verbose=False):
        mask_all = self.get_mask(gas_ids)
        if np.sum(mask_all) > 0:
            return True
        else:
            if verbose:
                print("No gas_ids above resolution limit.")
            return False

    # Get indices of selected gas particles.
    def get_idx(self, gas_ids):
        idx_g, num_excluded = self.get_unique_ids(gas_ids)

        return idx_g, num_excluded

    def get_density_peak(self, pos, rho):
        peak_idx = np.argmax(rho)
        return pos[peak_idx], peak_idx

    # Get center of mass for selected gas particles.
    def get_center_of_mass(self, m, pos, vel):
        M = np.sum(m)
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        u, v, w = vel[:, 0], vel[:, 1], vel[:, 2]

        x_cm = np.sum(np.multiply(m, x)) / M
        u_cm = np.sum(np.multiply(m, u)) / M
        y_cm = np.sum(np.multiply(m, y)) / M
        v_cm = np.sum(np.multiply(m, v)) / M
        z_cm = np.sum(np.multiply(m, z)) / M
        w_cm = np.sum(np.multiply(m, w)) / M

        cm_x = np.asarray([x_cm, y_cm, z_cm])
        cm_v = np.asarray([u_cm, v_cm, w_cm])

        return M, cm_x, cm_v

    # Get gas kinematics relative to x, v vectors.
    def get_relative_kinematics(self, m, pos, vel, point_x, point_v):
        x0, y0, z0 = point_x[0], point_x[1], point_x[2]
        u0, v0, w0 = point_v[0], point_v[1], point_v[2]

        x1, y1, z1 = pos[:, 0], pos[:, 1], pos[:, 2]
        u1, v1, w1 = vel[:, 0], vel[:, 1], vel[:, 2]

        x, y, z = x1 - x0, y1 - y0, z1 - z0
        u, v, w = u1 - u0, v1 - v0, w1 - w0

        if np.isscalar(m):
            return m, np.asarray([x, y, z]), np.asarray([u, v, w])
        else:
            return m, np.vstack((x, y, z)).T, np.vstack((u, v, w)).T

    # Get total mass of remaining gas.
    def get_total_mass(self, m):
        return np.sum(m)

    # Get effective radius of selected gas particles.
    def get_effective_radius(self, m, rho):
        vol = np.sum(m / rho)
        r = ((3.0 * vol) / (4.0 * np.pi)) ** (1.0 / 3.0)
        return r

    # Get velocity dispersion of selected gas particles.
    def get_velocity_dispersion(self, m, vel):
        u, v, w = vel[:, 0], vel[:, 1], vel[:, 2]
        sigma_3D = np.sqrt(
            (
                self.weight_std(u, m) ** 2.0
                + self.weight_std(v, m) ** 2.0
                + self.weight_std(w, m) ** 2.0
            )
        )
        return sigma_3D

    # Get angular momentum (with respect to center of mass) of selected gas particles.
    def get_angular_momentum(self, m, pos, vel):
        m_cm, x_cm, v_cm = self.get_center_of_mass(m, pos, vel)
        m_g, x_g, v_g = self.get_relative_kinematics(m, pos, vel, x_cm, v_cm)
        ang_mom_vec = np.sum(
            np.cross(x_g, np.multiply(np.reshape(m_g, (len(m_g), 1)), v_g)), axis=0
        )
        ang_mom_mag = np.linalg.norm(ang_mom_vec)
        ang_mom_unit_vec = ang_mom_vec / ang_mom_mag
        return ang_mom_unit_vec, ang_mom_mag

    # Get specific angular momentum (with respect to center of mass) of selected gas particles.
    def get_specific_angular_momentum(self, m, pos, vel):
        if len(m) == 1:
            return np.asarray([0.0, 0.0, 0.0]), 0.0
        m_cm, x_cm, v_cm = self.get_center_of_mass(m, pos, vel)
        m_g, x_g, v_g = self.get_relative_kinematics(m, pos, vel, x_cm, v_cm)
        # ang_mom_vec = np.sum(np.cross(x_g, v_g), axis=0)
        ang_mom_vec = (
            np.sum(
                np.cross(x_g, np.multiply(np.reshape(m_g, (len(m_g), 1)), v_g)), axis=0
            )
            / m_cm
        )
        ang_mom_mag = np.linalg.norm(ang_mom_vec)
        ang_mom_unit_vec = ang_mom_vec / ang_mom_mag
        return ang_mom_unit_vec, ang_mom_mag

    # Get gravitational potential energy (need pytreegrav module).
    def get_potential_energy(self, m, h, pos):
        E_pot = 0.5 * np.sum(m * pg.Potential(pos, m, h, G=self.G_code))
        return E_pot

    # Get kinetic energy [code units].
    def get_kinetic_energy(self, m, vel):
        dv = vel - np.average(vel, weights=m, axis=0)
        v_sqr = np.sum(dv**2, axis=1)
        E_kin = 0.5 * np.sum(m * v_sqr)
        return E_kin

    # Get magnetic energy [code units].
    def get_magnetic_energy(self, m, rho, B_mag):
        vol = (m / rho) * self.l_unit**3
        E_mag = (
            (1.0 / (8.0 * np.pi))
            * np.sum(B_mag * B_mag * vol)
            / (self.E_unit * self.m_unit)
        )
        return E_mag

    # Get internal energy [code units].
    def get_internal_energy(self, m, int_en):
        E_int = np.sum(m * int_en)
        return E_int

    def get_complete_stats(self, X, w=None):
        try:
            assert np.isclose(
                np.average(X, weights=w), self.weight_avg(X, w), rtol=1e-3
            )
            ##NOTE: USE DDOF=1 TO GET AN UNBIASED ESTIMATOR OF THE STANDARD DEVIATION(!)
            assert np.isclose(
                np.sqrt(np.cov(X, aweights=w, ddof=0)), self.weight_std(X, w), rtol=1e-3
            )
        except AssertionError:
            breakpoint()
        return (
            np.average(X, weights=w),
            np.sqrt(np.cov(X, aweights=w, ddof=0)),
            weighted_percentile(X, w, 16),
            weighted_percentile(X, w, 50),
            weighted_percentile(X, w, 84),
        )

    # Get aspect ratio of selected gas particles (prinicpal component analysis).
    def get_aspect_ratio(self, m, pos):
        dx = pos - np.mean(pos, axis=0)
        R = np.linalg.norm(dx, axis=1)
        ndim = pos.shape[1]
        I = np.eye(ndim) * np.mean(R**2)
        for i in range(ndim):
            for j in range(ndim):
                I[i, j] -= np.mean(dx[:, i] * dx[:, j])
        try:
            w, v = np.linalg.eig(I)
        except np.linalg.LinAlgError:
            breakpoint()
        R_p = np.sqrt(w)
        # return np.max(R_p)/np.min(R_p), np.sum(R_p/R_p.max()), R_p
        return R_p  # array of principle components.

    # Get gas properties of selected gas particles.
    # gas_ids: pass accretion_dict[sink_ID]['non_fb_ids'] (non-feedback particle IDs).
    # new particle uniqueness ID check added to get_idx in get_X methods.
    def get_gas_data(
        self, idx_g, num_feedback, num_feedback_new, skip_potential=False, verbose=True
    ):
        fields = [f for f in GAS_DATA_DTYPE if f[0] != "sink_id"]
        data = np.zeros(1, dtype=fields)[0]
        # data = np.zeros(
        #     24
        # )  # Exclude M_fb (not relevant new method for identifying feedback).

        if verbose:
            print("Getting number of gas particles...", flush=True)
            num_particles = len(idx_g)
            print("Analyzing {0:d} particles.".format(num_particles), flush=True)

        # Speedup: get idx_g once, get m, v, etc. once, and pass as arguments
        # to get_X functions.
        m, h, rho = self.p0_m[idx_g], self.p0_hsml[idx_g], self.p0_rho[idx_g]
        x, y, z = self.p0_x[idx_g], self.p0_y[idx_g], self.p0_z[idx_g]
        u, v, w = self.p0_u[idx_g], self.p0_v[idx_g], self.p0_w[idx_g]
        pos, vel = np.vstack((x, y, z)).T, np.vstack((u, v, w)).T
        B_mag = self.p0_B_mag[idx_g] * self.B_unit
        T_K = self.p0_temperature[idx_g]
        int_en = self.p0_E_int[idx_g]
        elec = self.p0_Ne[idx_g]

        # Compute gas properties.
        M_tot, x_cm, v_cm = self.get_center_of_mass(m, pos, vel)
        x_peak, peak_idx = self.get_density_peak(pos, rho)
        max_dist_peak = max([np.linalg.norm(row - x_peak) for row in pos])
        PE_global = pg.PotentialTarget(
            np.atleast_2d(x_peak),
            None,
            None,
            softening_target=np.atleast_1d(self.p0_hsml[peak_idx]),
            tree=self.tree,
            theta=0.5,
            G=self.G_code,
        )
        print(PE_global)

        L_unit_vec, L_mag = self.get_specific_angular_momentum(m, pos, vel)
        L_vec = L_mag * L_unit_vec
        R_eff = self.get_effective_radius(m, rho)
        R_p = self.get_aspect_ratio(m, pos)
        T_stats = self.get_complete_stats(T_K, w=m)
        B_stats = self.get_complete_stats(B_mag, w=m)
        Ne_stats = self.get_complete_stats(elec, w=m)
        rho_stats = self.get_complete_stats(self.p0_rho[idx_g], w=m)
        sig_3D = self.get_velocity_dispersion(m, vel)
        if skip_potential:
            if verbose:
                print("Skipping potential energy calculation...", flush=True)
            E_grav = 0.0
        else:
            if verbose:
                print("Getting potential energy with pytreegrav...", flush=True)
            E_grav = self.get_potential_energy(m, h, pos)
            if verbose:
                print("Done with potential energy calculation.", flush=True)
        E_kin = self.get_kinetic_energy(m, vel)
        E_mag = self.get_magnetic_energy(m, rho, B_mag)
        E_int = self.get_internal_energy(m, int_en)
        N_inc = len(idx_g)
        ##TO DO: CHECK WITH NINA ON DOUBLE-COUNTING(!)
        N_fb = num_feedback + num_feedback_new

        data["M_tot"] = M_tot  # Total mass.
        data["x_peak_1"], data["x_peak_2"], data["x_peak_3"] = (
            x_peak  # Density peak coordinates.
        )
        data["max_dist_peak"] = max_dist_peak
        data["x_cm_1"], data["x_cm_2"], data["x_cm_3"] = (
            x_cm  # Center of mass coordinates.
        )
        data["v_cm_1"], data["v_cm_2"], data["v_cm_3"] = (
            v_cm  # Center of mass velocity.
        )
        data["L_vec_1"], data["L_vec_2"], data["L_vec_3"] = (
            L_vec  # Specific angular momentum with respect to center of mass.
        )
        data["R_eff"] = R_eff  # Effective radius.
        data["R_p_1"], data["R_p_2"], data["R_p_2"] = R_p  # Shape parameters (PCA).
        data["T"], data["T_std"], data["T_16"], data["T_med"], data["T_84"] = (
            T_stats  # temperature stats
        )
        data["B"], data["B_std"], data["B_16"], data["B_med"], data["B_84"] = (
            B_stats  #  magnetic field magnitude (may need to use volume).
        )
        data["Ne"], data["Ne_std"], data["Ne_16"], data["Ne_med"], data["Ne_84"] = (
            Ne_stats  #  number e- per H nucleon.
        )
        (
            data["rho"],
            data["rho_std"],
            data["rho_16"],
            data["rho_med"],
            data["rho_84"],
        ) = rho_stats  # Average mass density
        data["sig_3D"] = sig_3D  # Average velocity dispersion.
        data["E_grav"] = E_grav  # Gravitational potential energy.
        data["E_kin"] = E_kin  # Kinetic energy.
        data["E_mag"] = E_mag  # Magnetic energy.
        data["E_int"] = E_int  # Internal energy (temperature proxy).
        data["N_inc"] = N_inc  # Number of gas particles included in calculations.
        data["N_fb"] = (
            N_fb  # Number of gas particles excluded due to being (presumed) feedback particles.
        )
        data["PE_global"] = PE_global

        return data

    def get_filt_peak(self, idx_g, max_dist=np.inf):
        rho = self.p0_rho[idx_g]
        x, y, z = self.p0_x[idx_g], self.p0_y[idx_g], self.p0_z[idx_g]
        pos = np.vstack((x, y, z)).T
        x_peak, peak_idx = self.get_density_peak(pos, rho)

        dist = np.array([np.linalg.norm(row - x_peak) for row in pos])
        return idx_g[dist < max_dist]

    def get_filt_peak_two_array(self, idx_g, idx_g_all, max_dist=np.inf, star=False):
        ##PEAK FROM FIRST ARRAY
        rho = self.p0_rho[idx_g]
        x, y, z = self.p0_x[idx_g], self.p0_y[idx_g], self.p0_z[idx_g]
        pos = np.vstack((x, y, z)).T
        x_peak, peak_idx = self.get_density_peak(pos, rho)
        ##POS FROM SECOND ARRAY
        if star:
            x, y, z = (
                self.stars_x[idx_g_all],
                self.stars_y[idx_g_all],
                self.stars_z[idx_g_all],
            )
        else:
            x, y, z = self.p0_x[idx_g_all], self.p0_y[idx_g_all], self.p0_z[idx_g_all]
        pos = np.vstack((x, y, z)).T

        dist_sq = np.sum((pos - x_peak) ** 2.0, axis=1)
        return idx_g_all[dist_sq < max_dist**2]

    # Get gas properties for each set of gas_ids in accretion_dict.
    def get_all_gas_data(
        self,
        acc_dict,
        skip_potential=False,
        verbose=True,
        use_all_sinks=True,
        sink_imin=None,
        sink_imax=None,
        max_dist=np.inf,
    ):

        # Unique sink IDs:
        if use_all_sinks:
            sink_IDs = acc_dict["sink_IDs"]
        else:
            sink_IDs = acc_dict["sink_IDs"][sink_imin : sink_imax + 1]
        num_sinks = len(sink_IDs)

        if verbose:
            print(
                "Getting data for {0:d} sink particles in this snapshot...".format(
                    num_sinks
                ),
                flush=True,
            )

        # all_data = np.zeros((num_sinks, 25))  # Entry 0: sink ID.
        # Initialize the global structured tracking array
        all_data = np.zeros(num_sinks, dtype=GAS_DATA_DTYPE)
        idx_g_all, num_feedback_new = self.get_idx(
            self.p0_ids,
        )
        ##PLACEHOLDER -- SINCE THIS IS NO LONGER MEANINGFUL FOR THE NEW OUTPUT
        num_feedback_new = -1

        # Loop over unique sinks.
        for j, sink_ID in enumerate(sink_IDs):

            if verbose:
                print("Getting data for sink ID {0:d}...".format(sink_ID), flush=True)

            # Get particle IDs of accreted non-feedback gas particles.
            acc_gas_ids = acc_dict[sink_ID]["non_fb_ids"]
            # num_feedback = np.sum(acc_dict[sink_ID]["fb_counts"])
            ##PLACEHOLDER SINCE THIS IS NO LONGER MEANINGFUL
            num_feedback = -1

            # Get idx of unique accreted non-feedback gas particles.
            idx_g, num_feedback_new = self.get_idx(acc_gas_ids)
            ##PLACEHOLDER -- SINCE THIS IS NO LONGER MEANINGFUL FOR THE NEW OUTPUT
            num_feedback_new = -1
            if len(idx_g) == 0:
                continue

            ##FINDING PARTICLES OF 2ND ARRAY WITHIN MAX_DIST OF DENSITY PEAK FROM 1ST ARRAY.
            idx_g_all_filtered = self.get_filt_peak_two_array(
                idx_g,
                idx_g_all,
                max_dist=max_dist,
            )
            if len(idx_g_all_filtered) == 0:
                continue
            idx_star = self.get_filt_peak_two_array(
                idx_g,
                np.arange(len(self.stars_x)).astype(int),
                max_dist=max_dist,
                star=True,
            )

            # Get gas properties.
            data = self.get_gas_data(
                idx_g_all_filtered,
                num_feedback,
                num_feedback_new,
                skip_potential=skip_potential,
            )
            # all_data[j, 0] = sink_ID  # Record sink ID.
            # all_data[j, 1:] = data
            # Save the sink_id, and assign the rest of the fields simultaneously
            all_data[j]["sink_id"] = sink_ID
            for field in data.dtype.names:
                all_data[j][field] = data[field]
            all_data[j]["num_stars"] = len(idx_star)

        return all_data

    # Write gas properties data to HDF5 file.
    def write_to_file(
        self,
        all_data,
        datadir,
        use_all_sinks=True,
        sink_imin=None,
        sink_imax=None,
        max_dist=np.inf,
    ):

        i = self.get_i()

        if use_all_sinks:
            fname = os.path.join(
                datadir,
                "snapshot_{0:03d}_accreted_gas_properties_{1}.pq".format(i, max_dist),
            )
        else:
            fname = os.path.join(
                datadir,
                "snapshot_{0:03d}_accreted_gas_properties_range_{1:d}_{2:d}_{3}.pq".format(
                    i, sink_imin, sink_imax, max_dist
                ),
            )
        all_data = pd.DataFrame(data=all_data)
        all_data.attrs["time"] = self.t
        all_data.attrs["m_unit"] = self.m_unit
        all_data.attrs["l_unit"] = self.l_unit
        all_data.attrs["v_unit"] = self.v_unit
        all_data.attrs["t_unit"] = self.t_unit
        all_data.attrs["B_unit"] = self.B_unit

        all_data.to_parquet(fname)

    # Utility functions.
    def weight_avg(self, data, weights):
        "Weighted average"
        weights = np.abs(weights)
        weightsum = np.sum(weights)
        if weightsum > 0:
            return np.sum(data * weights) / weightsum
        else:
            return 0

    def weight_std(self, data, weights):
        "Weighted standard deviation."
        weights = np.abs(weights)
        weightsum = np.sum(weights)
        if weightsum > 0:
            return np.sqrt(
                np.sum(((data - self.weight_avg(data, weights)) ** 2) * weights)
                / weightsum
            )
        else:
            return 0

    def _sigmoid_sqrt(self, x):
        return 0.5 * (1 + x / np.sqrt(1 + x * x))

    def _DMIN(self, a, b):
        return np.where(a < b, a, b)

    def _DMAX(self, a, b):
        return np.where(a > b, a, b)
