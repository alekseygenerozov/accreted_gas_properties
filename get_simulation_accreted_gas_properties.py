#!/usr/bin/env python3

import os
import sys

import numpy as np
import pandas as pd

sys.path.append("/work/08381/nina_af/frontera/accreted_gas_properties")

from sink_accretion_history_speedup import (
    Cloud,
    SinkAccretionHistory,
    SnapshotGasProperties,
)

# bhdir     = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/blackhole_details/'
# snapdir   = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'
# datadir   = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/accreted_gas_properties_v2/'

bhdir = "blackhole_details/"
snapdir = "/scratch3/03532/mgrudic/STARFORGE_RT/STARFORGE_v1.1/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42/output/"
datadir = "./"

# fname_gas = os.path.join(bhdir, "sink_accretion_data.txt")
# fname_sink = os.path.join(bhdir, "sink_formation_data.txt")
fname_gas = None
fname_sink = None

print("Getting accretion dict...")
acc_dict = SinkAccretionHistory(
    bhdir, fname_gas=fname_gas, fname_sink=fname_sink
).accretion_dict

M0, R0, alpha0 = 2e4, 10.0, 2.0
cloud = Cloud(M0, R0, alpha0)


def get_fname(i, snapdir=snapdir):
    return os.path.join(snapdir, "snapshot_{0:03d}.hdf5".format(i))


# Set snapshot range (later: pass as argument/detect from snapshot directory?)
i_min, i_max = 0, 489  #

# Set sink particle range (split into batches of ~20 sink particles).
sink_imin, sink_imax = 0, 99
use_all_sinks = True
max_dist = 0.5
first_snap_table = None
if os.path.exists("first_snap_table.pq"):
    first_snap_table = pd.read_parquet("first_snap_table.pq")


# Loop over snapshots.
for i in range(i_min, i_max + 1, 1):

    print("Writing snapshot {0:d}...".format(i), flush=True)
    s = SnapshotGasProperties(get_fname(i), cloud)
    all_data = s.get_all_gas_data(
        acc_dict,
        skip_potential=False,
        use_all_sinks=use_all_sinks,
        sink_imin=sink_imin,
        sink_imax=sink_imax,
        max_dist=max_dist,
        first_snap_table=first_snap_table,
    )
    s.write_to_file(
        all_data,
        datadir,
        use_all_sinks=use_all_sinks,
        sink_imin=sink_imin,
        sink_imax=sink_imax,
        max_dist=max_dist,
    )
