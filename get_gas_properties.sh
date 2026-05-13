#!/bin/bash

#SBATCH -J get_sink_data
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-user=ninaf@utexas.edu
#SBATCH --mail-type=all
#SBATCH -A AST23034

# source $HOME/.bashrc
# source $HOME/miniconda3/etc/profile.d/conda.sh

# conda init bash
# conda activate $HOME/miniconda3/envs/py311

# module unload python3

module load hdf5
module unload impi
module load python3/3.9.2

export QT_QPA_PLATFORM=offscreen

./get_simulation_accreted_gas_properties.py