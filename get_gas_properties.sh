#!/bin/bash

#SBATCH -J get_acc_data
#SBATCH -p small # Queue
#SBATCH -o acc_data_%j.o # Name of stdout output file 
#SBATCH -e acc_data_%j.e # Name of stderr error file
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
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

# ./get_simulation_accreted_gas_properties.py
python3 -m cProfile -o get_gas.prof get_simulation_accreted_gas_properties.py
