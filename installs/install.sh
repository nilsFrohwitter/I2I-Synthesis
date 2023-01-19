#!/bin/bash

# This is only the install script for working on slurm with enroot images. The detailed package list is elsewhere and can be installed with install_requirements

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then

  # put your pip install commands here
  
  # put your install commands here:
  conda install -c conda-forge nibabel -y
  
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

# This runs your wrapped command
"$@"
