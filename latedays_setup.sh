#!/bin/bash
# Usage: ". latedays_setup.sh"

module load gcc-4.9.2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda/bin:${PATH}
