#!/bin/sh
#
# Configure wukong at the file 'config'
# List all nodes at the file 'mpd.host'
# Share the input RDF data among all nodes through DFS (e.g., NFS or HDFS)

# Standalone Mode (w/o mpi):
# ../build/wukong config

# Distributed Mode (w/ mpi):
# NOTE: the hostfile of of mpiexec must match that of wukong (i.e., mpd.hosts)
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1
/usr/local/bin/mpiexec -x CUDA_DEVICE_WAITS_ON_EXCEPTION -x LD_LIBRARY_PATH -hostfile mpd.hosts -n $1 ../build/wukong config mpd.hosts
