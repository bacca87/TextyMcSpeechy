
#!/bin/bash
# run_container.sh:  This script provides a single alias to one of the available ways of starting a docker container.
#
# use one of the following options in this script
# bash prebuilt_container_run.sh  # launches prebuilt docker images which you downloaded
#    bash local_container_run.sh  # launches images you built locally

# The training stack now requires piper1-gpl, which is built from source in the
# Dockerfile. The prebuilt image on Docker Hub has not been updated for it, so
# default to the locally built image. Use prebuilt_container_run.sh instead if
# you have published/use an updated prebuilt image.
bash local_container_run.sh
