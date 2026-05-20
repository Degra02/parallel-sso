#!/bin/env bash

while [[ $# -gt 0 ]]; do
  case $1 in
  -x | --exec)
    EXEC="$2"
    shift # past argument
    shift # past value
    ;;
  -j | --job)
    JOB_NAME="$2"
    shift # past argument
    shift # past value
    ;;
  *)
    echo "Unexpected option '$1'"
    exit 1
    ;;
  esac
done

DEFAULT_POP=100
DEFAULT_DIMS=30
DEFAULT_STAGES=1000
DEFAULT_ROT=10

for (( i=1; i <= 16; i++ )); do
    while [[ $(qstat -u $USER | wc -l) -ge 30 ]]; do
      sleep 10
    done
    POP=$(( DEFAULT_POP * i ))
    JOB_NAME_TMP=${JOB_NAME}_${POP}
    sed \
      -e "s/\${JOB_NAME}/$JOB_NAME_TMP/g" \
      -e "s/\${POP}/$POP/g" \
      -e "s/\${DIMS}/$DEFAULT_DIMS/g" \
      -e "s/\${STAGES}/$DEFAULT_STAGES/g" \
      -e "s/\${ROT}/$DEFAULT_ROT/g" \
      ${EXEC} | qsub
done

for (( i=1; i <= 16; i++ )); do
    while [[ $(qstat -u $USER | wc -l) -ge 30 ]]; do
      sleep 10
    done
    DIMS=$(( DEFAULT_DIMS * i ))
    JOB_NAME_TMP=${JOB_NAME}_${DIMS}
    sed \
      -e "s/\${JOB_NAME}/$JOB_NAME_TMP/g" \
      -e "s/\${POP}/$DEFAULT_POP/g" \
      -e "s/\${DIMS}/$DIMS/g" \
      -e "s/\${STAGES}/$DEFAULT_STAGES/g" \
      -e "s/\${ROT}/$DEFAULT_ROT/g" \
      ${EXEC} | qsub
done

for (( i=1; i <= 16; i++ )); do
    while [[ $(qstat -u $USER | wc -l) -ge 30 ]]; do
      sleep 10
    done
    STAGES=$(( DEFAULT_STAGES * i ))
    JOB_NAME_TMP=${JOB_NAME}_${STAGES}
    sed \
      -e "s/\${JOB_NAME}/$JOB_NAME_TMP/g" \
      -e "s/\${POP}/$DEFAULT_POP/g" \
      -e "s/\${DIMS}/$DEFAULT_DIMS/g" \
      -e "s/\${STAGES}/$STAGES/g" \
      -e "s/\${ROT}/$DEFAULT_ROT/g" \
      ${EXEC} | qsub
done

for (( i=1; i <= 16; i++ )); do
    while [[ $(qstat -u $USER | wc -l) -ge 30 ]]; do
      sleep 10
    done
    ROT=$(( DEFAULT_ROT * i ))
    JOB_NAME_TMP=${JOB_NAME}_${ROT}
    sed \
      -e "s/\${JOB_NAME}/$JOB_NAME_TMP/g" \
      -e "s/\${POP}/$DEFAULT_POP/g" \
      -e "s/\${DIMS}/$DEFAULT_DIMS/g" \
      -e "s/\${STAGES}/$DEFAULT_STAGES/g" \
      -e "s/\${ROT}/$ROT/g" \
      ${EXEC} | qsub
done