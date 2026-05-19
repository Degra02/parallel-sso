#!/bin/env bash

while [[ $# -gt 0 ]]; do
  case $1 in
  -p | --procs)
    N_PROC="$2"
    shift # past argument
    shift # past value
    ;;
  -t | --thrds)
    N_THRD="$2"
    shift # past argument
    shift # past value
    ;;
  -j | --job)
    JOB_NAME="$2"
    shift # past argument
    shift # past value
    ;;
  -e | --exec)
    EXEC="$2"
    shift # past argument
    shift # past value
    ;;
  *)
    echo "Unexpected option '$1'"
    exit 1
    ;;
  esac
done

# if [ -z "$N_PROC" ]; then
#   echo "-p|--procs is mandatory"
#   exit 1
# fi
# if [ -z "$N_THRD" ]; then
#   echo "-t|--thrds is mandatory"
#   exit 1
# fi
# if [ -z "$JOB_NAME" ]; then
#   echo "-j|--job is mandatory"
#   exit 1
# fi

for ((procs=1; procs<=N_PROC; procs*=2)); do
  for ((thrds=1; thrds<=N_THRD; thrds*=2)); do
    while [[ $(qstat -u $USER | wc -l) -ge 30 ]]; do
      sleep 10
    done
    sed -e "s/\${N_THRD}/$thrds/g" -e "s/\${N_PROC}/$procs/g" -e "s/\${JOB_NAME}/$JOB_NAME/g" "$EXEC" | qsub
  done
done
