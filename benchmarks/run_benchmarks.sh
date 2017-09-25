#!/bin/bash

problem=1000
from=1000
to=10000
step=1000
measurements=50
device="GeForce GT 650M"

bench_root=".."

usage="\
Usage: $0
    --problem       matrix problem size (default: $problem)
    --from          start of iterations (default: $from)
    --to            end of iterations (deafult: $to)
    --step          iteration increment (default: $step)
    --measurements  measurements per iteration (default: $measurements)
    --device        set the OpenCL device for the execution (default: $device)
    --help          print this text
"

while [ $# -ne 0 ]; do
    case "$1" in
        -*=*) optarg=`echo "$1" | sed 's/[-_a-zA-Z0-9]*=//'` ;;
        *) optarg= ;;
    esac

    case "$1" in
        --help|-h)
            echo "${usage}" 1>&2
            exit 1
            ;;
        --step=*|-s=*)
            step=$optarg
            ;;
        --from=*|-f=*)
            from=$optarg
            ;;
        --to=*|-t=*)
            to=$optarg
            ;;
        --problem=*|-p=*)
            problem=$optarg
            ;;
        --measurements=*|-m=*)
            measurements=$optarg
            ;;
        --device=*|-d=*)
            device=$optarg
            ;;
        *)
            echo "Invalid option '$1'.  Try $0 --help to see available options."
            exit 1
            ;;
    esac
    shift
done

# cl_spawn core_spawn caf_comparision native_comparison caf_overhead
for i in caf_comparison native_comparison
do
    for iteration in $(seq $from $step $to); do
        next_file="${iteration}_cores_runtime_${i}.txt"
        rm -f $next_file
        # echo "$next_file"
        echo "[executing] ${bench_root}/build/bin/$i -s "$problem" -i $iteration -d $device >> $next_file"
        for measurement in $(seq 1 $measurements) ; do
          ${bench_root}/build/bin/$i -s $problem -i $iteration -d $device >> $next_file
        done
        echo "" >> $next_file
    done
done
