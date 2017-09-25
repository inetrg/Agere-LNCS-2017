# Agere-LNCS-2017

This repository exposes source code to reproduce experiments as presented in "OpenCL Actors - Adding Data Parallelism to Actor-based Programming with CAF".

This repository contains submodules. To get all contents, please clone with `git clone --recursive https://github.com/inetrg/Agere-LNCS-2017.git`. The command will clone and checkout a dedicated CAF version, VAST version and the Indexing repository.



## Prerequisites

In addition to a host that runs OpenCL, you will need the following projects. The script `build_deps.sh` will build them for you. If you want to do this manually, see `Building` below.

- [CAF](https://github.com/actor-framework/actor-framework)
- [VAST](https://github.com/vast-io/vast)
- [Bitmap Indexing](https://github.com/josephnoir/indexing)

All project require Cmake (>3.1) and a C++ compiler to build. CAF specifically depends on C++11 while VAST requires a C++14 compatible compiler.

For the measurements presented in the Paper, we used the following hosts:
- a late 2013 iMac with a 3.5 GHz Intel Core i7 running OS X 10.12 and OpenCL version 1.2
  - NVIDIA GeForce GTX 780M GPU with 4096 MB memory
- a Server with two twelve-core Intel Xeon CPUs clocked at 2.5 GHz running Linux (CentOS 7)
  - Tesla C2075 GPU with Nvidia driver version 375.20
  - Xeon Phi 5110P coprocessor using the Intel OpenCL Runtime 14.2



## Benchmarks

The paper present multiple benchmarks which are listed in separate sections below. The benchmark implementations are located in the `benchmarks` folder with an exception of the benchmark related to bitmap indexing which is located in the separate `indexing` repository that is included as a submodule.

The program `list_devices` included in the benchmarking programs can list the OpenCL devices available on your system.


### Use Case: Indexing

The paper (Section 4.2) shows a graph on indexing that compares OpenCL Actors to a CPU implementation using VAST. There is a program to generate test data (`generate`) in the build with the indexing project. Per default it generates values with 16 bit cardinality.

The programs for the comparison are the `phases` and `vst` executables. Both are available in the related files in the `src` folder. The script `measure.sh` in the indexing folder will create test data and perform the measurements for you.

If you want to perform different measurements, each program accepts a file with input data using the option `-f`. Per default, `phases` will choose the first available OpenCL device. You can also specify a device name in the code (line 337) or pass it as an argument via `-d`. For further information see `--help`.


### Pipeline Messaging Overhead

The indexing project further builds the `overhead` executable to estimate the time for passing messages between OpenCL actor stages (shortly mentioned in Section 3.6). It accepts the argument `--use_mapping_functions` to decide whether the mapping functions or round trip time for a complete calculation is used for the estimate. Additionally, the argument `-i I` determines how many iterations to execute.

The program prints the mean over I iterations in microseconds.


### Spawn Time

This benchmark is presented in Section 5.1. It is measured by two programs, one for core actors (`bench_spawn_core`) and one for OpenCL actors (`bench_spawn_cl`).

The OpenCL benchmark requires a matrix size > 0  (`-s N`) as well as an iteration count (`-i I`) that specifies how many actors will be spawned. The core actor benchmark only requires the iteration count (`-i I`) and ignores the size argument. The paper includes measurements for values of I from 100,000 to 1,000,000 in steps of 100,000.

Each program prints the time to create `I` actors in microseconds.


### Runtime Overhead

This benchmark is presented in Section 5.2 of the paper. It requires some adjustments to CAF as the timepoints used for measurement are usually included. The following external variables must be added to the command class `actor-framework/libcaf_opencl/caf/opencl/command.hpp`
in lines 44 and 45:
```
static std::chrono::high_resolution_clock::time_point a;
static std::chrono::high_resolution_clock::time_point b;
```
In line 100 at the beginning of the first `enqueue` function
```
a = std::chrono::high_resolution_clock::now();
```
And in line 229 in the first line of the `handle_results` function
```
  b = std::chrono::high_resolution_clock::now();
```

The related program is called `bench_overhead` and requires a matrix size as input (`-s N`). The paper includes measurements for sizes of `N` = 1000, 4000, 8000 and 12000. The benchmarks prints three measurements in microseconds: the total runtime, the time spent in OpenCL and the difference between both values (separated by commas).


### Baseline Comparison

The benchmark is presented in Section 5.3 of the paper. The programs for the comparison are `bench_native_comparison` for native OpenCL and `bench_caf_comparison` for the OpenCL actor. Both require a matrix size `-s N` and an iteration count `-i I`. Optionally, a device can be specified with `-d D`. The paper uses a size of `N` equal to 1,000 and iterations `I` from 1,000 to 10,000 in steps of 1,000.

Each program prints the runtime for `I` iterations in microseconds.


### Scaling in a heterogeneous setup

Section 5.4

- [ ] Find benchmark code


### Measurement Data

The Origin project file in the repository includes the data and the graphs in the paper.



## Building

The script `build_deps.sh` will build all project for you, but this can be done manually via the following commands:


### CAF

```
cd actor-framework
./configure --build-type=release
make -j 4
```


### VAST

```
cd vast
./configure --build-type=release --with-caf=../actor-framework/build
make -j 4
```


### Indexing

```
cd indexing
./configure --build-type=release --with-caf=../actor-framework/build --with-vast=../vast/build
make -j 4
```


### Benchmarks

```
cd benchmarks
./configure --build-type=release --with-caf=../actor-framework/build
make -j 4
```



## Notes

For more CAF-related benchmarks, see the [CAF Benchmark Suit](https://github.com/actor-framework/benchmarks). It compares CAF to various actor implementations such as Scala or Erlang. For more general actor benchmarks, take a look at the [Savina benchmark suite](https://github.com/shamsimam/savina).
