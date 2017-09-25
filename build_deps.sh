#!/bin/bash

ROOT_DIR=$(pwd)

git submodule update --init --recursive

cd $ROOT_DIR/actor-framework
./configure --build-type=release
make -j 4

cd $ROOT_DIR/vast
./configure --build-type=release --with-caf=$ROOT_DIR/actor-framework/build
make -j 4

cd $ROOT_DIR/indexing
./configure --build-type=release --with-caf=$ROOT_DIR/actor-framework/build --with-vast=$ROOT_DIR/vast/build
make -j 4

cd $ROOT_DIR/benchmarks
./configure --build-type=release --with-caf=$ROOT_DIR/actor-framework/build
make -j 4
