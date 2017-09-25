cmake_minimum_required(VERSION 2.6)
project(benchmarks CXX)

set(BENCHMARKS_VERSION_MAJOR 0)
set(BENCHMARKS_VERSION_MINOR 1)
set(BENCHMARKS_VERSION_PATCH 0)

# qt variables
set( BENCHMARKS_HDRS)
set( BENCHMARKS_UIS main.ui )
set( BENCHMARKS_RCS )


# prohibit in-source builds
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please use "
                        "./configure to choose a build directory and "
                        "initialize the build configuration.")
endif ()

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# check if the user provided CXXFLAGS on the command line
if (CMAKE_CXX_FLAGS)
    set(CXXFLAGS_PROVIDED true)
endif ()
if (CXXFLAGS_PROVIDED)
  set(CMAKE_CXX_FLAGS_DEBUG          "")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "")
  set(CMAKE_CXX_FLAGS_RELEASE        "")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")
else ()
  set(CMAKE_CXX_FLAGS                "-std=c++11 -Wextra -Wall -pedantic")
  set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os")
  set(CMAKE_CXX_FLAGS_RELEASE        "-O4")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
endif ()

# check for g++ >= 4.7 or clang++ > = 3.2
try_run(ProgramResult
        CompilationSucceeded
        ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/src/get_compiler_version.cpp
        RUN_OUTPUT_VARIABLE CompilerVersion)
if (NOT CompilationSucceeded OR NOT ProgramResult EQUAL 0)
  message(FATAL_ERROR "Cannot determine compiler version")
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  if (CompilerVersion VERSION_GREATER 4.6)
    message(STATUS "Found g++ version ${CompilerVersion}")
  else ()
    message(FATAL_ERROR "g++ >= 4.7 required (found: ${CompilerVersion}.")
  endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  if (CompilerVersion VERSION_GREATER 3.1)
    message(STATUS "Found clang++ version ${CompilerVersion}")
  else ()
    message(FATAL_ERROR "clang++ >= 3.2 required (found: ${CompilerVersion}.")
  endif ()
  if (NOT CXXFLAGS_PROVIDED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  endif ()
else ()
  message(FATAL_ERROR "Your C++ compiler does not support C++11.")
endif ()

if (CAF_LOG_LEVEL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCAF_LOG_LEVEL=${CAF_LOG_LEVEL}")
endif(CAF_LOG_LEVEL)

# set build type (evaluate ENABLE_DEBUG flag)
if (ENABLE_DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCAF_DEBUG")
elseif (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()

if (DISABLE_CONTEXT_SWITCHING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCAF_DISABLE_CONTEXT_SWITCHING")
endif ()

find_package(CAF COMPONENTS core opencl REQUIRED)

#find opencl
find_package(OPENCL REQUIRED)

# we need this to be able to include headers produced by uic in our code
# (CMAKE_BINARY_DIR holds a path to the build directory, while INCLUDE_DIRECTORIES()
# works just like INCLUDEPATH from qmake)
INCLUDE_DIRECTORIES(. include ${CMAKE_BINARY_DIR} ${CAF_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})

file(GLOB HEADERS "include/*.hpp")

add_executable(bench_caf_comparison src/opencl_caf.cpp src/util.cpp ${HEADERS})
target_link_libraries(bench_caf_comparison ${CMAKE_DL_LIBS} ${CAF_LIBRARIES} ${OpenCL_LIBRARIES})

add_executable(bench_native_comparison src/opencl_native.cpp src/util.cpp src/cmd.cpp ${HEADERS})
target_link_libraries(bench_native_comparison ${CMAKE_DL_LIBS} ${CAF_LIBRARIES} ${OpenCL_LIBRARIES})

add_executable(bench_overhead src/opencl_overhead.cpp src/util.cpp ${HEADERS})
target_link_libraries(bench_overhead ${CMAKE_DL_LIBS} ${CAF_LIBRARIES} ${OpenCL_LIBRARIES})

add_executable(bench_matrix src/cpu_matrix.cpp ${HEADERS})
target_link_libraries(bench_matrix ${CMAKE_DL_LIBS} ${CAF_LIBRARIES} ${OpenCL_LIBRARIES})

# add_executable(copy_ops_small src/copy_ops_small.cpp ${HEADERS})
# target_link_libraries(copy_ops_small ${CMAKE_DL_LIBS} ${CAF_LIBRARIES} ${OpenCL_LIBRARIES})

# add_executable(copy_ops_big src/copy_ops_big.cpp ${HEADERS})
# target_link_libraries(copy_ops_big ${CMAKE_DL_LIBS} ${CAF_LIBRARIES} ${OpenCL_LIBRARIES})

add_executable(bench_spawn_cl src/spawn_time.cpp src/util.cpp ${HEADERS})
target_link_libraries(bench_spawn_cl ${CMAKE_DL_LIBS} ${CAF_LIBRARIES} ${OpenCL_LIBRARIES})

add_executable(bench_spawn_core src/spawn_time_core.cpp ${HEADERS})
target_link_libraries(bench_spawn_core ${CMAKE_DL_LIBS} ${CAF_LIBRARIES} ${OpenCL_LIBRARIES})

add_executable(list_devices src/list_devices.cpp src/util.cpp ${HEADERS})
target_link_libraries(list_devices ${CMAKE_DL_LIBS} ${OpenCL_LIBRARIES})

# collect all compiler flags
string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)
set(ALL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${UPPER_BUILD_TYPE}}")
message(STATUS
      "\n====================|  Build Summary  |===================="
      "\nBuild type:      ${CMAKE_BUILD_TYPE}"
      "\nCXX:             ${CMAKE_CXX_COMPILER}"
      "\nCXXFLAGS:        ${ALL_CXX_FLAGS}"
      "\nLIBRARIS:        ${LD_FLAGS}"
      "\n"
      "\n===========================================================\n")
