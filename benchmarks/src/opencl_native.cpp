#include <vector>
#include <string>
#include <cstring>
#include <numeric>
#include <iostream>

#if defined __APPLE__ || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include "include/cmd.hpp"
#include "include/util.hpp"
#include "include/config.hpp"
#include "include/kernel.hpp"

using namespace std;

void usage(const char* prog) {
  cout << "usage: ./" << prog << endl
       << "  -s <size>        (matrix size, required)" << endl
       << "  -i <iterations>  (iterations to measure, required)" << endl
       << "  -d <device-name> (choose the device to use)" << endl
       << "The program only accepts arguments in that exact order." << endl;
}

int main(int argc, char** argv) {
  string device_wish;
  if (argc < 5 || string(argv[1]) != "-s" || string(argv[3]) != "-i") {
    usage(argv[0]);
    return 0;
  } else if (argc == 6 || argc > 7) {
    usage(argv[0]);
    return 0;
  } else if (argc == 7) {
    device_wish = string(argv[6]);
  }
  
  cl_int err = 0;
  
  // find platforms
  cl_uint num_platforms = 0;
  err = clGetPlatformIDs(0, nullptr, &num_platforms);
  check_cl_error(err, "clGetPlatformIDs");
  vector<cl_platform_id> platform_ids(num_platforms);
  err = clGetPlatformIDs(num_platforms, platform_ids.data(), nullptr);
  check_cl_error(err, "clGetPlatformIDs");
  cl_uint num_devices = 0;
  cl_device_id device = 0;
  vector<cl_device_id> devices;
  vector<cl_uint> device_types{CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU,
                               CL_DEVICE_TYPE_ACCELERATOR};
  for (auto& dev_type : device_types) {
    for (auto& platform : platform_ids) {
      err = clGetDeviceIDs(platform, dev_type, 0, nullptr, &num_devices);
      if (num_devices == 0)
        continue;
      check_cl_error(err, "clGetDeviceIDs");
      devices.resize(num_devices);
      err = clGetDeviceIDs(platform, dev_type, num_devices,
                           devices.data(), nullptr);
      check_cl_error(err, "clGetDeviceIDs");
      if (device_wish.empty()) {
        device = devices.front();
        break;
      } else {
        for (auto dev : devices) {
          size_t ret = 0;
          size_t size = 256;
          vector<char> buf(size);
          err = clGetDeviceInfo(dev, CL_DEVICE_NAME, size, buf.data(), &ret);
          string name(buf.data());
          if (name == device_wish) {
            device = dev;
            break;
          }
        }
      }
    }
    if (device != 0)
      break;
  }
  if (device == 0) {
    cout << "No OpenCL device " << (device_wish.empty() ? "" : "'")
         << device_wish << (device_wish.empty() ? "" : "' ")
         << "found." << endl;
    return 0;
  }
  
  auto matrix_size = static_cast<size_t>(stoi(argv[2]));
  auto iterations  = static_cast<size_t>(stoi(argv[4]));
  // create context
  context_ptr context;
  context.adopt(clCreateContext(0, 1, &device, nullptr, nullptr, &err));
  check_cl_error(err, "clCreateContext");
  // create command queue
  command_queue_ptr queue;
  queue.adopt(clCreateCommandQueue(context.get(), device,
                                   CL_QUEUE_PROFILING_ENABLE, &err));
  check_cl_error(err, "clCreateCommandQueue");
  // create program
  const char* src     = kernel_source;
  size_t      src_len = strlen(src);
  program_ptr prog;
  prog.adopt(clCreateProgramWithSource(context.get(), 1,
                                       static_cast<const char**>(&src),
                                       &src_len, &err));
  check_cl_error(err, "clCreateProgramWithSource");
  // build program from program object
  err = clBuildProgram(prog.get(), 0, nullptr, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    auto err_string = get_opencl_error(err);
    cl_build_status bs;
    err = clGetProgramBuildInfo(prog.get(), device, CL_PROGRAM_BUILD_STATUS,
                                sizeof(cl_build_status), &bs, nullptr);
    size_t ret_val_size;
    err = clGetProgramBuildInfo(prog.get(), device, CL_PROGRAM_BUILD_LOG,
                                0, nullptr, &ret_val_size);
    vector<char> build_log(ret_val_size+1);
    err = clGetProgramBuildInfo(prog.get(), device, CL_PROGRAM_BUILD_LOG,
                                ret_val_size, build_log.data(), nullptr);
    build_log[ret_val_size] = '\0';
    throw std::runtime_error("'clBuildProgram failed': " + err_string +
                             "\nBuild log: " + string(build_log.data()));
  }
  // init kernel
  kernel_ptr kernel;
  kernel.adopt(clCreateKernel(prog.get(), kernel_name, &err));
  check_cl_error(err, "clCreateKernel");
  cmd c(matrix_size, kernel, context, queue, iterations);
  auto start_ = chrono::high_resolution_clock::now();
  c.enqueue();
  c.wait();
  auto end_ = chrono::high_resolution_clock::now();
  cout << chrono::duration_cast<chrono::microseconds>((end_ - start_)).count()
       << endl;
}
