#include <iostream>
#include <fstream>

#include "include/util.hpp"

#if defined __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

using namespace std;

/*
 * Prints a buch of info about a gpu or cpu device.
 */
void print_device_info(cl_device_id id) {
  cl_int err;
  size_t return_size;
  size_t buf_size = 128;
  vector<char> buf(buf_size);

  err = clGetDeviceInfo(id, CL_DEVICE_NAME, buf_size, buf.data(), &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_NAME." << endl;
  }
  cout << ">>> '" << string(buf.data()) << "'" << endl;

  cl_uint max_compute_units = 0;
  err = clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_MAX_COMPUTE_UNITS." << endl;
  } else {
    cout << "  The number of parallel compute cores on the OpenCL device: '" << max_compute_units << "'" << endl;
  }

  size_t max_parameter_size = 0;
  err = clGetDeviceInfo(id, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &max_parameter_size, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_MAX_PARAMETER_SIZE for device." << endl;
  } else {
    cout << "  Max size in bytes of the arguments that can be passed to a kernel: '" << max_parameter_size << "'." << endl;
  }

  size_t max_work_group_size = 0;
  err = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_MAX_WORK_GROUP_SIZE for device." << endl;
  } else {
    cout << "  Maximum number of work-items in a work-group executing a kernel using the data parallel execution model: '" << max_work_group_size << "'" << endl;
  }

  cl_uint max_work_item_dimensions = 0;
  err = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dimensions, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS." << endl;
  } else {
    cout << "  Maximum dimensions that specify the global and local work-item IDs used by the data parallel execution model: '" << max_work_item_dimensions << "'" << endl;
  }

  vector<size_t> max_work_items_per_dimension(max_work_item_dimensions);
  err = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, &max_work_items_per_dimension, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_MAX_WORK_ITEM_SIZES." << endl;
  } else {
    cout << "  Maximum number of work-items that can be specified in each dimension of the work-group to clEnqueueNDRangeKernel: '(";
    for (size_t i = 0; i < max_work_item_dimensions; ++i) {
      cout << max_work_items_per_dimension[i];
      if (i < max_work_item_dimensions - 1) {
        cout << ", ";
      }
    }
    cout << ")'" << endl;
  }
  err = clGetDeviceInfo(id, CL_DEVICE_OPENCL_C_VERSION, buf_size, buf.data(), &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_OPENCL_C_VERSION." << endl;
  } else {
    cout << "  OpenCL C version string: '" << string(buf.data()) << "'" << endl;
  }

  cl_bool compiler_available;
  err = clGetDeviceInfo(id, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &compiler_available, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_COMPILER_AVAILABLE." << endl;
  } else {
    cout << "  Is CL_FALSE if the implementation does not have a compiler available to compile the program source. Is CL_TRUE if the compiler is available: '" << compiler_available << "'" << endl;
  }

  cl_bool is_little_endian;
  err = clGetDeviceInfo(id, CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool), &is_little_endian, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_ENDIAN_LITTLE for '" << id << "'." << endl;
  } else {
    cout << "  Is CL_TRUE if the OpenCL device is a little endian device and CL_FALSE otherwise: '" << is_little_endian << "'" << endl;
  }

  cl_ulong global_mem_cache_size;
  err = clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &global_mem_cache_size, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_GLOBAL_MEM_CACHE_SIZE." << endl;
  } else {
    cout << "  Size of global memory cache in bytes: '" << global_mem_cache_size << "'" << endl;
  }

  cl_ulong global_mem_size;
  err = clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_GLOBAL_MEM_SIZE for." << endl;
  } else {
    cout << "  Size of global device memory in bytes: '" << global_mem_size << "'" << endl;
  }

  cl_ulong local_mem_size;
  err = clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_LOCAL_MEM_SIZE for." << endl;
  } else {
    cout << "  Size of local memory arena in bytes: '" << local_mem_size << "'" << endl;
  }

  cl_uint max_clock_frequency;
  err = clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_clock_frequency, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_MAX_CLOCK_FREQUENCY." << endl;
  } else {
    cout << "  Maximum configured clock frequency of the device in MHz: '" << max_clock_frequency << "'" << endl;
  }

  cl_uint max_constant_args;
  err = clGetDeviceInfo(id, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(cl_uint), &max_constant_args, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_MAX_CONSTANT_ARGS." << endl;
  } else {
    cout << "  Max number of arguments declared with the __constant qualifier in a kernel: '" << max_constant_args << "'" << endl;
  }

  cl_ulong max_constant_buffer_size;
  err = clGetDeviceInfo(id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &max_constant_buffer_size, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE." << endl;
  } else {
    cout << "  Max size in bytes of a constant buffer allocation: '" << max_constant_buffer_size << "'" << endl;
  }

  cl_ulong max_mem_alloc_size;
  err = clGetDeviceInfo(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_MAX_MEM_ALLOC_SIZE." << endl;
  } else {
    cout << "  Max size of memory object allocation in bytes: '" << max_mem_alloc_size << "'" << endl;
  }

  size_t device_timer_resolution;
  err = clGetDeviceInfo(id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(size_t), &device_timer_resolution, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_PROFILING_TIMER_RESOLUTION." << endl;
  } else {
    cout << "  Describes the resolution of device timer (in nanoseconds): '" << device_timer_resolution << "'" << endl;
  }

  cl_command_queue_properties cq_props;
  err = clGetDeviceInfo(id, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &cq_props, &return_size);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting CL_DEVICE_QUEUE_PROPERTIES." << endl;
  } else {
    cout << "  Device has the following properties:" << endl;
    cout << "   CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE [";
    if (cq_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) cout << "x";
    else cout << " ";
    cout << "]" << endl << "   CL_QUEUE_PROFILING_ENABLE              [";
    if (cq_props & CL_QUEUE_PROFILING_ENABLE) cout << "x";
    else cout << " ";
    cout << "]" << endl;
  }
}

std::string type_string(cl_device_type device_type) {
  switch (device_type) {
    case CL_DEVICE_TYPE_GPU:
      return "GPU";
    case CL_DEVICE_TYPE_CPU:
      return "CPU";
    case CL_DEVICE_TYPE_ACCELERATOR:
      return "accelerator";
    default:
      return "unknown device";
  }
}

bool info_for(cl_platform_id platform, cl_device_type device_type) {
  cl_uint num_devices = 0;
  auto err = clGetDeviceIDs(platform, device_type, 0, NULL, &num_devices);
  cout << "> found " << num_devices << " " << type_string(device_type) << "(s)." << endl;
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error retrieving number of " << type_string(device_type) << "s: "
         << get_opencl_error(err) << endl;
    return false;
  }
  vector<cl_device_id> devices(num_devices);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Error getting " << type_string(device_type) << " device ids: "
         << get_opencl_error(err) << endl;
    return false;
  }
  cout << type_string(device_type) << " devices on the platfom:" << endl;
  for (size_t i = 0; i < num_devices; ++i)
    print_device_info(devices[i]);
  return true;
}

int main(void) {
  cl_int err = 0;

  cl_uint num_platforms;
  err = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (err != CL_SUCCESS) {
    cout << "[!!!] Could not get number of platforms: '" << get_opencl_error(err) << "'." << endl;
    return err;
  }
  cout << "Found " << num_platforms << " platform(s)." << endl;

  vector<cl_platform_id> platform_ids(num_platforms);
  err = clGetPlatformIDs(num_platforms, platform_ids.data(), nullptr);
  if (err != CL_SUCCESS) {
    cout << "[!!!] " << err << ": " << get_opencl_error(err) << endl;
    return err;
  }
  
  /* get name of our platform */
  for (size_t i = 0; i < num_platforms; ++i) {
    char name[128];
    size_t return_size;
    err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 128, name, &return_size);
    cout << "* " << name << endl;
    info_for(platform_ids[i], CL_DEVICE_TYPE_GPU);
    info_for(platform_ids[i], CL_DEVICE_TYPE_CPU);
    info_for(platform_ids[i], CL_DEVICE_TYPE_ACCELERATOR);
  }
}
