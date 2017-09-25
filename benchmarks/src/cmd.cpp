#include <numeric>
#include <iomanip>
#include <iterator>
#include <iostream>

#include "include/cmd.hpp"
#include "include/config.hpp"

using namespace std;

cmd::cmd(size_t size, kernel_ptr kernel, context_ptr context,
         command_queue_ptr queue, size_t iterations)
  : size_(size),
    kernel_(kernel),
    context_(context),
    queue_(queue),
    max_iterations_(iterations),
    current_iterations_(0),
//    matrix_1_(size * size),
//    matrix_2_(size * size),
//    result_(size * size),
    dimensions_({size, size}) {

}

cmd::~cmd() {
//  clReleaseMemObject(buf_in_1_);
//  clReleaseMemObject(buf_in_2_);
//  clReleaseMemObject(buf_out_);
  clReleaseEvent(write_events_[0]);
  clReleaseEvent(write_events_[1]);
  clReleaseEvent(kernel_event_);
  clReleaseEvent(read_event_);
}

void cmd::enqueue() {
  cl_int err;
  auto matrix_size = size_ * size_;
  auto buffer_size = sizeof(float) * matrix_size;
  matrix_1_.resize(matrix_size);
  matrix_2_.resize(matrix_size);
  iota(begin(matrix_1_), end(matrix_1_), 0);
  iota(begin(matrix_2_), end(matrix_2_), 0);
  result_.resize(matrix_size);

  buf_in_1_ = clCreateBuffer(context_.get(), CL_MEM_READ_ONLY, buffer_size, nullptr, &err);
  check_cl_error(err, "clCreateBuffer");
  buf_in_2_ = clCreateBuffer(context_.get(), CL_MEM_READ_ONLY, buffer_size, nullptr, &err);
  check_cl_error(err, "clCreateBuffer");
  buf_out_ = clCreateBuffer(context_.get(), CL_MEM_WRITE_ONLY, buffer_size, nullptr, &err);
  check_cl_error(err, "clCreateBuffer");
  
  err = clEnqueueWriteBuffer(queue_.get(), buf_in_1_, CL_FALSE, 0,
                             buffer_size, matrix_1_.data(),
                             0, nullptr, &write_events_[0]);
  err = clEnqueueWriteBuffer(queue_.get(), buf_in_2_, CL_FALSE, 0,
                             buffer_size, matrix_2_.data(),
                             0, nullptr, &write_events_[1]);

#if defined(__APPLE__)
  err = clEnqueueMarkerWithWaitList(queue_.get(), 2, write_events_, &marker_);
#else
  err = clEnqueueMarker(queue_.get(), &marker_);
#endif

  err = clSetKernelArg(kernel_.get(), 0, sizeof(cl_mem), (void*) &buf_in_1_);
  check_cl_error(err, "clSetKernelArg");
  err = clSetKernelArg(kernel_.get(), 1, sizeof(cl_mem), (void*) &buf_in_2_);
  check_cl_error(err, "clSetKernelArg");
  err = clSetKernelArg(kernel_.get(), 2, sizeof(cl_mem), (void*) &buf_out_);
  check_cl_error(err, "clSetKernelArg");

  // enqueue kernel
  err = clEnqueueNDRangeKernel(queue_.get(), kernel_.get(), dimensions_.size(),
                               nullptr,            // work item offsets
                               dimensions_.data(), // golbal dimensions
                               nullptr,            // local dimensions
                               1, &marker_, &kernel_event_);
  check_cl_error(err, "clEnqueueNDRangeKernel");
  err = clEnqueueReadBuffer(queue_.get(), buf_out_, CL_TRUE, 0,
                            sizeof(float) * result_.size(),
                            result_.data(), 1, &kernel_event_, &read_event_);
  check_cl_error(err, "clEnqueueReadBuffer");
  clFlush(queue_.get());

  // set callback for event
  err = clSetEventCallback(read_event_, CL_COMPLETE,
                           [](cl_event, cl_int, void* data) {
                               auto c = reinterpret_cast<cmd*>(data);
                               c->make_decision();
                           },
                           this);
  check_cl_error(err, "clSetEventCallback");
}

void cmd::wait() {
  unique_lock<mutex> lk(mtx_);
  cv_.wait(lk);
}

void cmd::make_decision() {
  ++current_iterations_;
#ifdef CL_ENABLE_DEBUG
  if (current_iterations_ >= max_iterations_) {
    for (size_t column = 0; column < size_; ++column) {
      for (size_t row = 0; row < size_; ++row) {
        cout << std::fixed << setprecision(2) << setw(9)
             << result_[row + column * size_];
      }
      cout << endl;
    }
  }
#endif
  clReleaseMemObject(buf_in_1_);
  clReleaseMemObject(buf_in_2_);
  clReleaseMemObject(buf_out_);
  if (current_iterations_ >= max_iterations_) {
    cv_.notify_one();
  } else {
    enqueue();
  }
}
