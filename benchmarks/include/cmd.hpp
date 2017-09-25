#ifndef CMD_HPP
#define CMD_HPP

#include <mutex>
#include <vector>
#include <string>
#include <future>
#include <condition_variable>

#include "include/util.hpp"

class cmd {
public:
  cmd(size_t size, kernel_ptr kernel, context_ptr context,
      command_queue_ptr queue, size_t iterations);
  ~cmd();
  void enqueue();
  void wait();

private:
  size_t size_;
  std::mutex mtx_;
  std::condition_variable cv_;
  cl_mem buf_in_1_;
  cl_mem buf_in_2_;
  cl_mem buf_out_;
  cl_event write_events_[2];
  cl_event kernel_event_;
  cl_event read_event_;
  cl_event marker_;
  kernel_ptr kernel_;
  context_ptr context_;
  command_queue_ptr queue_;

  size_t max_iterations_;
  size_t current_iterations_;

  std::vector<float> matrix_1_;
  std::vector<float> matrix_2_;
  std::vector<float> result_;
  std::vector<size_t> dimensions_;

  void make_decision();
};

#endif //CMD_HPP
