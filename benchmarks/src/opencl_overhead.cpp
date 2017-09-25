#include <chrono>
#include <vector>
#include <iomanip>
#include <numeric>
#include <iostream>

#include "caf/all.hpp"
#include "caf/opencl/all.hpp"

#include "include/util.hpp"
#include "include/config.hpp"
#include "include/kernel.hpp"

using namespace std;
using namespace std::chrono;
using namespace caf;
using namespace caf::opencl;

#define ADDED_TIMEPOINTS_TO_CAF
#undef ADDED_TIMEPOINTS_TO_CAF

/**
 * Note that this benchmark requires modification of CAF.
 * The external variables must be added to the command class,
 *    actor-framework/libcaf_opencl/caf/opencl/command.hpp
 * in lines 44 and 45 add
 *    static std::chrono::high_resolution_clock::time_point a;
 *    static std::chrono::high_resolution_clock::time_point b;
 * in line 100 at the beginning of the first enqueue function
 *    a = std::chrono::high_resolution_clock::now();
 * in line 229 in the first line of the function handle_results
 *    b = std::chrono::high_resolution_clock::now();
 */
#ifdef ADDED_TIMEPOINTS_TO_CAF
extern high_resolution_clock::time_point a;
extern high_resolution_clock::time_point b;
#else
high_resolution_clock::time_point a;
high_resolution_clock::time_point b;
#endif

namespace {

using calc_atom = atom_constant<atom("calc")>;

class multiplier : public event_based_actor {
public:
  multiplier(actor_config& cfg, size_t matrix_size, actor worker)
    : event_based_actor(cfg),
      count_(0),
      size_(matrix_size),
      worker_(worker) {
    // nop
  }

  behavior make_behavior() override {
    return {
      [=] (calc_atom) {
        ++count_;
        vector<float> m1(size_ * size_);
        vector<float> m2(size_ * size_);
        iota(m1.begin(), m1.end(), 0);
        iota(m2.begin(), m2.end(), 0);
        start_ = high_resolution_clock::now();
        send(worker_, move(m1), move(m2));
      },
      [=] (const vector<float>&) {
        end_ = high_resolution_clock::now();
        auto total = (end_ - start_);
        auto time_in_opencl = (b - a);
        cout << duration_cast<microseconds>(total).count()
             << ", " << duration_cast<microseconds>(time_in_opencl).count()
             << ", " << duration_cast<microseconds>(total - time_in_opencl).count()
             << endl;
        quit();
      }
    };
  }

private:
  size_t count_;
  size_t size_;
  actor worker_;
  chrono::high_resolution_clock::time_point start_;
  chrono::high_resolution_clock::time_point end_;
};

class config : public actor_system_config {
public:
  string device_name = "GeForce GT 650M";
  size_t size = 0;
  size_t iterations = 1;
  config() {
    load<opencl::manager>();
    opt_group{custom_options_, "global"}
    .add(device_name, "device,d", "device for computation (GeForce GT 650M, "
                      ", but will take first available device if not found)")
    .add(size, "size,s", "set matrix size (must be > 0)")
    .add(iterations, "iterations,i", "set iterations (deault: 1)");
  }
};

} // namespace anonymous

void caf_main(actor_system& system, const config& cfg) {
#ifndef ADDED_TIMEPOINTS_TO_CAF
  cout << "This benchmark requires you to adjust the command class "
          "in CAF and remove the `#undef ADDED_TIMEPOINTS_TO_CAF` at "
          "the top of this file" << std::endl;
  static_cast<void>(system);
  static_cast<void>(cfg);
#else
  auto& mngr = system.opencl_manager();
  // get device named in config ...
  auto opt = mngr.find_device_if([&](const opencl::device_ptr dev) {
    if (cfg.device_name.empty())
      return true;
    return dev->name() == cfg.device_name;
  });
  // ... or first one available
  if (!opt)
    opt = mngr.find_device_if([&](const opencl::device_ptr) { return true; });
  if (!opt) {
    cerr << "No device found." << endl;
    return;
  }
  auto dev = *opt;
  auto prog = mngr.create_program(kernel_source, "", dev);
  auto worker = mngr.spawn(prog, kernel_name,
                           nd_range{dim_vec{cfg.size, cfg.size}},
                           in<float>{}, in<float>{}, out<float>{});
  auto mult = system.spawn<multiplier>(cfg.size, worker);
  anon_send(mult, calc_atom::value);
  system.await_all_actors_done();
#endif
}

CAF_MAIN();
