#include <chrono>
#include <vector>
#include <iomanip>
#include <numeric>
#include <iostream>

#include "util.hpp"

#include "caf/all.hpp"
#include "caf/opencl/all.hpp"

#include "include/config.hpp"
#include "include/kernel.hpp"

using namespace std;
using namespace caf;
using namespace caf::opencl;

namespace {

using calc_atom = atom_constant<atom("calc")>;

class multiplier : public event_based_actor {
public:
  multiplier(actor_config& cfg,
             size_t iterations,
             size_t matrix_size,
             actor worker)
    : event_based_actor(cfg),
      count_(0),
      iterations_(iterations),
      size_(matrix_size),
      worker_(worker)
  { }

  behavior make_behavior() override {
    return {
      [=] (calc_atom) {
        vector<float> m1(size_ * size_);
        vector<float> m2(size_ * size_);
        iota(m1.begin(), m1.end(), 0);
        iota(m2.begin(), m2.end(), 0);
        send(worker_, move(m1), move(m2));
        ++count_;
      },
      [=] (const vector<float>& matrix) {
        if (count_ >= iterations_) {
#ifdef CL_ENABLE_DEBUG
          for (size_t column = 0; column < size_; ++column) {
            for (size_t row = 0; row < size_; ++row) {
              cout << fixed << setprecision(2) << setw(9)
                   << matrix[row + column * size_];
            }
            cout << endl;
          }
#else
          static_cast<void>(matrix);
#endif
          quit();
        } else {
          send(this, calc_atom::value);
        }
      }
    };
  }

private:
  size_t count_;
  size_t iterations_;
  size_t size_;
  actor worker_;
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

void caf_main(actor_system& system, const config& cfg) {
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
  {
    auto start_ = chrono::high_resolution_clock::now();
    auto worker = mngr.spawn(prog, kernel_name,
                             nd_range{dim_vec{cfg.size, cfg.size}},
                             in<float>{}, in<float>{}, out<float>{});
    auto mult = system.spawn<multiplier>(cfg.iterations, cfg.size, worker);
    anon_send(mult, calc_atom::value);
    system.await_all_actors_done();
    auto end_ = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>((end_ - start_)).count()
         << endl;
  }
}

} // namespace anonymous

CAF_MAIN();
