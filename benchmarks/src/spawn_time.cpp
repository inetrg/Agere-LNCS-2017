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
using namespace caf;
using namespace caf::opencl;

namespace {

class config : public actor_system_config {
public:
  string device_name = "GeForce GT 650M";
  size_t size = 0;
  size_t iterations = 1;
  //  announce<vector<float>>("vector_float");
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
  auto start_ = chrono::high_resolution_clock::now();
  auto prog = mngr.create_program(kernel_source, "", dev);
  auto ndr = nd_range{dim_vec{cfg.size}};
  for(size_t i = 1; i < cfg.iterations; ++i)
    mngr.spawn(prog, kernel_name6, ndr, in<float>{}, out<float>{});
  auto last = mngr.spawn(prog, kernel_name6, ndr, in<float>{}, out<float>{});
  vector<float> m1(cfg.size);
  iota(m1.begin(), m1.end(), 0);
  {
    scoped_actor self{system};
    self->send(last, m1);
    self->receive(
      [&] (const vector<float>& matrix) {
#ifdef CL_ENABLE_DEBUG
        for (size_t pos = 0; pos < size; ++pos)
            cout << fixed << setprecision(2) << setw(9) << matrix[pos];
        cout << endl;
#else
        static_cast<void>(matrix);
#endif
      }
    );
  }
  auto end_ = chrono::high_resolution_clock::now();
  cout << chrono::duration_cast<chrono::microseconds>((end_ - start_)).count()
       << endl;
  system.await_all_actors_done();
}

} // namespace anonymous

CAF_MAIN();
