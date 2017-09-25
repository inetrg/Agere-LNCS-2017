#include <chrono>
#include <vector>
#include <iomanip>
#include <numeric>
#include <iostream>

#include "caf/all.hpp"

#include "include/config.hpp"

using namespace std;
using namespace caf;

namespace {

using done_atom = atom_constant<atom("done")>;

class dummy : public event_based_actor {
public:
  dummy(actor_config& cfg) : event_based_actor(cfg) {
    // nop
  }
  behavior make_behavior() override {
    return {
      [=] (done_atom done) { return done; }
    };
  }
};

class config : public actor_system_config {
public:
  string device_name = "GeForce GT 650M";
  size_t size = 0;
  size_t iterations = 1;
  config() {
    opt_group{custom_options_, "global"}
    .add(device_name, "device,d", "Will be ignored. Just here for a unified interface.")
    .add(size, "size,s", "set matrix size (must be > 0)")
    .add(iterations, "iterations,i", "set iterations (deault: 1)");
  }
};

void caf_main(actor_system& system, const config& cfg) {
  auto start_ = chrono::high_resolution_clock::now();
  for(size_t i = 1; i < cfg.iterations; ++i)
      system.spawn<dummy, lazy_init>();
  auto last = system.spawn<dummy>();
  {
    scoped_actor self{system};
    self->send(last, done_atom::value);
    self->receive([] (done_atom) {
        // nop
    });
  }
  auto end_ = chrono::high_resolution_clock::now();
  cout << chrono::duration_cast<chrono::microseconds>((end_ - start_)).count()
       << endl;
  system.await_all_actors_done();
}

} // namespace anonymous

CAF_MAIN();
