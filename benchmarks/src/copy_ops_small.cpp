#include <vector>
#include <iomanip>
#include <numeric>
#include <iostream>

#include "caf/all.hpp"
#include "caf/opencl/spawn_cl.hpp"

#include "include/config.hpp"
#include "include/kernel.hpp"

using namespace std;
using namespace caf;
using namespace caf::opencl;

class copy_guy : public event_based_actor {
public:
  copy_guy(size_t iterations,
           size_t size,
           actor worker)
    : count_(0),
      iterations_(iterations),
      size_(size),
      worker_(worker)
  { }

  behavior make_behavior() override {
    return {
      on(atom("calc")) >> [=] {
        vector<float> m1(size_ * size_ * size_);
        iota(m1.begin(), m1.end(), 0);
        send(worker_, move(m1));
        ++count_;
      },
      on_arg_match >> [=] (const vector<float>& buffer) {
        if (count_ >= iterations_) {
#ifdef CL_ENABLE_DEBUG
          for (size_t pos = 0; pos < size_; ++pos) {
              cout << fixed << setprecision(2) << setw(9)
                   << buffer[pos];
          }
          cout << endl;
#else
          static_cast<void>(buffer);
#endif
          quit();
        } else {
          send(this, atom("calc"));
        }
      },
      others() >> [] {
         cout << "unknown message!" << endl;
      }
    };
  }

private:
  size_t count_;
  size_t iterations_;
  size_t size_;
  actor  worker_;
};


int main(int argc, char** argv) {
  size_t size = 0;
  size_t iterations  = 1;
  auto res = message_builder(argv + 1, argv + argc).extract_opts({
    {"size,s", "set matrix size (must be > 0)", size},
    {"iterations,i", "set iterations (deault: 1)", iterations}
  });
  if(! res.error.empty()) {
    cerr << res.error << endl;
    return 1;
  }
  if (res.opts.count("help") > 0 || size <= 0) {
    cout << res.helptext << endl;
    return 0;
  }
  announce<vector<float>>("vector_float");
  auto prog = program::create(kernel_source, 0);
  auto worker = spawn_cl(prog, kernel_name6,
                         spawn_config{dim_vec{size}},
                         in<float*>{}, out<float*>{});
  auto cpy = spawn<copy_guy>(iterations, size, worker);
  anon_send(cpy, atom("calc"));
  await_all_actors_done();
  shutdown();
}
