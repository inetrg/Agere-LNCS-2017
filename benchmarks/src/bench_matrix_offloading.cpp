
#include <chrono>

#include "config.hpp"

#include "caf/all.hpp"
#include "caf/opencl/all.hpp"

using namespace std;
using namespace caf;
using namespace caf::opencl;

namespace {

constexpr const char* kernel_source = R"__(
  __kernel void mandelbrot(__global float* config,
                           __global int* output) {
    unsigned iterations = config[0];
    unsigned width = config[1];
    unsigned height = config[2];
    float min_re = config[3];
    float max_re = config[4];
    float min_im = config[5];
    float max_im = config[6];
    float re_factor = (max_re - min_re) / (width - 1);
    float im_factor = (max_im - min_im) / (height - 1);
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);
    float z_re = min_re + x * re_factor;
    float z_im = max_im - y * im_factor;
    float const_re = z_re;
    float const_im = z_im;
    unsigned cnt = 0;
    float cond = 0;
    do {
      float tmp_re = z_re;
      float tmp_im = z_im;
      z_re = ( tmp_re * tmp_re - tmp_im * tmp_im ) + const_re;
      z_im = ( 2 * tmp_re * tmp_im ) + const_im;
      cond = (z_re - z_im) * (z_re - z_im);
      ++cnt;
    } while (cnt < iterations && cond <= 4.0f);
    output[x+y*width] = cnt;
  }
)__";

#ifdef NDEBUG
#define DEBUG(x)
#else
#define DEBUG(x) cerr << x << endl;
#endif

} // namespace <anonymous>

using ack_atom = atom_constant<atom("ack")>;

// how much of the problem is offloaded to the OpenCL device
unsigned long with_opencl = 0;

// global values to track the time
chrono::system_clock::time_point cpu_start;
chrono::system_clock::time_point opencl_start;
chrono::system_clock::time_point total_start;
chrono::system_clock::time_point cpu_end;
chrono::system_clock::time_point opencl_end;
chrono::system_clock::time_point total_end;
unsigned long time_opencl = 0;
unsigned long time_cpu = 0;

// calculates mandelbrot with OpenCL
void mandel_cl(event_based_actor* self,
               const string& device_name,
               uint32_t iterations,
               uint32_t width,
               uint32_t height,
               float_type min_real,
               float_type max_real,
               float_type min_imag,
               float_type max_imag) {
  auto& mngr = self->system().opencl_manager();
  auto opt = mngr.find_device_if([&](const opencl::device_ptr dev) {
    if (device_name.empty())
      return true;
    return dev->name() == device_name;
  });
  if (!opt)
    throw std::runtime_error("No device called '" + device_name + "' found.");
  auto dev = *opt;
  auto prog = mngr.create_program(kernel_source, "", dev);
  auto unbox_args = [](message& msg) -> optional<message> {
    return msg;
  };
  auto box_res = [&] (vector<int> result) -> message {
    opencl_end = std::chrono::system_clock::now();
    return make_message(move(result));
  };
  vector<float_type> cljob {
    static_cast<float_type>(iterations),
    static_cast<float_type>(width),
    static_cast<float_type>(height),
    min_real, max_real,
    min_imag, max_imag
  };
  nd_range ndr{dim_vec{width, height}};
  opencl_start = chrono::system_clock::now();
  auto clworker = mngr.spawn(prog, "mandelbrot", ndr, unbox_args, box_res,
                             in<float_type>{}, out<int>{});
  self->request(clworker, infinite, move(cljob)).then (
    [=](const vector<int>& result) {
      static_cast<void>(result);
      DEBUG("Mandelbrot with OpenCL calculated");
    }
  );
}

template<typename T>
T get_cut(T start, T end, uint32_t percentage) {
  auto dist = (abs(start) + abs(end)) * percentage / 100.0;
  return (dist - abs(start));
}

template<typename T>
T get_bottom(T distance, uint32_t percentage) {
  return distance * percentage / 100;
}

template<typename T>
T get_top(T distance, uint32_t percentage) {
  return distance * (100 - percentage) / 100;
}

class config : public actor_system_config {
public:
  string device_name = "GeForce GT 650M";
  size_t iterations = default_iterations;
  uint32_t width = default_width;
  uint32_t height = default_height;
  uint32_t offloaded = 0;
  config() {
    load<opencl::manager>();
    opt_group{custom_options_, "global"}
    .add(device_name, "device,d", "device for computation (GeForce GT 650M, "
                      ", but will take first available device if not found)")
    .add(width, "width,W", "set width (16000)")
    .add(height, "height,H", "set height (16000")
    .add(iterations, "iterations,i", "set iterations (deault: 500)")
    .add(offloaded,"with-opencl,o", "part calculated with OpenCL in % (0)");
  }
};

void caf_main(actor_system& system, const config& cfg) {
  total_start = chrono::system_clock::now();
  with_opencl = cfg.offloaded;
  auto iterations = cfg.iterations;
  auto on_cpu  = 100 - cfg.offloaded;
  auto min_re  = default_min_real;
  auto max_re  = default_max_real;
  auto min_im  = default_min_imag;
  auto max_im  = default_max_imag;

  auto scale = [&](const float_type ratio) {
    float_type abs_re = fabs(max_re + (-1 * min_re)) / 2;
    float_type abs_im = fabs(max_im + (-1 * min_im)) / 2;
    float_type mid_re = min_re + abs_re;
    float_type mid_im = min_im + abs_im;
    auto dist = abs_re * ratio;
    min_re = mid_re - dist;
    max_re = mid_re + dist;
    min_im = mid_im - dist;
    max_im = mid_im + dist;
  };
  scale(default_scaling);

  auto cpu_width  = get_bottom(cfg.width, on_cpu);
  auto cpu_height = cfg.height;
  auto cpu_min_re = min_re;
  auto cpu_max_re = get_cut(min_re, max_re, on_cpu);
  auto cpu_min_im = min_im;
  auto cpu_max_im = max_im;
  DEBUG("[CPU] width: " << cpu_width
        << "(" << cpu_min_re << " to " << cpu_max_re << ")");

  auto opencl_width  = get_top(cfg.width, on_cpu);
  auto opencl_height = cfg.height;
  auto opencl_min_re = get_cut(min_re, max_re, on_cpu);
  auto opencl_max_re = max_re;
  auto opencl_min_im = min_im;
  auto opencl_max_im = max_im;
  DEBUG("[OpenCL] width: " << opencl_width
        << "(" << opencl_min_re << " to " << opencl_max_re << ")");

  if (opencl_width > 0) {
    // trigger calculation with OpenCL
    system.spawn(mandel_cl, cfg.device_name, iterations, opencl_width, opencl_height,
                 opencl_min_re, opencl_max_re, opencl_min_im, opencl_max_im);
  }

  cpu_start = chrono::system_clock::now();
  if (cpu_width > 0) {
    scoped_actor cnt{system};
    // trigger calculation on the CPU
    vector<int> image(cpu_width * cpu_height);
    auto re_factor = (cpu_max_re - cpu_min_re) / (cpu_width - 1);
    auto im_factor = (cpu_max_im - cpu_min_im) / (cpu_height - 1);
    int* indirection = image.data();
    for (uint32_t im = 0; im < cpu_height; ++im) {
      system.spawn([&cnt, indirection, cpu_width, cpu_min_re, cpu_max_re,
                    cpu_min_im, cpu_max_im, re_factor, im_factor, im,
                    iterations] (event_based_actor* self) {
        for (uint32_t re = 0; re < cpu_width; ++re) {
          auto z_re = cpu_min_re + re * re_factor;
          auto z_im = cpu_max_im - im * im_factor;
          auto const_re = z_re;
          auto const_im = z_im;
          uint32_t cnt = 0;
          float_type cond = 0;
          do {
            auto tmp_re = z_re;
            auto tmp_im = z_im;
            z_re = (tmp_re * tmp_re - tmp_im * tmp_im) + const_re;
            z_im = (2 * tmp_re * tmp_im) + const_im;
            cond = z_re * z_re + z_im * z_im;
            ++cnt;
          } while (cnt < iterations && cond <= 4.0f);
          indirection[re + im * cpu_width] = cnt;
        }
        self->send(cnt, ack_atom::value);
      });
    }
    unsigned i = 0;
    cnt->receive_for(i, cpu_height)( [](ack_atom) { /* nop */ } );
    // await_all_actors_done();
    cpu_end = chrono::system_clock::now();
    DEBUG("Mandelbrot on CPU calculated");
  }

  system.await_all_actors_done();
  total_end = chrono::system_clock::now();
  if (cpu_width > 0) {
    time_cpu = chrono::duration_cast<chrono::milliseconds>(
      cpu_end - cpu_start
    ).count();
  }
  if (opencl_width > 0) {
    time_opencl = chrono::duration_cast<chrono::milliseconds>(
      opencl_end - opencl_start
    ).count();
  }
  auto time_total = chrono::duration_cast<chrono::milliseconds>(
    total_end - total_start
  ).count();
  cout << with_opencl
       << ", " << time_total
       << ", " << time_cpu
       << ", " << time_opencl
       << endl;
  return;
}

CAF_MAIN();
