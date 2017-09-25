#ifndef UTIL_HPP
#define UTIL_HPP

#include <memory>
#include <string>
#include <algorithm>
#include <type_traits>

#include "caf/opencl/program.hpp"

#if defined __APPLE__ || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

std::string get_opencl_error(cl_int err);
void check_cl_error(cl_int err, const std::string& message);

/// Create program for a given device type (pick the first available).
/// Acceptable: cpu, gpu, accelerator - otherwise just choose the default one.
caf::opencl::program create_program(const std::string& dev_type,
                                    const char* source,
                                    const char* options = nullptr);

template<typename T, cl_int (*ref)(T), cl_int (*deref)(T)>
class smart_ptr {

    typedef typename std::remove_pointer<T>::type element_type;

    typedef element_type*       pointer;
    typedef element_type&       reference;
    typedef const element_type* const_pointer;
    typedef const element_type& const_reference;


 public:

    smart_ptr(pointer ptr = nullptr) : m_ptr(ptr) {
        if (m_ptr) ref(m_ptr);
    }

    ~smart_ptr() { reset(); }

    smart_ptr(const smart_ptr& other) : m_ptr(other.m_ptr) {
        if (m_ptr) ref(m_ptr);
    }

    smart_ptr(smart_ptr&& other) : m_ptr(other.m_ptr) {
        other.m_ptr = nullptr;
    }

    smart_ptr& operator=(pointer ptr) {
        reset(ptr);
        return *this;
    }

    smart_ptr& operator=(smart_ptr&& other) {
        std::swap(m_ptr, other.m_ptr);
        return *this;
    }

    smart_ptr& operator=(const smart_ptr& other) {
        smart_ptr tmp{other};
        std::swap(m_ptr, tmp.m_ptr);
        return *this;
    }

    inline void reset(pointer ptr = nullptr) {
        if (m_ptr) deref(m_ptr);
        m_ptr = ptr;
        if (ptr) ref(ptr);
    }

    // does not modify reference count of ptr
    inline void adopt(pointer ptr) {
        reset();
        m_ptr = ptr;
    }

    inline pointer get() const { return m_ptr; }

    inline pointer operator->() const { return m_ptr; }

    inline reference operator*() const { return *m_ptr; }

    inline bool operator!() const { return m_ptr == nullptr; }

    inline explicit operator bool() const { return m_ptr != nullptr; }

 private:

    pointer m_ptr;

};

inline cl_int clReleaseDeviceDummy (cl_device_id) { return 0; }
inline cl_int clRetainDeviceDummy  (cl_device_id) { return 0; }

typedef smart_ptr<cl_mem,clRetainMemObject,clReleaseMemObject> mem_ptr;
typedef smart_ptr<cl_event,clRetainEvent,clReleaseEvent>       event_ptr;
typedef smart_ptr<cl_kernel,clRetainKernel,clReleaseKernel>    kernel_ptr;
typedef smart_ptr<cl_context,clRetainContext,clReleaseContext> context_ptr;
typedef smart_ptr<cl_program,clRetainProgram,clReleaseProgram> program_ptr;
typedef smart_ptr<cl_device_id,clRetainDeviceDummy,clReleaseDeviceDummy>
        device_ptr;
typedef smart_ptr<cl_command_queue,clRetainCommandQueue,clReleaseCommandQueue>
        command_queue_ptr;

#endif // UTIL_HPP
