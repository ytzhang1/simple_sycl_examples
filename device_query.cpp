#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main(){
    queue gpuq{ gpu_selector_v };
    
    std::cout << "device name: "
        << gpuq.get_device().get_info<info::device::name>() << "\n";
    std::cout << "gpu max work group size: "
        << gpuq.get_device().get_info<info::device::max_work_group_size>() << '\n';
    std::cout << "supports double precision ? " 
        << gpuq.get_device().has(aspect::fp64) << "\n";
}