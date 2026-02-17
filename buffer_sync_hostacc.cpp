#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main(){
    constexpr int N = 10;
    auto R = range<1>( N );
    std::vector<float> v(N,10);

    for(int i = 0; i < N; i++)
        std::cout << v[i] << " ";
    std::cout<<'\n';
    
    queue q{ gpu_selector_v };
    std::cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Running on device: "
        << q.get_device().get_info<info::device::vendor>() << "\n";

    buffer buf{v};
    q.submit(
        [&](handler& h) {
            auto a = buf.get_access<access::mode::write>(h);
            h.parallel_for(R, [=](id<1> i) { a[ i ] = -2; } );
        }
    );

    //!!! data is copied back to host by invoking host accessor 
    auto bufr = buf.get_host_access();
    
    for(int i = 0; i < N; i++)
        std::cout << v[i] << " ";
    std::cout<<'\n';

}