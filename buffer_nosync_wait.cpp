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

    //!!! wait doesn't copy data back to host!
    q.wait();
    
    for(int i = 0; i < N; i++)
        std::cout << v[i] << " ";
    std::cout<<'\n';

}