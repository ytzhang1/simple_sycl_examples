#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main(){
    constexpr int N = 15;
    auto R = range<1>( N );
    std::vector<float> v(N,10);
    for(int i = 0; i < N; i++)
        std::cout << v[i] << " ";
    std::cout<<'\n';
    
    queue q;
    std::cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Running on device: "
        << q.get_device().get_info<info::device::vendor>() << "\n";

    {
        buffer buf{v};
        try{
          q.submit(
            [&](handler& h) {
                auto a = buf.get_access<access::mode::write>(h);
                h.parallel_for(R, [=](id<1> i) {
                a[ i ] = -2; 
                });
            }
          );
        }catch(std::exception & e){
            std::cout<<"Exception sync.\n";
            std::cout<<e.what()<<'\n';
        }
    } // buffer destructed, data is copied back to host
    
    for(int i = 0; i < N; i++)
        std::cout << v[i] << " ";
    std::cout<<'\n';
}