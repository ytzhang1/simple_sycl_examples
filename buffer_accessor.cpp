#include <sycl/sycl.hpp>
#include <sycl/accessor.hpp>
#include <iostream>

using namespace sycl;

int main(){
    constexpr int N = 15;
    auto R = range<1>( N );
    
    queue q{gpu_selector_v};
    std::cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Running on device: "
        << q.get_device().get_info<info::device::vendor>() << "\n";

    buffer<int> buf{R};
    //auto acc = buf.get_access<access::mode::read_write>(); //the program hangs
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
    
    {
        //has to put acc1 in a scope, so that it's destructed at the end of scope.
        //Otherwise, it'll keep holding the data.
        auto acc1 = buf.get_host_access();
        for(int i = 0; i < N; i++)
            std::cout << acc1[i] << " ";
        std::cout<<'\n';
    }
    
    try{
      q.submit(
        [&](handler& h) {
            auto a = buf.get_access<access::mode::write>(h);
            h.parallel_for(R, [=](id<1> i) {
            a[ i ] = 2; 
            });
        }
      );
    }catch(std::exception & e){
        std::cout<<"Exception sync.\n";
        std::cout<<e.what()<<'\n';
    }
    
    {
        auto acc2 = buf.get_host_access();
        for(int i = 0; i < N; i++)
            std::cout << acc2[i] << " ";
        std::cout<<'\n';
    }
}