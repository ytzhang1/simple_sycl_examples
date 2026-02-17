#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main()
{
    sycl::range<1> r{0};
    sycl::buffer<int> b{r};
    std::cout<<"b size: "<<b.size()<<'\n';

    int * ptr = nullptr;
    sycl::range<1> r1{0};
    sycl::buffer<int> b1{ptr, r1};
    std::cout<<"b1 size: "<<b1.size()<<'\n';
    
    sycl::queue q;
    std::vector<int> v(0); //0 sized vector
    {
        sycl::buffer b2{v};
        std::cout<<"b2 size: "<<b2.size()<<'\n';
        
        /*
        q.submit([&](sycl::handler& cgh){
           auto acc = b2.get_access(cgh);
           cgh.single_task(
               [=](){acc[0]=1;}  //This will trigger segmentation fault on device.
           );
        }).wait();
        
        {
           auto acc = b2.get_host_access();
           std::cout<<acc[0]<<"\n";
        }
        */
    }
    
    v.resize(1);
    v[0]=1;
    {
        sycl::buffer b3{v};
        std::cout<<"b3 size: "<<b3.size()<<'\n';
    
        q.submit([&](sycl::handler& cgh){
           auto acc = b3.get_access(cgh);
           cgh.single_task(
               [=](){acc[0]=2;}
           );
        });
        {
            auto acc = b3.get_host_access();
            std::cout<<acc[0]<<"\n";
        }
    }
}
