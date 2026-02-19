#include <sycl/sycl.hpp> 
#include <sycl/ext/oneapi/bfloat16.hpp> 
    
int main() { 
    using bf16 = sycl::ext::oneapi::bfloat16;
    const size_t N = 16;

    // Host data in bf16
    std::vector<bf16> hA(N), hB(N), hC(N);
    for (size_t i = 0; i < N; ++i) {
        float a = 0.001f * static_cast<float>(i);
        float b = 1.0f + 0.0005f * static_cast<float>(i);
        hA[i] = bf16(a);
        hB[i] = bf16(b);
        float sum = static_cast<float>(hA[i]) + static_cast<float>(hB[i]);
        hC[i] = bf16(sum);
        std::cout << i << ": " << a << " + " << b << " = " << sum << "\n";
        
        a = static_cast<float>(hA[i]);
        b = static_cast<float>(hB[i]);
        sum = static_cast<float>(hC[i]);
        std::cout <<" : " << a << " + " << b << " = " << sum << "\n";
    }
    return 0;
}
