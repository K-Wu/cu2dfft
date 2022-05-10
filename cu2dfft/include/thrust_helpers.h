#pragma once
#include "cu2dfft.h"

struct is_close_float2
{
    __host__ __device__ bool operator()(float2 x, float2 y) const
    {
        //return fabs(x.x - y.x) < 1e-6 && fabs(x.y - y.y) < 1e-6;
        return fabs(x.x - y.x) < fabs(x.x)*0.001 && fabs(x.y - y.y) < fabs(x.y)*0.001;
    }
};

std::ostream &operator <<(std::ostream &o, const cufftComplex &buffer)
{
    
    return o <<"("<< buffer.x<< ","<<buffer.y <<")";
}

template <typename Iterator>
void print_range(const std::string &name, Iterator first, Iterator last)
{
    // from thrust example
    typedef typename std::iterator_traits<Iterator>::value_type T;

    std::cout << name << ": (" << std::distance(first, last) << ")";
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}