/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <cusp/vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>

namespace cusp
{

namespace detail
{
    // square<T> computes the square of a number f(x) -> x*x
    template <typename T>
        struct square
        {
            __host__ __device__
                T operator()(const T& x) const { 
                    return x * x;
                }
        };
    
    template <typename T>
        struct scale
        {
            const T a;

            scale(const T& _a) : a(_a) {}

            __host__ __device__
                T operator()(const T& x) const { 
                    return a * x;
                }
        };

    template <typename T>
        struct scale_and_add
        {
            const T a;

            scale_and_add(const T& _a) : a(_a) {}

            __host__ __device__
                T operator()(const T& x, const T& y) const { 
                    return a * x + y;
                }
        };


    // dispatch different host/device pointer wrappers
    template <class MemorySpace>
        struct pointer_wrapper {};

    template <>
        struct pointer_wrapper<cusp::host> 
        {
            template <typename ValueType>
                static ValueType * wrap_ptr(ValueType * ptr) { return ptr; }
        };
    
    template <>
        struct pointer_wrapper<cusp::device> 
        {
            template <typename ValueType>
                static thrust::device_ptr<ValueType> wrap_ptr(ValueType * ptr) { return thrust::device_ptr<ValueType>(ptr); }
        };
} // end namespace detail


template <class MemorySpace>
struct blas : public detail::pointer_wrapper<MemorySpace>
{
    template <typename ValueType>
        static void axpy(const size_t n, const ValueType alpha, const ValueType * x, ValueType * y) {
            thrust::transform(wrap_ptr(x), wrap_ptr(x + n), wrap_ptr(y), wrap_ptr(y), detail::scale_and_add<ValueType>(alpha));
        }
    
    template <typename ValueType>
        static void copy(const size_t n, const ValueType * x, ValueType * y) {
            thrust::copy(wrap_ptr(x), wrap_ptr(x + n), wrap_ptr(y));
        }
    
    template <typename ValueType>
        static ValueType dot(const size_t n, const ValueType * x, const ValueType * y) {
            const ValueType init = 0;
            return thrust::inner_product(wrap_ptr(x), wrap_ptr(x + n), wrap_ptr(y), init);
        }
    
    template <typename ValueType>
        static void fill(const size_t n, const ValueType alpha, ValueType * x) {
            thrust::fill(wrap_ptr(x), wrap_ptr(x + n), alpha);
        }
    
    template <typename ValueType>
        static ValueType nrm2(const size_t n, const ValueType * x) {
            detail::square<ValueType> unary_op;
            thrust::plus<ValueType>  binary_op;
            const ValueType init = 0;
            return std::sqrt( thrust::transform_reduce(wrap_ptr(x), wrap_ptr(x + n), unary_op, init, binary_op) );
        }
    
    template <typename ValueType>
        static void scal(const size_t n, const ValueType alpha, ValueType * x) {
            thrust::transform(wrap_ptr(x), wrap_ptr(x + n), wrap_ptr(x), detail::scale<ValueType>(alpha));
        }
}; //end blas


} // end namespace cusp
