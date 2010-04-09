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

#include <cusp/detail/config.h>

#include <cusp/blas.h>

#include <limits>
#include <iostream>
#include <iomanip>

// Classes to monitor iterative solver progress, check for convergence, etc.
// Follows the implementation of Iteration in the ITL:
//   http://www.osl.iu.edu/research/itl/doc/Iteration.html

namespace cusp
{

template <typename ValueType>
class default_monitor
{
    public:

    template <typename Vector>
    default_monitor(const Vector& b, size_t iteration_limit = 500, ValueType relative_tolerance = 1e-5)
        : b_norm(cusp::blas::nrm2(b)),
          r_norm(std::numeric_limits<ValueType>::max()),
          iteration_limit_(iteration_limit),
          iteration_count_(0),
          relative_tolerance_(relative_tolerance)
    {}

    void operator++(void) {  ++iteration_count_; } // prefix
    //void operator++(int)  {  iteration++; } // postfix

    template <typename Vector>
    bool finished(const Vector& r)
    {
        r_norm = cusp::blas::nrm2(r);
        
        return converged() || iteration_count() >= iteration_limit();
    }
    
    bool converged() const
    {
        if (b_norm == 0)
            return r_norm < relative_tolerance();
        else
            return r_norm < relative_tolerance() * b_norm;
    }

    size_t iteration_count() const { return iteration_count_; }
    size_t iteration_limit() const { return iteration_limit_; }

    ValueType relative_tolerance() const { return relative_tolerance_; }

    protected:
    
    ValueType r_norm;
    ValueType b_norm;
    ValueType relative_tolerance_;

    size_t iteration_limit_;
    size_t iteration_count_;
};


template <typename ValueType=double>
class verbose_monitor : public default_monitor<ValueType>
{
    typedef cusp::default_monitor<ValueType> super;

    public:

    template <typename Vector>
    verbose_monitor(const Vector& b, size_t iteration_limit = 500, ValueType relative_tolerance = 1e-5)
        : super(b, iteration_limit, relative_tolerance)
    {
        std::cout << "Solver will continue until ";
        std::cout << "residual " << super::b_norm * super::relative_tolerance() << " or reaching ";
        std::cout << super::iteration_limit() << " iterations " << std::endl;
        std::cout << "  Iteration Number  | Residual Norm" << std::endl;
    }
    
    template <typename Vector>
    bool finished(const Vector& r)
    {
        super::r_norm = cusp::blas::nrm2(r);

        std::cout << "       "  << std::setw(10) << super::iteration_count();
        std::cout << "       "  << std::setw(10) << std::scientific << super::r_norm << std::endl;

        if (super::converged())
        {
            std::cout << "Successfully converged after " << super::iteration_count() << " iterations." << std::endl;
            return true;
        }
        else if (super::iteration_count() >= super::iteration_limit())
        {
            std::cout << "Failed to converge after " << super::iteration_count() << " iterations." << std::endl;
            return true;
        }
        else
        {
            return false;
        }
    }
};

} // end namespace cusp

