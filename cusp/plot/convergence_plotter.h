/*
 *  Copyright 2008-2014 NVIDIA Corporation
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

/*! \file convergence_plotter.h
 *  \brief Javascript convergence generator
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>

#include <string>

namespace cusp
{
namespace plot
{

class convergence_plotter {

  typedef cusp::array1d<double,cusp::host_memory> HostArray;

  private:
    std::vector<std::string> names;
    std::vector<HostArray> residuals;

  public:
    template<typename Array>
    void append(const std::string& name, const Array& residual_history);

    void generate(const char* plot_name);
};

} // end namespace plot
} // end namespace cusp

#include <cusp/plot/detail/convergence_plotter.inl>
