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

/*! \file convergence_plot.h
 *  \brief Javascript convergence generator
 */

#include <cusp/detail/config.h>

#include <cusp/array1d.h>

#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <vector>

#include <cusp/plot/detail/convergence_stub.h>

namespace cusp
{
namespace plot
{

template<typename Array>
void convergence_plotter
::append(const std::string& name, const Array& residual_history) {
    names.push_back(name);
    residuals.push_back(residual_history);
}

void convergence_plotter
::generate(const char* plot_name) {
    std::string data_tag("//data");

    std::string plot_str(detail::convergence_stub);
    size_t index = plot_str.find(data_tag) + data_tag.size();

    std::stringstream ss;
    ss << std::endl << "var names = [";
    for(size_t i = 0; i < names.size(); i++)
        ss << "'" + names[i] + "',";
    ss << "];" << std::endl;

    ss << std::endl << std::scientific << std::setw(4) << "var res = [";
    for(size_t i = 0; i < residuals.size(); i++) {
        ss << "[";
        thrust::copy(residuals[i].begin(), residuals[i].end(), std::ostream_iterator<double>(ss, ","));
        ss << "],";
    }
    ss << "];" << std::endl;

    plot_str.insert(index, ss.str());
    std::ofstream plot(plot_name);
    plot.write(plot_str.c_str(), plot_str.length());
    plot.close();
}

} // end namespace plot
} // end namespace cusp
