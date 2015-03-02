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

#include <cusp/multiply.h>
#include <cusp/monitor.h>
#include <cusp/blas/blas.h>

namespace cusp
{

template <typename MatrixType, typename SmootherType, typename SolverType>
template <typename MatrixType2, typename SmootherType2, typename SolverType2>
multilevel<MatrixType,SmootherType,SolverType>
::multilevel(const multilevel<MatrixType2,SmootherType2,SolverType2>& M)
    : solver(M.solver)
{
    for( size_t lvl = 0; lvl < M.levels.size(); lvl++ )
      levels.push_back(M.levels[lvl]);

    levels[0].A = *M.A_ptr;
    A_ptr = &levels[0].A;
}

template <typename MatrixType, typename SmootherType, typename SolverType>
template <typename Array1, typename Array2>
void multilevel<MatrixType,SmootherType,SolverType>
::operator()(const Array1& b, Array2& x)
{
    // perform 1 V-cycle
    _solve(b, x, 0);
}

template <typename MatrixType, typename SmootherType, typename SolverType>
template <typename Array1, typename Array2>
void multilevel<MatrixType,SmootherType,SolverType>
::solve(const Array1& b, Array2& x)
{
    cusp::monitor<ValueType> monitor(b);

    solve(b, x, monitor);
}

template <typename MatrixType, typename SmootherType, typename SolverType>
template <typename Array1, typename Array2, typename Monitor>
void multilevel<MatrixType,SmootherType,SolverType>
::solve(const Array1& b, Array2& x, Monitor& monitor)
{
    const size_t n = A_ptr->num_rows;

    // use simple iteration
    cusp::array1d<ValueType,MemorySpace> update(n);
    cusp::array1d<ValueType,MemorySpace> residual(n);

    // compute initial residual
    cusp::multiply(*A_ptr, x, residual);
    cusp::blas::axpby(b, residual, residual, ValueType(1.0), ValueType(-1.0));

    while(!monitor.finished(residual))
    {
        _solve(residual, update, 0);

        // x += M * r
        cusp::blas::axpy(update, x, ValueType(1.0));

        // update residual
        cusp::multiply(*A_ptr, x, residual);
        cusp::blas::axpby(b, residual, residual, ValueType(1.0), ValueType(-1.0));
        ++monitor;
    }
}

template <typename MatrixType, typename SmootherType, typename SolverType>
template <typename Array1, typename Array2>
void multilevel<MatrixType,SmootherType,SolverType>
::_solve(const Array1& b, Array2& x, const size_t i)
{
    if (i + 1 == levels.size())
    {
        // coarse grid solve
        // TODO streamline
        cusp::array1d<ValueType,cusp::host_memory> temp_b(b);
        cusp::array1d<ValueType,cusp::host_memory> temp_x(x.size());
        solver(temp_b, temp_x);
        x = temp_x;
    }
    else
    {
        // initialize solution
        cusp::blas::fill(x, ValueType(0));

       // presmooth
        if(i == 0)
          levels[i].smoother.presmooth(*A_ptr, b, x);
        else
          levels[i].smoother.presmooth(levels[i].A, b, x);

        // compute residual <- b - A*x
        if(i == 0)
          cusp::multiply(*A_ptr, x, levels[i].residual);
        else
          cusp::multiply(levels[i].A, x, levels[i].residual);

        cusp::blas::axpby(b, levels[i].residual, levels[i].residual, ValueType(1.0), ValueType(-1.0));

        // restrict to coarse grid
        cusp::multiply(levels[i].R, levels[i].residual, levels[i + 1].b);

        // compute coarse grid solution
        _solve(levels[i + 1].b, levels[i + 1].x, i + 1);

        // apply coarse grid correction
        cusp::multiply(levels[i].P, levels[i + 1].x, levels[i].residual);
        cusp::blas::axpy(levels[i].residual, x, ValueType(1.0));

        // postsmooth
        if(i == 0)
          levels[i].smoother.postsmooth(*A_ptr, b, x);
        else
          levels[i].smoother.postsmooth(levels[i].A, b, x);
    }
}

template <typename MatrixType, typename SmootherType, typename SolverType>
void multilevel<MatrixType,SmootherType,SolverType>
::print( void )
{
    size_t num_levels = levels.size();
    double nnz = A_ptr->num_entries;

    std::cout << "\tNumber of Levels:\t" << num_levels << std::endl;
    std::cout << "\tOperator Complexity:\t" << operator_complexity() << std::endl;
    std::cout << "\tGrid Complexity:\t" << grid_complexity() << std::endl;
    std::cout << "\tlevel\tunknowns\tnonzeros:\t" << std::endl;

    for(size_t index = 1; index < num_levels; index++)
        nnz += levels[index].A.num_entries;

    double percent = A_ptr->num_entries / nnz;
    std::cout << "\t" << 0 << "\t" << A_ptr->num_cols << "\t\t" \
              << A_ptr->num_entries << " \t[" << 100*percent << "%]" \
              << std::endl;

    for(size_t index = 1; index < num_levels; index++)
    {
        percent = levels[index].A.num_entries / nnz;
        std::cout << "\t" << index << "\t" << levels[index].A.num_cols << "\t\t" \
                  << levels[index].A.num_entries << " \t[" << 100*percent << "%]" \
                  << std::endl;
    }
}

template <typename MatrixType, typename SmootherType, typename SolverType>
double multilevel<MatrixType,SmootherType,SolverType>
::operator_complexity( void )
{
    size_t nnz = A_ptr->num_entries;

    for(size_t index = 1; index < levels.size(); index++)
        nnz += levels[index].A.num_entries;

    return (double) nnz / (double) A_ptr->num_entries;
}

template <typename MatrixType, typename SmootherType, typename SolverType>
double multilevel<MatrixType,SmootherType,SolverType>
::grid_complexity( void )
{
    size_t unknowns = A_ptr->num_rows;

    for(size_t index = 1; index < levels.size(); index++)
        unknowns += levels[index].A.num_rows;

    return (double) unknowns / (double) A_ptr->num_rows;
}

} // end namespace cusp

