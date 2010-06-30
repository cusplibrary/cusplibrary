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

namespace cusp
{

// TODO rename krylov_m -> krylov_m
namespace krylov_m
{

// TODO move these to the .inl and put inside cusp::krylov::detail namespace
namespace trans_m
{
  template <typename InputIterator1, typename InputIterator2,
            typename InputIterator3,
	    typename OutputIterator1,
	    typename ScalarType>
  void compute_z_m(InputIterator1 z_0_s_b, InputIterator1 z_0_s_e,
		InputIterator2 z_m1_s_b, InputIterator3 sig_b,
		OutputIterator1 z_1_s_b,
		ScalarType beta_m1, ScalarType beta_0, ScalarType alpha_0);

  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator1,
	    typename ScalarType>
  void compute_b_m(InputIterator1 z_1_s_b, InputIterator1 z_1_s_e,
		InputIterator2 z_0_s_b, OutputIterator1 beta_0_s_b,
		ScalarType beta_0);
  template <typename Array1, typename Array2, typename Array3,
            typename Array4, typename ScalarType>
  void compute_z_m(const Array1& z_0_s, const Array2& z_m1_s,
		const Array3& sig, Array4& z_1_s,
		ScalarType beta_m1, ScalarType beta_0, ScalarType alpha_0);

  template <typename Array1, typename Array2, typename Array3,
            typename ScalarType>
  void compute_b_m(const Array1& z_1_s, const Array2& z_0_s,
		Array3& beta_0_s, ScalarType beta_0);

  template <typename InputIterator1, typename InputIterator2,
            typename InputIterator3, typename OutputIterator,
            typename ScalarType>
  void compute_a_m(InputIterator1 z_0_s_b, InputIterator1 z_0_s_e,
		InputIterator2 z_1_s_b, InputIterator3 beta_0_s_b,
                OutputIterator alpha_0_s_b,
		ScalarType beta_0, ScalarType alpha_0);

  template <typename Array1, typename Array2, typename Array3,
            typename Array4, typename ScalarType>
  void compute_a_m(const Array1& z_0_s, const Array2& z_1_s,
                const Array3& beta_0_s, Array4& alpha_0_s,
		ScalarType beta_0, ScalarType alpha_0);

  template <typename Array1, typename Array2, typename Array3,
            typename Array4, typename Array5, typename Array6>
  void compute_xp_m(const Array1& alpha_0_s, const Array2& z_1_s,
                const Array3& beta_0_s, const Array4& r_0,
                Array5& x_0_s, Array6& p_0_s);

  template <typename Array1, typename Array2>
    void vectorize_copy(const Array1& source, Array2& dest);

} // end namespace trans_m

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup krylov_methods Krylov Methods
 *  \ingroup iterative_solvers
 *  \{
 */

/*! \p cg_m 
 * \TODO DOCUMENT
 *
 */
template <class LinearOperator,
          class VectorType1,
          class VectorType2,
          class VectorType3>
void cg_m(LinearOperator& A,
          VectorType1& x,
          VectorType2& b,
          VectorType3& sigma);

/*! \p cg_m 
 * \TODO DOCUMENT
 *
 */
template <class LinearOperator,
          class VectorType1,
          class VectorType2,
          class VectorType3,
          class Monitor>
void cg_m(LinearOperator& A,
        VectorType1& x, VectorType2& b, VectorType3& sigma,
        Monitor& monitor);
/*! \}
 */

} // end namespace krylov_m
} // end namespace cusp

#include <cusp/krylov/detail/cg-m.inl>

