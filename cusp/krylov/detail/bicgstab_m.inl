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

#include <cusp/array1d.h>
#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/monitor.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>

#include <thrust/iterator/transform_iterator.h>


/*
 * The point of these routines is to solve systems of the type
 *
 * (A+\sigma Id)x = b
 *
 * for a number of different \sigma, iteratively, for sparse A, without
 * additional matrix-vector multiplication.
 *
 * The idea comes from arXiv:hep-lat/9612014
 *
 */

// put everything in cusp
namespace cusp
{

// this namespace contains things that are like cusp::krylov
// different name chosen to avoid the possibility of collisions
namespace krylov
{

// structs in this namespace do things that are somewhat blas-like, but
// are not usual blas operations (e.g. they aren't all linear in all arguments)
//
// except for KERNEL_VCOPY all of these structs perform operations that
// are specific to CG-M
namespace detail_m
{
  // computes new \zeta
  template <typename ScalarType>
    struct KERNEL_Z
  {
    ScalarType beta_m1;
    ScalarType beta_0;
    ScalarType alpha_0;

    KERNEL_Z(ScalarType _beta_m1, ScalarType _beta_0, ScalarType _alpha_0)
      : beta_m1(_beta_m1), beta_0(_beta_0), alpha_0(_alpha_0)
    {}

    template <typename Tuple>
    __host__ __device__
      void operator()(Tuple t)
    {
      // compute \zeta_1^\sigma
      thrust::get<0>(t)=thrust::get<1>(t)*thrust::get<2>(t)*beta_m1/
                        (beta_0*alpha_0*(thrust::get<2>(t)-thrust::get<1>(t))
                         +beta_m1*thrust::get<2>(t)*(ScalarType(1)-
                                                     beta_0*thrust::get<3>(t)));
    }
  };

  // computes new \beta
  template <typename ScalarType>
    struct KERNEL_B
  {
    ScalarType beta_0;

    KERNEL_B(ScalarType _beta_0) : beta_0(_beta_0)
    {}

    template <typename Tuple>
    __host__ __device__
      void operator()(Tuple t)
    {
      // compute \beta_0^\sigma
      thrust::get<0>(t)=beta_0*thrust::get<1>(t)/thrust::get<2>(t);
      //thrust::get<0>(t)=thrust::get<1>(t);
    }
  };

  // computes new alpha
  template <typename ScalarType>
    struct KERNEL_A
  {
    ScalarType beta_0;
    ScalarType alpha_0;

    // note: only the ratio alpha_0/beta_0 enters in the computation, it might
    // be better just to pass this ratio
    KERNEL_A(ScalarType _beta_0, ScalarType _alpha_0)
      : beta_0(_beta_0), alpha_0(_alpha_0)
    {}

    template <typename Tuple>
    __host__ __device__
      void operator()(Tuple t)
    {
      // compute \alpha_0^\sigma
      thrust::get<0>(t)=alpha_0/beta_0*thrust::get<2>(t)*thrust::get<3>(t)/
                        thrust::get<1>(t);
    }
  };

  //computes new w_1
  template <typename ScalarType>
    struct KERNEL_W
  {
    ScalarType beta_0;
    KERNEL_W(const ScalarType _beta_0) : beta_0(_beta_0) {}

    template <typename Tuple>
    __host__ __device__
      void operator()(Tuple t)
    {
      thrust::get<0>(t)=thrust::get<1>(t)+beta_0*thrust::get<2>(t);
    }
  };

  //computes new s_0
  template <typename ScalarType>
    struct KERNEL_S
  {
    ScalarType alpha_1;
    ScalarType chi_0;
    KERNEL_S(ScalarType _alpha_1, ScalarType _chi_0) :
	    alpha_1(_alpha_1), chi_0(_chi_0)
    {}

    template <typename Tuple>
    __host__ __device__
      void operator()(Tuple t)
    {
      thrust::get<0>(t)=thrust::get<1>(t)
	                +alpha_1*(thrust::get<0>(t)-chi_0*thrust::get<2>(t));
    }
  };

  //computes new chi_0^s
  template <typename ScalarType>
    struct KERNEL_CHI
  {
    ScalarType chi_0;
    KERNEL_CHI(ScalarType _chi_0) : chi_0(_chi_0)
    {}

    template <typename Tuple>
    __host__ __device__
      void operator()(Tuple t)
    {
      thrust::get<0>(t)=chi_0/(ScalarType(1.0)+chi_0*thrust::get<1>(t));
    }
  };

  //computes new rho_1^s
  template <typename ScalarType>
    struct KERNEL_RHO
  {
    ScalarType chi_0;
    KERNEL_RHO(ScalarType _chi_0) : chi_0(_chi_0)
    {}

    template <typename Tuple>
    __host__ __device__
      void operator()(Tuple t)
    {
      thrust::get<0>(t)=thrust::get<1>(t)/
	                (ScalarType(1.0)+chi_0*thrust::get<2>(t));
			//(ScalarType(1.0)+chi_0*thrust::get<2>(t));
    }
  };

  // computes new s
  template <typename ScalarType>
    struct KERNEL_SS : thrust::binary_function<int, ScalarType, ScalarType>
  {
    int N;
    const ScalarType *raw_ptr_beta_0_s;
    const ScalarType *raw_ptr_chi_0_s;
    const ScalarType *raw_ptr_rho_0_s;
    const ScalarType *raw_ptr_zeta_0_s;
    const ScalarType *raw_ptr_alpha_1_s;
    const ScalarType *raw_ptr_rho_1_s;
    const ScalarType *raw_ptr_zeta_1_s;
    const ScalarType *raw_ptr_r_0;
    const ScalarType *raw_ptr_r_1;
    const ScalarType *raw_ptr_w_1;

    KERNEL_SS(int _N, const ScalarType *_raw_ptr_beta_0_s,
              const ScalarType *_raw_ptr_chi_0_s,
              const ScalarType *_raw_ptr_rho_0_s,
              const ScalarType *_raw_ptr_zeta_0_s,
              const ScalarType *_raw_ptr_alpha_1_s,
              const ScalarType *_raw_ptr_rho_1_s,
              const ScalarType *_raw_ptr_zeta_1_s,
              const ScalarType *_raw_ptr_r_0,
              const ScalarType *_raw_ptr_r_1,
              const ScalarType *_raw_ptr_w_1) :
	     N(_N),
             raw_ptr_beta_0_s(_raw_ptr_beta_0_s),
             raw_ptr_chi_0_s(_raw_ptr_chi_0_s),
             raw_ptr_rho_0_s(_raw_ptr_rho_0_s),
             raw_ptr_zeta_0_s(_raw_ptr_zeta_0_s),
             raw_ptr_alpha_1_s(_raw_ptr_alpha_1_s),
             raw_ptr_rho_1_s(_raw_ptr_rho_1_s),
             raw_ptr_zeta_1_s(_raw_ptr_zeta_1_s),
             raw_ptr_r_0(_raw_ptr_r_0),
             raw_ptr_r_1(_raw_ptr_r_1),
             raw_ptr_w_1(_raw_ptr_w_1)
   {}

    __host__ __device__
      ScalarType operator()(int index, ScalarType val)
    {
      unsigned int N_s = index / N;
      unsigned int N_n = index % N;
      
      // return the transformed result
      return raw_ptr_zeta_1_s[N_s]*raw_ptr_rho_1_s[N_s]*raw_ptr_r_1[N_n]
	      +raw_ptr_alpha_1_s[N_s]*
	      (val-raw_ptr_chi_0_s[N_s]*raw_ptr_rho_0_s[N_s]
	       /raw_ptr_beta_0_s[N_s]*(raw_ptr_zeta_1_s[N_s]*raw_ptr_w_1[N_n]
		       -raw_ptr_zeta_0_s[N_s]*raw_ptr_r_0[N_n]));
	    
    }
  };

  // computes new x
  template <typename ScalarType>
    struct KERNEL_X : thrust::binary_function<int, ScalarType, ScalarType>
  {
    int N;
    const ScalarType *raw_ptr_beta_0_s;
    const ScalarType *raw_ptr_chi_0_s;
    const ScalarType *raw_ptr_rho_0_s;
    const ScalarType *raw_ptr_zeta_1_s;
    const ScalarType *raw_ptr_w_1;
    const ScalarType *raw_ptr_s_0_s;

    KERNEL_X(int _N, const ScalarType *_raw_ptr_beta_0_s,
	     const ScalarType *_raw_ptr_chi_0_s,
	     const ScalarType *_raw_ptr_rho_0_s,
	     const ScalarType *_raw_ptr_zeta_1_s,
	     const ScalarType *_raw_ptr_w_1,
	     const ScalarType *_raw_ptr_s_0_s) :
	     N(_N),
             raw_ptr_beta_0_s(_raw_ptr_beta_0_s),
             raw_ptr_chi_0_s(_raw_ptr_chi_0_s),
             raw_ptr_rho_0_s(_raw_ptr_rho_0_s),
             raw_ptr_zeta_1_s(_raw_ptr_zeta_1_s),
             raw_ptr_w_1(_raw_ptr_w_1),
             raw_ptr_s_0_s(_raw_ptr_s_0_s)
   {}

    __host__ __device__
      ScalarType operator()(int index, ScalarType val)
    {
      unsigned int N_s = index / N;
      unsigned int N_n = index % N;

      
      // return the transformed result
      return val-raw_ptr_beta_0_s[N_s]*raw_ptr_s_0_s[index]
	      +raw_ptr_chi_0_s[N_s]*raw_ptr_rho_0_s[N_s]
	      *raw_ptr_zeta_1_s[N_s]*raw_ptr_w_1[N_n];
      /*
      return val-raw_ptr_beta_0_s[N_s];
      */
    }
  };

  // computes new p
  template <typename ScalarType>
    struct KERNEL_P : thrust::binary_function<int, ScalarType, ScalarType>
  {
    int N;
    const ScalarType *alpha_0_s;
    const ScalarType *z_1_s;
    const ScalarType *r_0;

    KERNEL_P(int _N, const ScalarType *_alpha_0_s,
		    const ScalarType *_z_1_s, const ScalarType *_r_0):
	    N(_N), alpha_0_s(_alpha_0_s), z_1_s(_z_1_s), r_0(_r_0)
    {}

    __host__ __device__
      ScalarType operator()(int index, ScalarType val)
    {
      unsigned int N_s = index / N;
      unsigned int N_i = index % N;
      
      // return the transformed result
      return z_1_s[N_s]*r_0[N_i]+alpha_0_s[N_s]*val;
    }
  };

  // like blas::copy, but copies the same array many times into a larger array
  template <typename ScalarType>
    struct KERNEL_VCOPY : thrust::unary_function<int, ScalarType>
  {
    int N_t;
    const ScalarType *source;

    KERNEL_VCOPY(int _N_t, const ScalarType *_source) :
	    N_t(_N_t), source(_source)
    {}

    __host__ __device__
      ScalarType operator()(int index)
    {
      unsigned int N   = index % N_t;
      return source[N];
    }

  };

} // end namespace detail_m

// Methods in this namespace are all routines that involve using
// thrust::for_each to perform some transformations on arrays of data.
//
// Except for vectorize_copy, these are specific to CG-M.
//
// Each has a version that takes Array inputs, and another that takes iterators
// as input. The CG-M routine only explicitly refers version with Arrays as
// arguments. The Array version calls the iterator version which uses
// a struct from cusp::krylov::detail_m.
namespace trans_m
{
  // compute \zeta_1^\sigma, using iterators
  // uses detail_m::KERNEL_Z
  template <typename InputIterator1, typename InputIterator2,
            typename InputIterator3,
	    typename OutputIterator1,
	    typename ScalarType>
  void compute_z_m(InputIterator1 z_0_s_b, InputIterator1 z_0_s_e,
		InputIterator2 z_m1_s_b, InputIterator3 sig_b,
		OutputIterator1 z_1_s_b,
		ScalarType beta_m1, ScalarType beta_0, ScalarType alpha_0)
  {
    size_t N = z_0_s_e - z_0_s_b;
    thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(z_1_s_b,z_0_s_b,z_m1_s_b,sig_b)),
    thrust::make_zip_iterator(thrust::make_tuple(z_1_s_b,z_0_s_b,z_m1_s_b,sig_b))+N,
    cusp::krylov::detail_m::KERNEL_Z<ScalarType>(beta_m1,beta_0,alpha_0)
    );
  }

  // compute \beta_0^\sigma, using iterators
  // uses detail_m::KERNEL_B
  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator1,
	    typename ScalarType>
  void compute_b_m(InputIterator1 z_1_s_b, InputIterator1 z_1_s_e,
		InputIterator2 z_0_s_b, OutputIterator1 beta_0_s_b,
		ScalarType beta_0)
  {
    size_t N = z_1_s_e - z_1_s_b;

    thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(beta_0_s_b,z_1_s_b,z_0_s_b)),
    thrust::make_zip_iterator(thrust::make_tuple(beta_0_s_b,z_1_s_b,z_0_s_b))+N,
    cusp::krylov::detail_m::KERNEL_B<ScalarType>(beta_0)
    );
  }

  // compute \zeta_1^\sigma, using arrays
  template <typename Array1, typename Array2, typename Array3,
            typename Array4, typename ScalarType>
  void compute_z_m(const Array1& z_0_s, const Array2& z_m1_s,
		const Array3& sig, Array4& z_1_s,
		ScalarType beta_m1, ScalarType beta_0, ScalarType alpha_0)
  {
    // sanity checks
    cusp::blas::detail::assert_same_dimensions(z_0_s,z_m1_s,z_1_s);
    cusp::blas::detail::assert_same_dimensions(z_1_s,sig);

    // compute
    cusp::krylov::trans_m::compute_z_m(z_0_s.begin(),z_0_s.end(),
		    z_m1_s.begin(),sig.begin(),z_1_s.begin(),
                    beta_m1,beta_0,alpha_0);

  }

  // \beta_0^\sigma using arrays
  template <typename Array1, typename Array2, typename Array3,
            typename ScalarType>
  void compute_b_m(const Array1& z_1_s, const Array2& z_0_s,
		Array3& beta_0_s, ScalarType beta_0)
  {
    // sanity checks
    cusp::blas::detail::assert_same_dimensions(z_1_s,z_0_s,beta_0_s);

    // compute
    cusp::krylov::trans_m::compute_b_m(z_1_s.begin(),z_1_s.end(),
		    z_0_s.begin(),beta_0_s.begin(),beta_0);
  }

  // compute \alpha_0^\sigma, and swap \zeta_i^\sigma using iterators
  // uses detail_m::KERNEL_A
  template <typename InputIterator1, typename InputIterator2,
            typename InputIterator3, typename OutputIterator,
            typename ScalarType>
  void compute_a_m(InputIterator1 z_0_s_b, InputIterator1 z_0_s_e,
		InputIterator2 z_1_s_b, InputIterator3 beta_0_s_b,
                OutputIterator alpha_0_s_b,
		ScalarType beta_0, ScalarType alpha_0)
  {
    size_t N = z_0_s_e - z_0_s_b;
    thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(alpha_0_s_b,z_0_s_b,z_1_s_b,beta_0_s_b)),
    thrust::make_zip_iterator(thrust::make_tuple(alpha_0_s_b,z_0_s_b,z_1_s_b,beta_0_s_b))+N,
    cusp::krylov::detail_m::KERNEL_A<ScalarType>(beta_0,alpha_0));
  }

  // compute \alpha_0^\sigma, and swap \zeta_i^\sigma using arrays
  template <typename Array1, typename Array2, typename Array3,
            typename Array4, typename ScalarType>
  void compute_a_m(const Array1& z_0_s, const Array2& z_1_s,
                const Array3& beta_0_s, Array4& alpha_0_s,
		ScalarType beta_0, ScalarType alpha_0)
  {
    // sanity checks
    cusp::blas::detail::assert_same_dimensions(z_0_s,z_1_s);
    cusp::blas::detail::assert_same_dimensions(z_0_s,alpha_0_s,beta_0_s);

    // compute
    cusp::krylov::trans_m::compute_a_m(z_0_s.begin(), z_0_s.end(),
		z_1_s.begin(), beta_0_s.begin(), alpha_0_s.begin(),
                beta_0, alpha_0);
  }

  // compute x^\sigma
  // uses detail_m::KERNEL_X
  template <typename Array1, typename Array2, typename Array3,
            typename Array4, typename Array5, typename Array6, typename Array7>
  void compute_x_m(const Array1& beta_0_s, const Array2& chi_0_s,
                const Array3& rho_0_s, const Array4& zeta_1_s,
                const Array5& w_1, const Array6& s_0_s, Array7& x_0_s)
  {
    // sanity check
    cusp::blas::detail::assert_same_dimensions(beta_0_s,chi_0_s,rho_0_s);
    cusp::blas::detail::assert_same_dimensions(rho_0_s,zeta_1_s);
    cusp::blas::detail::assert_same_dimensions(s_0_s,x_0_s);

    size_t N = w_1.end()-w_1.begin();
    size_t N_s = beta_0_s.end()-beta_0_s.begin();
    size_t N_t = x_0_s.end()-x_0_s.begin();
    assert (N_t == N*N_s);

    // counting iterators to pass to thrust::transform
    thrust::counting_iterator<int> counter(0);

    // get raw pointers for passing to kernels
    typedef typename Array1::value_type   ScalarType;
    const ScalarType *raw_ptr_beta_0_s = thrust::raw_pointer_cast(beta_0_s.data());
    const ScalarType *raw_ptr_chi_0_s  = thrust::raw_pointer_cast(chi_0_s.data());
    const ScalarType *raw_ptr_rho_0_s  = thrust::raw_pointer_cast(rho_0_s.data());
    const ScalarType *raw_ptr_zeta_1_s = thrust::raw_pointer_cast(zeta_1_s.data());
    const ScalarType *raw_ptr_w_1      = thrust::raw_pointer_cast(w_1.data());
    const ScalarType *raw_ptr_s_0_s    = thrust::raw_pointer_cast(s_0_s.data());

    // compute x
    thrust::transform(counter,counter+N_t,x_0_s.begin(),x_0_s.begin(),
    cusp::krylov::detail_m::KERNEL_X<ScalarType>(N,raw_ptr_beta_0_s,raw_ptr_chi_0_s,raw_ptr_rho_0_s,raw_ptr_zeta_1_s,raw_ptr_w_1,raw_ptr_s_0_s));
  }
  
  // compute s^\sigma
  // uses detail_m::KERNEL_SS
  template <typename Array1, typename Array2, typename Array3, typename Array4,
	   typename Array5, typename Array6, typename Array7, typename Array8,
	   typename Array9, typename Array10, typename Array11>
  void compute_s_m(const Array1& beta_0_s, const Array2& chi_0_s,
                const Array3& rho_0_s, const Array4& zeta_0_s,
                const Array5& alpha_1_s, const Array6& rho_1_s,
		const Array7& zeta_1_s,
                const Array8& r_0, Array9& r_1,
                const Array10& w_1, Array11& s_0_s)
  {
    // sanity check
    cusp::blas::detail::assert_same_dimensions(beta_0_s,chi_0_s,rho_0_s);
    cusp::blas::detail::assert_same_dimensions(beta_0_s,zeta_0_s,zeta_1_s);
    cusp::blas::detail::assert_same_dimensions(alpha_1_s,rho_1_s,zeta_1_s);
    cusp::blas::detail::assert_same_dimensions(r_0,r_1,w_1);

    size_t N = w_1.end()-w_1.begin();
    size_t N_s = beta_0_s.end()-beta_0_s.begin();
    size_t N_t = s_0_s.end()-s_0_s.begin();
    assert (N_t == N*N_s);

    // counting iterators to pass to thrust::transform
    thrust::counting_iterator<int> counter(0);

    // get raw pointers for passing to kernels
    typedef typename Array1::value_type   ScalarType;
    const ScalarType *raw_ptr_beta_0_s  = thrust::raw_pointer_cast(beta_0_s.data());
    const ScalarType *raw_ptr_chi_0_s   = thrust::raw_pointer_cast(chi_0_s.data());
    const ScalarType *raw_ptr_rho_0_s   = thrust::raw_pointer_cast(rho_0_s.data());
    const ScalarType *raw_ptr_zeta_0_s  = thrust::raw_pointer_cast(zeta_0_s.data());
    const ScalarType *raw_ptr_alpha_1_s = thrust::raw_pointer_cast(alpha_1_s.data());
    const ScalarType *raw_ptr_rho_1_s   = thrust::raw_pointer_cast(rho_1_s.data());
    const ScalarType *raw_ptr_zeta_1_s  = thrust::raw_pointer_cast(zeta_1_s.data());
    const ScalarType *raw_ptr_r_0       = thrust::raw_pointer_cast(r_0.data());
    const ScalarType *raw_ptr_r_1       = thrust::raw_pointer_cast(r_1.data());
    const ScalarType *raw_ptr_w_1       = thrust::raw_pointer_cast(w_1.data());

    // compute x
    thrust::transform(counter,counter+N_t,s_0_s.begin(),s_0_s.begin(),
    cusp::krylov::detail_m::KERNEL_SS<ScalarType>(N, raw_ptr_beta_0_s, raw_ptr_chi_0_s, raw_ptr_rho_0_s, raw_ptr_zeta_0_s, raw_ptr_alpha_1_s, raw_ptr_rho_1_s, raw_ptr_zeta_1_s, raw_ptr_r_0, raw_ptr_r_1, raw_ptr_w_1));
  }

  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator, typename ScalarType>
  void compute_w_1_m(InputIterator1 r_0_b, InputIterator1 r_0_e,
		    InputIterator2 As_b, OutputIterator w_1_b,
		    ScalarType beta_0)
  {
    size_t N = r_0_e-r_0_b;
    thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(w_1_b,r_0_b,As_b)),
    thrust::make_zip_iterator(thrust::make_tuple(w_1_b,r_0_b,As_b))+N,
    cusp::krylov::detail_m::KERNEL_W<ScalarType>(beta_0));
  }

  template <typename Array1, typename Array2, typename Array3,
            typename ScalarType>
  void compute_w_1_m(const Array1& r_0, const Array2& As, Array3& w_1,
		  ScalarType beta_0)
  {
    // sanity checks
    cusp::blas::detail::assert_same_dimensions(r_0,As,w_1);

    // compute
    cusp::krylov::trans_m::compute_w_1_m(r_0.begin(),r_0.end(),
		    As.begin(),w_1.begin(),beta_0);
  }

  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator, typename ScalarType>
  void compute_r_1_m(InputIterator1 w_1_b, InputIterator1 w_1_e,
		    InputIterator2 Aw_b, OutputIterator r_1_b,
		    ScalarType chi_0)
  {
    size_t N = w_1_e-w_1_b;
    thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(r_1_b,w_1_b,Aw_b)),
    thrust::make_zip_iterator(thrust::make_tuple(r_1_b,w_1_b,Aw_b))+N,
    cusp::krylov::detail_m::KERNEL_W<ScalarType>(-chi_0));
  }

  template <typename Array1, typename Array2, typename Array3,
            typename ScalarType>
  void compute_r_1_m(const Array1& w_1, const Array2& Aw, Array3& r_1,
		  ScalarType chi_0)
  {
    // sanity checks
    cusp::blas::detail::assert_same_dimensions(w_1,Aw,r_1);

    // compute
    cusp::krylov::trans_m::compute_r_1_m(w_1.begin(),w_1.end(),
		    Aw.begin(),r_1.begin(),chi_0);
  }

  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator, typename ScalarType>
  void compute_s_0_m(InputIterator1 r_1_b, InputIterator1 r_1_e,
		    InputIterator2 As_b, OutputIterator s_0_b,
		    ScalarType alpha_1, ScalarType chi_0)
  {
    size_t N = r_1_e-r_1_b;
    thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(s_0_b,r_1_b,As_b)),
    thrust::make_zip_iterator(thrust::make_tuple(s_0_b,r_1_b,As_b))+N,
    cusp::krylov::detail_m::KERNEL_S<ScalarType>(alpha_1,chi_0));
  }

  template <typename Array1, typename Array2, typename Array3,
            typename ScalarType>
  void compute_s_0_m(const Array1& r_1, const Array2& As, Array3& s_0,
		  ScalarType alpha_1, ScalarType chi_0)
  {
    // sanity checks
    cusp::blas::detail::assert_same_dimensions(r_1,As,s_0);

    // compute
    cusp::krylov::trans_m::compute_s_0_m(r_1.begin(),r_1.end(),
		    As.begin(),s_0.begin(),alpha_1,chi_0);
  }

  template <typename InputIterator, typename OutputIterator,
	    typename ScalarType>
  void compute_chi_m(InputIterator sigma_b, InputIterator sigma_e,
		     OutputIterator chi_0_s_b, ScalarType chi_0)
  {
    size_t N = sigma_e-sigma_b;
    thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(chi_0_s_b,sigma_b)),
    thrust::make_zip_iterator(thrust::make_tuple(chi_0_s_b,sigma_b))+N,
    cusp::krylov::detail_m::KERNEL_CHI<ScalarType>(chi_0));
  }

  template <typename Array1, typename Array2, typename ScalarType>
  void compute_chi_m(const Array1& sigma, Array2& chi_0_s, ScalarType chi_0)
  {
    // sanity checks
    cusp::blas::detail::assert_same_dimensions(sigma,chi_0_s);

    // compute
    cusp::krylov::trans_m::compute_chi_m(sigma.begin(),sigma.end(),
		    chi_0_s.begin(),chi_0);
  }

  template <typename InputIterator1, typename InputIterator2,
	    typename OutputIterator, typename ScalarType>
  void compute_rho_m(InputIterator1 rho_0_s_b, InputIterator1 rho_0_s_e,
		     InputIterator2 sigma_b, OutputIterator rho_1_s_b,
		     ScalarType chi_0)
  {
    size_t N = rho_0_s_e-rho_0_s_b;
    thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(rho_1_s_b,rho_0_s_b,sigma_b)),
    thrust::make_zip_iterator(thrust::make_tuple(rho_1_s_b,rho_0_s_b,sigma_b))+N,
    cusp::krylov::detail_m::KERNEL_RHO<ScalarType>(chi_0));
  }

  template <typename Array1, typename Array2, typename Array3,
	    typename ScalarType>
  void compute_rho_m(const Array1& rho_0_s, const Array2& sigma,
		  Array3& rho_1_s, ScalarType chi_0)
  {
    // sanity checks
    cusp::blas::detail::assert_same_dimensions(sigma,rho_0_s,rho_1_s);

    // compute
    cusp::krylov::trans_m::compute_rho_m(rho_0_s.begin(),rho_0_s.end(),
		    sigma.begin(),rho_1_s.begin(),chi_0);
  }

  // multiple copy of array to another array
  // this is just a vectorization of blas::copy
  // uses detail_m::KERNEL_VCOPY
  template <typename Array1, typename Array2>
    void vectorize_copy(const Array1& source, Array2& dest)
  {
    // sanity check
    size_t N = source.end()-source.begin();
    size_t N_t = dest.end()-dest.begin();
    assert ( N_t%N == 0 );

    // counting iterators to pass to thrust::transform
    thrust::counting_iterator<int> counter(0);

    // pointer to data
    typedef typename Array1::value_type   ScalarType;
    const ScalarType *raw_ptr_source = thrust::raw_pointer_cast(source.data());

    // compute
    thrust::transform(counter,counter+N_t,dest.begin(),
    cusp::krylov::detail_m::KERNEL_VCOPY<ScalarType>(N,raw_ptr_source));

  }

} // end namespace trans_m

// BiCGStab-M routine that uses the default monitor to determine completion
template <class LinearOperator,
          class VectorType1, class VectorType2, class VectorType3>
void bicgstab_m(LinearOperator& A,
        VectorType1& x, VectorType2& b, VectorType3& sigma)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::default_monitor<ValueType> monitor(b);

    return bicgstab_m(A, x, b, sigma, monitor);
}

// BiCGStab-M routine that takes a user specified monitor
template <class LinearOperator,
          class VectorType1, class VectorType2, class VectorType3,
          class Monitor>
void bicgstab_m(LinearOperator& A,
        VectorType1& x, VectorType2& b, VectorType3& sigma,
        Monitor& monitor)
{
  //
  // This bit is initialization of the solver.
  //


  // shorthand for typenames
  typedef typename LinearOperator::value_type   ValueType;
  typedef typename LinearOperator::memory_space MemorySpace;

  // sanity checking
  const size_t N = A.num_rows;
  const size_t N_t = x.end()-x.begin();
  const size_t test = b.end()-b.begin();
  const size_t N_s = sigma.end()-sigma.begin();

  assert(A.num_rows == A.num_cols);
  assert(N_t == N*N_s);
  assert(N == test);

  //clock_t start = clock();

  // w has data used in computing the soln.
  cusp::array1d<ValueType,MemorySpace> w_0(N);
  cusp::array1d<ValueType,MemorySpace> w_1(N);
  cusp::array1d<ValueType,MemorySpace> w_i(N);

  // stores residuals
  cusp::array1d<ValueType,MemorySpace> r_0(N);
  cusp::array1d<ValueType,MemorySpace> r_1(N);

  // used in iterates
  cusp::array1d<ValueType,MemorySpace> xx_0(N,ValueType(0));

  // used in iterates
  cusp::array1d<ValueType,MemorySpace> s_0(N);
  cusp::array1d<ValueType,MemorySpace> s_0_s(N_t);

  // stores parameters used in the iteration
  cusp::array1d<ValueType,MemorySpace> z_m1_s(N_s,ValueType(1));
  cusp::array1d<ValueType,MemorySpace> z_0_s(N_s,ValueType(1));
  cusp::array1d<ValueType,MemorySpace> z_1_s(N_s);

  cusp::array1d<ValueType,MemorySpace> alpha_0_s(N_s,ValueType(0));
  cusp::array1d<ValueType,MemorySpace> beta_0_s(N_s);

  cusp::array1d<ValueType,MemorySpace> rho_0_s(N_s,ValueType(1));
  cusp::array1d<ValueType,MemorySpace> rho_1_s(N_s);
  cusp::array1d<ValueType,MemorySpace> chi_0_s(N_s);

  // stores parameters used in the iteration for the undeformed system
  ValueType beta_m1, beta_0(ValueType(1));
  ValueType alpha_0(ValueType(0));

  ValueType delta_0, delta_1;
  ValueType phi_0;
  ValueType chi_0;

  // stores the value of the matrix-vector product we have to compute
  cusp::array1d<ValueType,MemorySpace> As(N);
  cusp::array1d<ValueType,MemorySpace> Aw(N);

  // set up the initial conditions for the iteration
  cusp::blas::copy(b,r_0);
  cusp::blas::copy(b,w_0);
  cusp::blas::copy(w_0,w_i);

  // set up the intitial guess
  cusp::blas::fill(x.begin(),x.end(),ValueType(0));

  // set up initial value of p_0 and p_0^\sigma
  cusp::krylov::trans_m::vectorize_copy(b,s_0_s);
  cusp::blas::copy(b,s_0);
  cusp::multiply(A,s_0,As);

  delta_1 = cusp::blas::dotc(w_0,r_0);
  phi_0 = cusp::blas::dotc(w_0,As)/delta_1;
  
  //
  // Initialization is done. Solve iteratively
  //
  while (!monitor.finished(r_0))
  {
    // recycle iterates
    beta_m1 = beta_0;
    beta_0 = ValueType(-1.0)/phi_0;
    delta_0 = delta_1;

    // compute \zeta_1^\sigma
    cusp::krylov::trans_m::compute_z_m(z_0_s, z_m1_s, sigma, z_1_s,
                                      beta_m1, beta_0, alpha_0);
    // compute \beta_0^\sigma
    cusp::krylov::trans_m::compute_b_m(z_1_s, z_0_s, beta_0_s, beta_0);

    // call w_1 kernel
    cusp::krylov::trans_m::compute_w_1_m(r_0, As, w_1, beta_0);

    // compute the matrix-vector product Aw
    cusp::multiply(A,w_1,Aw);

    // compute chi_0
    chi_0 = cusp::blas::dotc(Aw,w_1)/cusp::blas::dotc(Aw,Aw);

    // compute shifted rho, chi
    cusp::krylov::trans_m::compute_chi_m(sigma,chi_0_s,chi_0);
    cusp::krylov::trans_m::compute_rho_m(rho_0_s,sigma,rho_1_s,chi_0);

    // compute new residual
    cusp::krylov::trans_m::compute_r_1_m(w_1,Aw,r_1,chi_0);

    // compute the new solution
    cusp::krylov::trans_m::compute_x_m(beta_0_s, chi_0_s, rho_0_s, z_1_s,
		    w_1, s_0_s, x);

    // compute the new delta
    delta_1 = cusp::blas::dotc(w_i,r_1);

    // compute new alpha
    alpha_0 = -beta_0*delta_1/delta_0/chi_0;

    // calculate \alpha_0^\sigma
    cusp::krylov::trans_m::compute_a_m(z_0_s, z_1_s, beta_0_s,
                                      alpha_0_s, beta_0, alpha_0);

    // compute s_0
    cusp::krylov::trans_m::compute_s_0_m(r_1,As,s_0,alpha_0,chi_0);

    // compute s_0^sigma
    cusp::krylov::trans_m::compute_s_m(beta_0_s, chi_0_s, rho_0_s, z_0_s,
		    alpha_0_s, rho_1_s, z_1_s, r_0, r_1, w_1, s_0_s);

    // compute As
    cusp::multiply(A,s_0,As);

    // compute new phi
    phi_0 = cusp::blas::dotc(w_i,As)/delta_1;

    // recycle w_i and r_i
    cusp::blas::copy(w_1,w_0);
    cusp::blas::copy(r_1,r_0);

    // recycle \zeta_i^\sigma
    cusp::blas::copy(z_0_s,z_m1_s);
    cusp::blas::copy(z_1_s,z_0_s);

    // recycle \rho_i^\sigma
    cusp::blas::copy(rho_1_s,rho_0_s);
    
    ++monitor;

  }// finished iteration

  //cudaThreadSynchronize();

  // MFLOPs excludes BLAS operations
  //double elapsed = ((double) (clock() - start)) / CLOCKS_PER_SEC;
  //double MFLOPs = 2* ((double) i * (double) A.num_entries)/ (1e6 * elapsed);
  //printf("-iteration completed in %lfms  ( > %6.2lf MFLOPs )\n",1000*elapsed, MFLOPs );

  
} // end cg_m

} // end namespace krylov
} // end namespace cusp
