#include <unittest/unittest.h>
#include <cusp/complex.h>
#include <complex>

#include <cusp/verify.h>

// Macro to create host and device versions of a unit test
#define DECLARE_NUMERIC_UNITTEST(VTEST)                    \
void VTEST##Float(void)   {  VTEST<float>(); }  \
void VTEST##Double(void)   {  VTEST<double>(); }  \
DECLARE_UNITTEST(VTEST##Float);                                 \
DECLARE_UNITTEST(VTEST##Double);

template <typename ValueType>
__host__ __device__ cusp::complex<ValueType> test_complex_members(){
  cusp::complex<ValueType> a(ValueType(1.0),ValueType(1.0));
  cusp::complex<ValueType> b(ValueType(2.0),ValueType(1.0));
  a += b;
  a -= b;
  a *= b;
  a /= b;
  return a;
}

template <typename ValueType>
__host__ __device__ cusp::complex<ValueType> test_complex_non_members(){
  cusp::complex<ValueType> a(ValueType(3.0),ValueType(1.0));
  cusp::complex<ValueType> b(ValueType(2.0),ValueType(-1.0));
  a = a*ValueType(2.0);
  a = ValueType(2.0)*b;
  a = a*b;
  a = a/ValueType(2.0);
  a = ValueType(2.0)/b;
  a = a/b;
  a = a-ValueType(2.0);
  a = ValueType(2.0)-b;
  a = a-b;
  a = a+ValueType(2.0);
  a = ValueType(2.0)+b;
  a = a+b;
  b = cusp::abs(b);
  b = cusp::arg(b);
  b = cusp::norm(b);
  b = cusp::conj(b);
  b = cusp::polar(ValueType(0.3),ValueType(3.0));

  a = cusp::cos(b);
  b = cusp::cosh(a);
  a = cusp::exp(b);

  b = cusp::log(a);
  a = cusp::log10(b);

  b = cusp::pow(a,b);
  a = cusp::pow(b,ValueType(1.3));
  b = cusp::pow(ValueType(1.4),a);
  a = cusp::pow(b,4);
  b = cusp::sin(a);
  a = cusp::sinh(b);
  b = cusp::sqrt(a);
  a = cusp::tan(b);
  b = cusp::tanh(a);
  a = cusp::acos(b);
  b = cusp::asin(a);
  a = cusp::atan(b);
  return a;
}

template <typename ValueType>
__host__ __device__ cusp::complex<ValueType> test_complex_compilation_entry(){
  return test_complex_members<ValueType>() + test_complex_non_members<ValueType>();
}

template <typename ValueType>
__global__ void test_complex_compilation_kernel(cusp::complex<ValueType> * a){
  cusp::complex<ValueType> ret = test_complex_compilation_entry<ValueType>();
  *a = ret;
}

template <typename ValueType>
void test_complex_compilation()
{
  cusp::complex<ValueType> a;
  cusp::complex<ValueType> * d_a;  
  cudaMalloc(&d_a,sizeof(cusp::complex<ValueType>));
  test_complex_compilation_kernel<ValueType>
    <<<1,1>>>(d_a);
  cudaMemcpy(&a,d_a,sizeof(cusp::complex<ValueType>),cudaMemcpyDeviceToHost);
  std::complex<ValueType> b(a.real(),a.imag());
  a = test_complex_compilation_entry<ValueType>();
  ASSERT_ALMOST_EQUAL(a.real(),b.real());
}
DECLARE_NUMERIC_UNITTEST(test_complex_compilation);


