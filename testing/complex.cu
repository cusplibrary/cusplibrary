#include <unittest/unittest.h>
#include <cusp/array1d.h>
#include <cusp/complex.h>
#include <complex>

#include <cusp/verify.h>

#define ASSERT_COMPLEX_ALMOST_EQUAL(X,Y) {unittest::assert_almost_equal((X.real()),(Y.real()), __FILE__, __LINE__);unittest::assert_almost_equal((X.imag()),(Y.imag()), __FILE__, __LINE__);}

template< typename T1, typename T2 >
struct is_same_type      { enum { result = false }; };

template< typename T>
struct is_same_type<T,T> { enum { result = true }; };

#ifdef __GNUC__
extern "C"{
  float __complex__ cacosf(float __complex__ z);
  double __complex__ cacos(double __complex__ z);
  float __complex__ casinf(float __complex__ z);
  double __complex__ casin(double __complex__ z);
  float __complex__ catanf(float __complex__ z);
  double __complex__ catan(double __complex__ z);
  float __complex__ cacoshf(float __complex__ z);
  double __complex__ cacosh(double __complex__ z);
  float __complex__ casinhf(float __complex__ z);
  double __complex__ casinh(double __complex__ z);
  float __complex__ catanhf(float __complex__ z);
  double __complex__ catanh(double __complex__ z);
  double creal(double __complex__ z);
  double cimag(double __complex__ z);
}
#endif


// Macro to create host and device versions of a unit test
#define DECLARE_NUMERIC_UNITTEST(VTEST)                    \
void VTEST##Float(void)   {  VTEST<float>(); }  \
void VTEST##Double(void)   {  VTEST<double>(); }  \
DECLARE_UNITTEST(VTEST##Float);                                 \
DECLARE_UNITTEST(VTEST##Double);

template <typename ValueType>
__host__ bool compareWithStd(cusp::complex<ValueType> a){
  //  std::cout << "Testing " << a << std::endl; 
  cusp::complex<ValueType> b(a.real(),a.imag());
  std::complex<ValueType> s_a(a.real(),a.imag());
  std::complex<ValueType> s_b(b.real(),b.imag());
  ASSERT_COMPLEX_ALMOST_EQUAL(a,s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b += a;
  s_b += s_a;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b -= a;
  s_b -= s_a;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b *= a;
  s_b *= s_a;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b /= a;
  s_b /= s_a;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

  b = a*ValueType(2.0);
  s_b = s_a*ValueType(2.0);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = ValueType(2.0)*a;
  s_b = ValueType(2.0)*s_a;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = b*a;
  s_b = s_b*s_a;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = a/ValueType(2.0);
  s_b = s_a/ValueType(2.0);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = ValueType(2.0)/a;
  s_b = ValueType(2.0)/s_a;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = a/b;
  s_b = s_a/s_b;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = a-ValueType(2.0);
  s_b = s_a-ValueType(2.0);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = ValueType(2.0)-a;
  s_b = ValueType(2.0)-s_a;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = a-b;
  s_b = s_a-s_b;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = a+ValueType(2.0);
  s_b = s_a+ValueType(2.0);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = ValueType(2.0)+a;
  s_b = ValueType(2.0)+s_a;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = a+b;
  s_b = s_a+s_b;
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

  b = cusp::abs(a);
  s_b = std::abs(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::arg(a);
  s_b = std::arg(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::norm(a);
  s_b = std::norm(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::conj(a);
  s_b = std::conj(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::polar(norm(a),a.imag());
  s_b = std::polar(norm(a),a.imag());
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

  b = cusp::cos(a);
  s_b = std::cos(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::cosh(a);
  s_b = std::cosh(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::exp(a);
  s_b = std::exp(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

  b = cusp::log(a);
  s_b = std::log(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::log10(a);
  s_b = std::log10(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

  b = cusp::pow(a,b);
  s_b = std::pow(s_a,s_b);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::pow(a,ValueType(1.3));
  s_b = std::pow(s_a,ValueType(1.3));
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::pow(ValueType(1.4),a);
  s_b = std::pow(ValueType(1.4),s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  // Missing int implentation in std
  b = cusp::pow(a,4);
  s_b = std::pow(s_a,ValueType(4.0));
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::sin(a);
  s_b = std::sin(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::sinh(a);
  s_b = std::sinh(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::sqrt(a);
  s_b = std::sqrt(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::tan(a);  
  s_b = std::tan(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  b = cusp::tanh(a);
  s_b = std::tanh(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

  // Inverse trigonometic functions not part of standard 
  /*
  a = cusp::acos(b);
  s_a = std::acos(s_b);
  ASSERT_COMPLEX_ALMOST_EQUAL(a,s_a);
  b = cusp::asin(a);
  s_b = std::asin(s_a);
  ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
  a = cusp::atan(b);
  s_a = std::atan(s_b);
  ASSERT_COMPLEX_ALMOST_EQUAL(a,s_a);
  */

#ifdef __GNUC__  
  /* Use the c99 complex function from gcc to test the
   function not part of the standard */
  if(is_same_type<ValueType,float>::result){
    __complex__ float g_a;
    __complex__ float g_b;
    g_a = s_a.real() + s_a.imag()*__I__;
    g_b = s_b.real() + s_b.imag()*__I__;
    b = cusp::acos(a);
    g_b = cacosf(g_a);
    s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
    ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

    g_a = s_a.real() + s_a.imag()*__I__;
    g_b = s_b.real() + s_b.imag()*__I__;
    b = cusp::asin(a);
    g_b = casinf(g_a);
    s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
    ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

    g_a = s_a.real() + s_a.imag()*__I__;
    g_b = s_b.real() + s_b.imag()*__I__;
    b = cusp::atan(a);
    g_b = catanf(g_a);
    s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
    ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

    g_a = s_a.real() + s_a.imag()*__I__;
    g_b = s_b.real() + s_b.imag()*__I__;
    b = cusp::acosh(a);
    g_b = cacoshf(g_a);
    s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
    ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

    g_a = s_a.real() + s_a.imag()*__I__;
    g_b = s_b.real() + s_b.imag()*__I__;
    b = cusp::asinh(a);
    g_b = casinhf(g_a);
    s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
    ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

    if(a != cusp::complex<ValueType>(1,0)){
      g_a = s_a.real() + s_a.imag()*__I__;
      g_b = s_b.real() + s_b.imag()*__I__;
      b = cusp::atanh(a);
      g_b = catanhf(g_a);
      s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
      ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
    }
  }else if(is_same_type<ValueType,double>::result){
    __complex__ double g_a;
    __complex__ double g_b;
    g_a = s_a.real() + s_a.imag()*__I__;
    g_b = s_b.real() + s_b.imag()*__I__;
    b = cusp::acos(a);
    g_b = cacos(g_a);
    s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
    ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

    g_a = s_a.real() + s_a.imag()*__I__;
    g_b = s_b.real() + s_b.imag()*__I__;
    b = cusp::asin(a);
    g_b = casin(g_a);
    s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
    ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

    g_a = s_a.real() + s_a.imag()*__I__;
    g_b = s_b.real() + s_b.imag()*__I__;
    b = cusp::atan(a);
    g_b = catan(g_a);
    s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
    ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

    g_a = s_a.real() + s_a.imag()*__I__;
    g_b = s_b.real() + s_b.imag()*__I__;
    b = cusp::acosh(a);
    g_b = cacosh(g_a);
    s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
    ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

    g_a = s_a.real() + s_a.imag()*__I__;
    g_b = s_b.real() + s_b.imag()*__I__;
    b = cusp::asinh(a);
    g_b = casinh(g_a);
    s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
    ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);

    if(a != cusp::complex<ValueType>(1,0)){
      g_a = s_a.real() + s_a.imag()*__I__;
      g_b = s_b.real() + s_b.imag()*__I__;
      b = cusp::atanh(a);
      g_b = catanh(g_a);
      s_b = std::complex<ValueType>(creal(g_b),cimag(g_b));
      ASSERT_COMPLEX_ALMOST_EQUAL(b,s_b);
    }
  }
#endif
  return true;
}

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

#if __CUDA_ARCH__ < 130
// Don't try to run the double precision tests if the compiled
// architecture doesn't support it 
template <>
__global__ void test_complex_compilation_kernel(cusp::complex<double> * a){
}
#endif

bool compiled_architecture_supports_double(void){
#if __CUDA_ARCH__ >= 130
  return true;
#else
  return false;
#endif
}

bool device_supports_double(void)
{
    int current_device = -1;
    cudaDeviceProp properties;

    cudaError_t error = cudaGetDevice(&current_device);
    if(error)
        throw thrust::system_error(error, thrust::cuda_category());

    if(current_device < 0)
        throw thrust::system_error(cudaErrorNoDevice, thrust::cuda_category());
    
    // the properties weren't found, ask the runtime to generate them
    error = cudaGetDeviceProperties(&properties, current_device);

    if(error)
      throw thrust::system_error(error, thrust::cuda_category());

    return properties.major >= 2 || (properties.major == 1 && properties.minor >= 3);
}

template <typename MemorySpace>
void TestComplexRealConversion()
{
  typedef float                Real;
  typedef cusp::complex<float> Complex;

  cusp::array1d<Real, MemorySpace>    real_values(4);
  cusp::array1d<Complex, MemorySpace> complex_values(4);

  // test real->complex conversion
  real_values[0] = 0;
  real_values[1] = 1;
  real_values[2] = 2;
  real_values[3] = 3;
  
  complex_values = real_values;

  ASSERT_EQUAL((Complex) complex_values[0], Complex(0,0));
  ASSERT_EQUAL((Complex) complex_values[1], Complex(1,0));
  ASSERT_EQUAL((Complex) complex_values[2], Complex(2,0));
  ASSERT_EQUAL((Complex) complex_values[3], Complex(3,0));
}
DECLARE_HOST_DEVICE_UNITTEST(TestComplexRealConversion);


template <typename ValueType>
struct TestComplexStdComplexConversion
{
  void operator()(void)
  {
    typedef std::complex<ValueType>  StdComplex;
    typedef cusp::complex<ValueType> CuspComplex;

    ASSERT_EQUAL(CuspComplex(StdComplex(0,0)), CuspComplex(0,0));
    ASSERT_EQUAL(CuspComplex(StdComplex(0,1)), CuspComplex(0,1));
    ASSERT_EQUAL(CuspComplex(StdComplex(1,0)), CuspComplex(1,0));
    ASSERT_EQUAL(CuspComplex(StdComplex(1,2)), CuspComplex(1,2));

    // can't test StdComplex(CuspComplex(...)) due to constructor ambiguity

    { StdComplex a(0,0); CuspComplex b = a;  ASSERT_EQUAL(b, CuspComplex(0,0)); }
    { StdComplex a(0,1); CuspComplex b = a;  ASSERT_EQUAL(b, CuspComplex(0,1)); }
    { StdComplex a(1,0); CuspComplex b = a;  ASSERT_EQUAL(b, CuspComplex(1,0)); }
    { StdComplex a(1,2); CuspComplex b = a;  ASSERT_EQUAL(b, CuspComplex(1,2)); }

    { CuspComplex a(0,0); StdComplex b = a;  ASSERT_EQUAL(b, StdComplex(0,0)); }
    { CuspComplex a(0,1); StdComplex b = a;  ASSERT_EQUAL(b, StdComplex(0,1)); }
    { CuspComplex a(1,0); StdComplex b = a;  ASSERT_EQUAL(b, StdComplex(1,0)); }
    { CuspComplex a(1,2); StdComplex b = a;  ASSERT_EQUAL(b, StdComplex(1,2)); }
  }
};
SimpleUnitTest<TestComplexStdComplexConversion, unittest::type_list<float,double> > TestComplexStdComplexConversionInstance;


template <typename ValueType>
void TestComplex()
{
  cusp::complex<ValueType> a;
  cusp::complex<ValueType> * d_a;  
  cudaMalloc(&d_a,sizeof(cusp::complex<ValueType>));
  test_complex_compilation_kernel<ValueType>
    <<<1,1>>>(d_a);
  cudaMemcpy(&a,d_a,sizeof(cusp::complex<ValueType>),cudaMemcpyDeviceToHost);
  std::complex<ValueType> b(a.real(),a.imag());
  a = test_complex_compilation_entry<ValueType>();
  // Don't check for equality between host and device code when the 
  // hardware device does not support double precision 
  if(is_same_type<ValueType,double>::result == false ||
     (device_supports_double() && compiled_architecture_supports_double())){
    ASSERT_COMPLEX_ALMOST_EQUAL(a,b);
  }
  // Test twice the unit circle 
  for(int i = 0;i < 32;i++){
    ValueType theta(ValueType(i*M_PI/8));
    compareWithStd<ValueType>(cusp::polar<ValueType>(ValueType(1),theta));
  }
}
DECLARE_NUMERIC_UNITTEST(TestComplex);


