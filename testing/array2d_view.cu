#include <unittest/unittest.h>

#include <cusp/array2d.h>

#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cusp/print.h> // TODO delete

template <typename MemorySpace>
void TestArray2dView(void)
{
  typedef typename cusp::array2d<int, MemorySpace>  Container;
  typedef typename cusp::array1d<int, MemorySpace>  Array;
  typedef typename Array::iterator                  Iterator;
  typedef typename cusp::array1d_view<Iterator>     ArrayView;
  typedef cusp::array2d_view<ArrayView>             View;

  // construct from array2d container
  {
    Container C(3,2,0);

    View V(C);
    ASSERT_EQUAL(V.num_rows,       3);
    ASSERT_EQUAL(V.num_cols,       2);
    ASSERT_EQUAL(V.num_entries,    6);
    ASSERT_EQUAL(V.pitch,          2);
    ASSERT_EQUAL(V.values.size(),  6);

    V(0,0) = 1; V(0,1) = 2;
    V(1,0) = 3; V(1,1) = 4;
    V(2,0) = 5; V(2,1) = 6;

    ASSERT_EQUAL(C(0,0), 1);  ASSERT_EQUAL(C(0,1), 2);
    ASSERT_EQUAL(C(1,0), 3);  ASSERT_EQUAL(C(1,1), 4);
    ASSERT_EQUAL(C(2,0), 5);  ASSERT_EQUAL(C(2,1), 6);
  }    
  
  // construct from array1d view
  {
    Array A(6);

    View V(3, 2, 2, ArrayView(A));
    ASSERT_EQUAL(V.num_rows,       3);
    ASSERT_EQUAL(V.num_cols,       2);
    ASSERT_EQUAL(V.num_entries,    6);
    ASSERT_EQUAL(V.pitch,          2);
    ASSERT_EQUAL(V.values.size(),  6);

    V(0,0) = 1; V(0,1) = 2;
    V(1,0) = 3; V(1,1) = 4;
    V(2,0) = 5; V(2,1) = 6;

    ASSERT_EQUAL(A[0], 1);
    ASSERT_EQUAL(A[1], 2);
    ASSERT_EQUAL(A[2], 3);
    ASSERT_EQUAL(A[3], 4);
    ASSERT_EQUAL(A[4], 5);
    ASSERT_EQUAL(A[5], 6);
  }    
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dView);

template <typename MemorySpace>
void TestArray2dViewResize(void)
{
  typedef typename cusp::array2d<int, MemorySpace>  Container;
  typedef typename cusp::array1d<int, MemorySpace>  Array;
  typedef typename Array::iterator                  Iterator;
  typedef typename cusp::array1d_view<Iterator>     ArrayView;
  typedef cusp::array2d_view<ArrayView>             View;

  // construct from array1d view
  {
    Array A(6);

    A[0] = 1;  A[1] = 2;  A[2] = 3;  A[3] = 4;  A[4] = 5;  A[5] = 6;

    View V(3, 2, 2, ArrayView(A));
    ASSERT_EQUAL(V.num_rows,          3);
    ASSERT_EQUAL(V.num_cols,          2);
    ASSERT_EQUAL(V.num_entries,       6);
    ASSERT_EQUAL(V.pitch,             2);
    ASSERT_EQUAL(V.values.size(),     6);
    ASSERT_EQUAL(V.values.capacity(), 6);
   
    ASSERT_EQUAL(V(0,0), 1); ASSERT_EQUAL(V(0,1), 2);
    ASSERT_EQUAL(V(1,0), 3); ASSERT_EQUAL(V(1,1), 4);

    V.resize(2,2);
    
    ASSERT_EQUAL(V.num_rows,          2);
    ASSERT_EQUAL(V.num_cols,          2);
    ASSERT_EQUAL(V.num_entries,       4);
    ASSERT_EQUAL(V.pitch,             2);
    ASSERT_EQUAL(V.values.size(),     4);
    ASSERT_EQUAL(V.values.capacity(), 6);
    
    ASSERT_EQUAL(V(0,0), 1); ASSERT_EQUAL(V(0,1), 2);
    ASSERT_EQUAL(V(1,0), 3); ASSERT_EQUAL(V(1,1), 4);
    
    V.resize(2,2,3);
    
    ASSERT_EQUAL(V.num_rows,          2);
    ASSERT_EQUAL(V.num_cols,          2);
    ASSERT_EQUAL(V.num_entries,       4);
    ASSERT_EQUAL(V.pitch,             3);
    ASSERT_EQUAL(V.values.size(),     6);
    ASSERT_EQUAL(V.values.capacity(), 6);
    
    ASSERT_EQUAL(V(0,0), 1); ASSERT_EQUAL(V(0,1), 2);
    ASSERT_EQUAL(V(1,0), 4); ASSERT_EQUAL(V(1,1), 5);

    ASSERT_THROWS(V.resize(4,2,2), cusp::not_implemented_exception);
  }    
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dViewResize);


template <typename MemorySpace>
void TestMakeArray2dView(void)
{
  typedef typename cusp::array2d<int, MemorySpace>    Container;
  typedef typename cusp::array1d<int, MemorySpace>    Array;
  typedef typename Array::iterator                    Iterator;
  typedef typename Array::const_iterator              ConstIterator;
  typedef typename cusp::array1d_view<Iterator>       ArrayView;
  typedef typename cusp::array1d_view<ConstIterator>  ConstArrayView;
  typedef cusp::array2d_view<ArrayView>               View;
  typedef cusp::array2d_view<ConstArrayView>          ConstView;

  // construct from array1d_view
  {
    Array A(6);

    View V = cusp::make_array2d_view(2, 3, 3, cusp::make_array1d_view(A), cusp::row_major());

    ASSERT_EQUAL(V.num_rows, 2);
    ASSERT_EQUAL(V.num_cols, 3);
    ASSERT_EQUAL(V.pitch,    3);
    ASSERT_EQUAL_QUIET(V.values.begin(), A.begin());
    
    // ensure view is mutable
    V(0,0) = 17;

    ASSERT_EQUAL(V(0,0), 17);
    ASSERT_EQUAL(A[0], 17);
    
    cusp::array2d_view<ArrayView,cusp::column_major> W = cusp::make_array2d_view(2, 3, 2, cusp::make_array1d_view(A), cusp::column_major());

    ASSERT_EQUAL(W.num_rows, 2);
    ASSERT_EQUAL(W.num_cols, 3);
    ASSERT_EQUAL(W.pitch,    2);
    ASSERT_EQUAL_QUIET(V.values.begin(), A.begin());
    
    // ensure view is mutable
    V(1,2) = 23;

    ASSERT_EQUAL(V(1,2), 23);
    ASSERT_EQUAL(A[5], 23);
  }

  // construct from array2d_view
  {
    Container A(2,3);

    View W(A);

    View V = cusp::make_array2d_view(W);

    ASSERT_EQUAL(V.num_rows, 2);
    ASSERT_EQUAL(V.num_cols, 3);
    ASSERT_EQUAL(V.pitch,    3);
    ASSERT_EQUAL_QUIET(V.values.begin(), A.values.begin());
    
    // ensure view is mutable
    V(0,0) = 17;

    ASSERT_EQUAL(V(0,0), 17);
    ASSERT_EQUAL(A(0,0), 17);

  }

  // construct from container
  {
    Container A(2,3);

    View V = cusp::make_array2d_view(A);

    ASSERT_EQUAL(V.num_rows, 2);
    ASSERT_EQUAL(V.num_cols, 3);
    ASSERT_EQUAL(V.pitch,    3);
    ASSERT_EQUAL_QUIET(V.values.begin(), A.values.begin());
    
    // ensure view is mutable
    V(0,0) = 17;

    ASSERT_EQUAL(V(0,0), 17);
    ASSERT_EQUAL(A(0,0), 17);

  }

  // construct from const container
  {
    const Container A(2,3);

    ConstView V = cusp::make_array2d_view(A);

    ASSERT_EQUAL(V.num_rows, 2);
    ASSERT_EQUAL(V.num_cols, 3);
    ASSERT_EQUAL(V.pitch,    3);
    ASSERT_EQUAL_QUIET(V.values.begin(), A.values.begin());
  }
}
DECLARE_HOST_DEVICE_UNITTEST(TestMakeArray2dView);


template <typename MemorySpace>
void TestArray2dViewEquality(void)
{
  typedef typename cusp::array2d<int, MemorySpace>  Container;
  typedef typename cusp::array1d<int, MemorySpace>  Array;
  typedef typename Array::iterator                  Iterator;
  typedef typename cusp::array1d_view<Iterator>     ArrayView;
  typedef cusp::array2d_view<ArrayView>             View;

  Container A(2,2);
  A(0,0) = 10;  A(0,1) = 20;
  A(1,0) = 30;  A(1,1) = 40;
  
  Container B(3,2);
  B(0,0) = 10;  B(0,1) = 20;
  B(1,0) = 30;  B(1,1) = 40;
  B(2,0) = 50;  B(2,1) = 60;

  View V(A);
  View W(B);

  ASSERT_EQUAL_QUIET(A == V, true);
  ASSERT_EQUAL_QUIET(V == A, true);
  ASSERT_EQUAL_QUIET(V == V, true);
  ASSERT_EQUAL_QUIET(A != V, false);
  ASSERT_EQUAL_QUIET(V != A, false);
  ASSERT_EQUAL_QUIET(V != V, false);

  ASSERT_EQUAL_QUIET(V == B, false);
  ASSERT_EQUAL_QUIET(B == V, false);
  ASSERT_EQUAL_QUIET(V == W, false);
  ASSERT_EQUAL_QUIET(W == V, false);
  ASSERT_EQUAL_QUIET(V != B, true);
  ASSERT_EQUAL_QUIET(B != V, true);
  ASSERT_EQUAL_QUIET(V != W, true);
  ASSERT_EQUAL_QUIET(W != V, true);

  W.resize(2,2);
  
  ASSERT_EQUAL_QUIET(V == W, true);
  ASSERT_EQUAL_QUIET(V != W, false);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dViewEquality);

