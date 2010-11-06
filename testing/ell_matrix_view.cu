#include <unittest/unittest.h>

#include <cusp/ell_matrix.h>
#include <cusp/multiply.h>

template <typename MemorySpace>
void TestEllMatrixView(void)
{
  typedef int                                                           IndexType;
  typedef float                                                         ValueType;
  typedef typename cusp::ell_matrix<IndexType,ValueType,MemorySpace>    Matrix;
  typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
  typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
  typedef typename cusp::array1d_view<IndexIterator>                    IndexView1d;
  typedef typename cusp::array1d_view<ValueIterator>                    ValueView1d;
  typedef typename cusp::array2d_view<IndexView1d,cusp::column_major>   IndexView2d;
  typedef typename cusp::array2d_view<ValueView1d,cusp::column_major>   ValueView2d;
  typedef typename cusp::ell_matrix_view<IndexView2d,ValueView2d>       View;

  Matrix M(3, 2, 6, 2);

  View V(3, 2, 6,
      cusp::make_array2d_view(M.column_indices),
      cusp::make_array2d_view(M.values));

  ASSERT_EQUAL(V.num_rows,    3);
  ASSERT_EQUAL(V.num_cols,    2);
  ASSERT_EQUAL(V.num_entries, 6);
  
  ASSERT_EQUAL_QUIET(V.column_indices.num_rows,       M.column_indices.num_rows);
  ASSERT_EQUAL_QUIET(V.column_indices.num_cols,       M.column_indices.num_cols);
  ASSERT_EQUAL_QUIET(V.column_indices.num_entries,    M.column_indices.num_entries);
  ASSERT_EQUAL_QUIET(V.column_indices.pitch,          M.column_indices.pitch);
  ASSERT_EQUAL_QUIET(V.column_indices.values.begin(), M.column_indices.values.begin());
  ASSERT_EQUAL_QUIET(V.column_indices.values.end(),   M.column_indices.values.end());
  ASSERT_EQUAL_QUIET(V.values.num_rows,               M.values.num_rows);
  ASSERT_EQUAL_QUIET(V.values.num_cols,               M.values.num_cols);
  ASSERT_EQUAL_QUIET(V.values.num_entries,            M.values.num_entries);
  ASSERT_EQUAL_QUIET(V.values.pitch,                  M.values.pitch);
  ASSERT_EQUAL_QUIET(V.values.values.begin(),         M.values.values.begin());
  ASSERT_EQUAL_QUIET(V.values.values.end(),           M.values.values.end());
  
  View W(M);
  
  ASSERT_EQUAL(W.num_rows,    3);
  ASSERT_EQUAL(W.num_cols,    2);
  ASSERT_EQUAL(W.num_entries, 6);

  ASSERT_EQUAL_QUIET(W.column_indices.num_rows,       M.column_indices.num_rows);
  ASSERT_EQUAL_QUIET(W.column_indices.num_cols,       M.column_indices.num_cols);
  ASSERT_EQUAL_QUIET(W.column_indices.num_entries,    M.column_indices.num_entries);
  ASSERT_EQUAL_QUIET(W.column_indices.pitch,          M.column_indices.pitch);
  ASSERT_EQUAL_QUIET(W.column_indices.values.begin(), M.column_indices.values.begin());
  ASSERT_EQUAL_QUIET(W.column_indices.values.end(),   M.column_indices.values.end());
  ASSERT_EQUAL_QUIET(W.values.num_rows,               M.values.num_rows);
  ASSERT_EQUAL_QUIET(W.values.num_cols,               M.values.num_cols);
  ASSERT_EQUAL_QUIET(W.values.num_entries,            M.values.num_entries);
  ASSERT_EQUAL_QUIET(W.values.pitch,                  M.values.pitch);
  ASSERT_EQUAL_QUIET(W.values.values.begin(),         M.values.values.begin());
  ASSERT_EQUAL_QUIET(W.values.values.end(),           M.values.values.end());
}
DECLARE_HOST_DEVICE_UNITTEST(TestEllMatrixView);


template <typename MemorySpace>
void TestEllMatrixViewAssignment(void)
{
  typedef int                                                           IndexType;
  typedef float                                                         ValueType;
  typedef typename cusp::ell_matrix<IndexType,ValueType,MemorySpace>    Matrix;
  typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
  typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
  typedef typename cusp::array1d_view<IndexIterator>                    IndexView1d;
  typedef typename cusp::array1d_view<ValueIterator>                    ValueView1d;
  typedef typename cusp::array2d_view<IndexView1d,cusp::column_major>   IndexView2d;
  typedef typename cusp::array2d_view<ValueView1d,cusp::column_major>   ValueView2d;
  typedef typename cusp::ell_matrix_view<IndexView2d,ValueView2d>       View;

  Matrix M(3, 2, 6, 2);

  View V = M;
  
  ASSERT_EQUAL(V.num_rows,    3);
  ASSERT_EQUAL(V.num_cols,    2);
  ASSERT_EQUAL(V.num_entries, 6);

  ASSERT_EQUAL_QUIET(V.column_indices.num_rows,       M.column_indices.num_rows);
  ASSERT_EQUAL_QUIET(V.column_indices.num_cols,       M.column_indices.num_cols);
  ASSERT_EQUAL_QUIET(V.column_indices.num_entries,    M.column_indices.num_entries);
  ASSERT_EQUAL_QUIET(V.column_indices.pitch,          M.column_indices.pitch);
  ASSERT_EQUAL_QUIET(V.column_indices.values.begin(), M.column_indices.values.begin());
  ASSERT_EQUAL_QUIET(V.column_indices.values.end(),   M.column_indices.values.end());
  ASSERT_EQUAL_QUIET(V.values.num_rows,               M.values.num_rows);
  ASSERT_EQUAL_QUIET(V.values.num_cols,               M.values.num_cols);
  ASSERT_EQUAL_QUIET(V.values.num_entries,            M.values.num_entries);
  ASSERT_EQUAL_QUIET(V.values.pitch,                  M.values.pitch);
  ASSERT_EQUAL_QUIET(V.values.values.begin(),         M.values.values.begin());
  ASSERT_EQUAL_QUIET(V.values.values.end(),           M.values.values.end());

  View W = V;
  
  ASSERT_EQUAL(W.num_rows,    3);
  ASSERT_EQUAL(W.num_cols,    2);
  ASSERT_EQUAL(W.num_entries, 6);

  ASSERT_EQUAL_QUIET(W.column_indices.num_rows,       M.column_indices.num_rows);
  ASSERT_EQUAL_QUIET(W.column_indices.num_cols,       M.column_indices.num_cols);
  ASSERT_EQUAL_QUIET(W.column_indices.num_entries,    M.column_indices.num_entries);
  ASSERT_EQUAL_QUIET(W.column_indices.pitch,          M.column_indices.pitch);
  ASSERT_EQUAL_QUIET(W.column_indices.values.begin(), M.column_indices.values.begin());
  ASSERT_EQUAL_QUIET(W.column_indices.values.end(),   M.column_indices.values.end());
  ASSERT_EQUAL_QUIET(W.values.num_rows,               M.values.num_rows);
  ASSERT_EQUAL_QUIET(W.values.num_cols,               M.values.num_cols);
  ASSERT_EQUAL_QUIET(W.values.num_entries,            M.values.num_entries);
  ASSERT_EQUAL_QUIET(W.values.pitch,                  M.values.pitch);
  ASSERT_EQUAL_QUIET(W.values.values.begin(),         M.values.values.begin());
  ASSERT_EQUAL_QUIET(W.values.values.end(),           M.values.values.end());
  
}
DECLARE_HOST_DEVICE_UNITTEST(TestEllMatrixViewAssignment);


template <typename MemorySpace>
void TestMakeEllMatrixView(void)
{
  typedef int                                                           IndexType;
  typedef float                                                         ValueType;
  typedef typename cusp::ell_matrix<IndexType,ValueType,MemorySpace>    Matrix;
  typedef typename cusp::array1d<IndexType,MemorySpace>::iterator       IndexIterator;
  typedef typename cusp::array1d<ValueType,MemorySpace>::iterator       ValueIterator;
  typedef typename cusp::array1d_view<IndexIterator>                    IndexView1d;
  typedef typename cusp::array1d_view<ValueIterator>                    ValueView1d;
  typedef typename cusp::array2d_view<IndexView1d,cusp::column_major>   IndexView2d;
  typedef typename cusp::array2d_view<ValueView1d,cusp::column_major>   ValueView2d;
  typedef typename cusp::ell_matrix_view<IndexView2d,ValueView2d>       View;

  // construct view from parts
  {
    Matrix M(3, 2, 6, 2);

    View V =
      cusp::make_ell_matrix_view(3, 2, 6,
          cusp::make_array2d_view(M.column_indices),
          cusp::make_array2d_view(M.values));
    
    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    2);
    ASSERT_EQUAL(V.num_entries, 6);

    ASSERT_EQUAL_QUIET(V.column_indices,   M.column_indices);
    ASSERT_EQUAL_QUIET(V.values,           M.values);
  }
  
  // construct view from matrix
  {
    Matrix M(3, 2, 6, 2);

    View V = cusp::make_ell_matrix_view(M);
    
    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    2);
    ASSERT_EQUAL(V.num_entries, 6);
    
    ASSERT_EQUAL_QUIET(V.column_indices,   M.column_indices);
    ASSERT_EQUAL_QUIET(V.values,           M.values);
  }
  
  // construct view from view
  {
    Matrix M(3, 2, 6, 2);

    View X = cusp::make_ell_matrix_view(M);
    View V = cusp::make_ell_matrix_view(X);
    
    ASSERT_EQUAL(V.num_rows,    3);
    ASSERT_EQUAL(V.num_cols,    2);
    ASSERT_EQUAL(V.num_entries, 6);

    ASSERT_EQUAL_QUIET(V.column_indices,   M.column_indices);
    ASSERT_EQUAL_QUIET(V.values,           M.values);
  }
 
//  // construct view from const matrix
//  {
//    const Matrix M(3, 2, 6);
//    
//    ASSERT_EQUAL(cusp::make_ell_matrix_view(M).num_rows,    3);
//    ASSERT_EQUAL(cusp::make_ell_matrix_view(M).num_cols,    2);
//    ASSERT_EQUAL(cusp::make_ell_matrix_view(M).num_entries, 6);
//
//    ASSERT_EQUAL_QUIET(cusp::make_ell_matrix_view(M).row_offsets.begin(),    M.row_offsets.begin());
//    ASSERT_EQUAL_QUIET(cusp::make_ell_matrix_view(M).row_offsets.end(),      M.row_offsets.end());
//    ASSERT_EQUAL_QUIET(cusp::make_ell_matrix_view(M).column_indices.begin(), M.column_indices.begin());
//    ASSERT_EQUAL_QUIET(cusp::make_ell_matrix_view(M).column_indices.end(),   M.column_indices.end());
//    ASSERT_EQUAL_QUIET(cusp::make_ell_matrix_view(M).values.begin(),         M.values.begin());
//    ASSERT_EQUAL_QUIET(cusp::make_ell_matrix_view(M).values.end(),           M.values.end());
//  }
}
DECLARE_HOST_DEVICE_UNITTEST(TestMakeEllMatrixView);

