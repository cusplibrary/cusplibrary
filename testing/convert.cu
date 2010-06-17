#include <unittest/unittest.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/verify.h>

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::csr_matrix<IndexType, ValueType, Space> & csr)
{
    csr.resize(4, 4, 7);

    csr.row_offsets[0] = 0;
    csr.row_offsets[1] = 2;
    csr.row_offsets[2] = 3;
    csr.row_offsets[3] = 6;
    csr.row_offsets[4] = 7;
    
    csr.column_indices[0] = 0;   csr.values[0] = 10.25; 
    csr.column_indices[1] = 1;   csr.values[1] = 11.00;
    csr.column_indices[2] = 2;   csr.values[2] = 12.50;
    csr.column_indices[3] = 0;   csr.values[3] = 13.75;
    csr.column_indices[4] = 2;   csr.values[4] = 14.00; 
    csr.column_indices[5] = 3;   csr.values[5] = 15.25;
    csr.column_indices[6] = 1;   csr.values[6] = 16.50;
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::coo_matrix<IndexType, ValueType, Space> & coo)
{
    coo.resize(4, 4, 7);

    coo.row_indices[0] = 0;  coo.column_indices[0] = 0;  coo.values[0] = 10.25; 
    coo.row_indices[1] = 0;  coo.column_indices[1] = 1;  coo.values[1] = 11.00;
    coo.row_indices[2] = 1;  coo.column_indices[2] = 2;  coo.values[2] = 12.50;
    coo.row_indices[3] = 2;  coo.column_indices[3] = 0;  coo.values[3] = 13.75;
    coo.row_indices[4] = 2;  coo.column_indices[4] = 2;  coo.values[4] = 14.00; 
    coo.row_indices[5] = 2;  coo.column_indices[5] = 3;  coo.values[5] = 15.25;
    coo.row_indices[6] = 3;  coo.column_indices[6] = 1;  coo.values[6] = 16.50;
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::dia_matrix<IndexType, ValueType, Space> & dia)
{
    dia.resize(4, 4, 7, 3, 1);

    dia.diagonal_offsets[0] = -2;
    dia.diagonal_offsets[1] =  0;
    dia.diagonal_offsets[2] =  1;

    dia.values.values[ 0] =  0.00; 
    dia.values.values[ 1] =  0.00; 
    dia.values.values[ 2] = 13.75; 
    dia.values.values[ 3] = 16.50; 
    dia.values.values[ 4] = 10.25; 
    dia.values.values[ 5] =  0.00; 
    dia.values.values[ 6] = 14.00; 
    dia.values.values[ 7] =  0.00; 
    dia.values.values[ 8] = 11.00; 
    dia.values.values[ 9] = 12.50; 
    dia.values.values[10] = 15.25; 
    dia.values.values[11] =  0.00; 
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::ell_matrix<IndexType, ValueType, Space> & ell)
{
    ell.resize(4, 4, 7, 3, 1);

    const int X = cusp::ell_matrix<IndexType, ValueType, Space>::invalid_index;

    ell.column_indices.values[ 0] =  0;  ell.values.values[ 0] = 10.25; 
    ell.column_indices.values[ 1] =  2;  ell.values.values[ 1] = 12.50;
    ell.column_indices.values[ 2] =  0;  ell.values.values[ 2] = 13.75;
    ell.column_indices.values[ 3] =  1;  ell.values.values[ 3] = 16.50;
    
    ell.column_indices.values[ 4] =  1;  ell.values.values[ 4] = 11.00; 
    ell.column_indices.values[ 5] =  X;  ell.values.values[ 5] =  0.00; 
    ell.column_indices.values[ 6] =  2;  ell.values.values[ 6] = 14.00;
    ell.column_indices.values[ 7] =  X;  ell.values.values[ 7] =  0.00;

    ell.column_indices.values[ 8] =  X;  ell.values.values[ 8] =  0.00;
    ell.column_indices.values[ 9] =  X;  ell.values.values[ 9] =  0.00;
    ell.column_indices.values[10] =  3;  ell.values.values[10] = 15.25;
    ell.column_indices.values[11] =  X;  ell.values.values[11] =  0.00;
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::hyb_matrix<IndexType, ValueType, Space> & hyb)
{
    hyb.resize(4, 4, 4, 3, 1, 1); 

    hyb.ell.column_indices.values[0] = 0;  hyb.ell.values.values[0] = 10.25; 
    hyb.ell.column_indices.values[1] = 2;  hyb.ell.values.values[1] = 12.50;
    hyb.ell.column_indices.values[2] = 0;  hyb.ell.values.values[2] = 13.75;
    hyb.ell.column_indices.values[3] = 1;  hyb.ell.values.values[3] = 16.50;

    hyb.coo.row_indices[0] = 0; hyb.coo.column_indices[0] = 1; hyb.coo.values[0] = 11.00; 
    hyb.coo.row_indices[1] = 2; hyb.coo.column_indices[1] = 2; hyb.coo.values[1] = 14.00;
    hyb.coo.row_indices[2] = 2; hyb.coo.column_indices[2] = 3; hyb.coo.values[2] = 15.25;
}

template <typename ValueType, typename Space, class Orientation>
void initialize_conversion_example(cusp::array2d<ValueType, Space, Orientation> & dense)
{
    dense.resize(4, 4);

    dense(0,0) = 10.25;  dense(0,1) = 11.00;  dense(0,2) =  0.00;  dense(0,3) =  0.00;
    dense(1,0) =  0.00;  dense(1,1) =  0.00;  dense(1,2) = 12.50;  dense(1,3) =  0.00;
    dense(2,0) = 13.75;  dense(2,1) =  0.00;  dense(2,2) = 14.00;  dense(2,3) = 15.25;
    dense(3,0) =  0.00;  dense(3,1) = 16.50;  dense(3,2) =  0.00;  dense(3,3) =  0.00;
}

template <typename MatrixType>
void verify_conversion_example(const MatrixType& matrix)
{
    typedef typename MatrixType::value_type ValueType;

    cusp::array2d<ValueType, cusp::host_memory> dense(matrix);
   
    ASSERT_EQUAL(dense.num_rows,    4);
    ASSERT_EQUAL(dense.num_cols,    4);
    ASSERT_EQUAL(dense.num_entries, 16);

    ASSERT_EQUAL(dense(0,0), 10.25);  
    ASSERT_EQUAL(dense(0,1), 11.00);  
    ASSERT_EQUAL(dense(0,2),  0.00);  
    ASSERT_EQUAL(dense(0,3),  0.00);
    ASSERT_EQUAL(dense(1,0),  0.00);  
    ASSERT_EQUAL(dense(1,1),  0.00);
    ASSERT_EQUAL(dense(1,2), 12.50);
    ASSERT_EQUAL(dense(1,3),  0.00);
    ASSERT_EQUAL(dense(2,0), 13.75);
    ASSERT_EQUAL(dense(2,1),  0.00);
    ASSERT_EQUAL(dense(2,2), 14.00);
    ASSERT_EQUAL(dense(2,3), 15.25);
    ASSERT_EQUAL(dense(3,0),  0.00);
    ASSERT_EQUAL(dense(3,1), 16.50);
    ASSERT_EQUAL(dense(3,2),  0.00);
    ASSERT_EQUAL(dense(3,3),  0.00);
}


template <class DestinationType, class HostSourceType>
void TestConversion(DestinationType dst, HostSourceType src)
{
    typedef typename HostSourceType::template rebind<cusp::device_memory>::type DeviceSourceType;
    
    {
        HostSourceType src;

        initialize_conversion_example(src);

        {  DestinationType dst(src);              verify_conversion_example(dst);  cusp::assert_is_valid_matrix(dst); }
        {  DestinationType dst;       dst = src;  verify_conversion_example(dst);  cusp::assert_is_valid_matrix(dst); }
    }

    {
        DeviceSourceType src;

        initialize_conversion_example(src);

        {  DestinationType dst(src);              verify_conversion_example(dst);  cusp::assert_is_valid_matrix(dst); }
        {  DestinationType dst;       dst = src;  verify_conversion_example(dst);  cusp::assert_is_valid_matrix(dst); }
    }
    
}


template <typename DestinationMatrixType>
void TestConversionTo(DestinationMatrixType dst)
{
    TestConversion(DestinationMatrixType(), 
                   cusp::coo_matrix<int, float, cusp::host_memory>());
    TestConversion(DestinationMatrixType(), 
                   cusp::csr_matrix<int, float, cusp::host_memory>());
    TestConversion(DestinationMatrixType(), 
                   cusp::dia_matrix<int, float, cusp::host_memory>());
    TestConversion(DestinationMatrixType(), 
                   cusp::ell_matrix<int, float, cusp::host_memory>());
    TestConversion(DestinationMatrixType(), 
                   cusp::hyb_matrix<int, float, cusp::host_memory>());
    TestConversion(DestinationMatrixType(), 
                   cusp::array2d<float, cusp::host_memory, cusp::row_major>());
    TestConversion(DestinationMatrixType(), 
                   cusp::array2d<float, cusp::host_memory, cusp::column_major>());
}

///////////////////////////
// Main Conversion Tests //
///////////////////////////
template <class SparseMatrix>
void TestConversionTo(void)
{
    TestConversionTo(SparseMatrix());
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestConversionTo);

template <class Space>
void TestConversionToArray2d(void)
{
    TestConversionTo(cusp::array2d<float, Space, cusp::row_major>());
    TestConversionTo(cusp::array2d<float, Space, cusp::column_major>());
}
DECLARE_HOST_DEVICE_UNITTEST(TestConversionToArray2d);

//////////////////////////////
// Special Conversion Tests //
//////////////////////////////
void TestConvertCsrToDiaMatrixHost(void)
{
    cusp::csr_matrix<int, float, cusp::host_memory> csr;
    cusp::dia_matrix<int, float, cusp::host_memory> dia;

    // initialize host matrix
    initialize_conversion_example(csr);

    //ASSERT_THROWS(cusp::detail::host::convert(csr, dia, 1.0, 4), cusp::format_conversion_exception);

    cusp::detail::host::convert(csr, dia, 3.0, 4);

    // compare csr and dia
    ASSERT_EQUAL(dia.num_rows,    csr.num_rows);
    ASSERT_EQUAL(dia.num_cols,    csr.num_cols);
    ASSERT_EQUAL(dia.num_entries, csr.num_entries);

    ASSERT_EQUAL(dia.diagonal_offsets[ 0],  -2);
    ASSERT_EQUAL(dia.diagonal_offsets[ 1],   0);
    ASSERT_EQUAL(dia.diagonal_offsets[ 2],   1);

    ASSERT_EQUAL(dia.values(0,0),  0.00);
    ASSERT_EQUAL(dia.values(1,0),  0.00);
    ASSERT_EQUAL(dia.values(2,0), 13.75);
    ASSERT_EQUAL(dia.values(3,0), 16.50);
    
    ASSERT_EQUAL(dia.values(0,1), 10.25);
    ASSERT_EQUAL(dia.values(1,1),  0.00);
    ASSERT_EQUAL(dia.values(2,1), 14.00);
    ASSERT_EQUAL(dia.values(3,1),  0.00);
    
    ASSERT_EQUAL(dia.values(0,2), 11.00);
    ASSERT_EQUAL(dia.values(1,2), 12.50);
    ASSERT_EQUAL(dia.values(2,2), 15.25);
    ASSERT_EQUAL(dia.values(3,2),  0.00);

    cusp::assert_is_valid_matrix(dia);
}
DECLARE_UNITTEST(TestConvertCsrToDiaMatrixHost);

void TestConvertCsrToEllMatrixHost(void)
{
    cusp::csr_matrix<int, float, cusp::host_memory> csr;
    cusp::ell_matrix<int, float, cusp::host_memory> ell;

    // initialize host matrix
    initialize_conversion_example(csr);

    //ASSERT_THROWS(cusp::detail::host::convert(csr, ell, 1.0, 1), cusp::format_conversion_exception);

    cusp::detail::host::convert(csr, ell, 3.0, 1);

    const int X = cusp::ell_matrix<int, float, cusp::host_memory>::invalid_index;

    // compare csr and dia
    ASSERT_EQUAL(ell.num_rows,    csr.num_rows);
    ASSERT_EQUAL(ell.num_cols,    csr.num_cols);
    ASSERT_EQUAL(ell.num_entries, csr.num_entries);
    ASSERT_EQUAL(ell.column_indices.num_rows, 4);
    ASSERT_EQUAL(ell.column_indices.num_cols, 3);
    ASSERT_EQUAL(ell.column_indices.values[ 0],  0);  ASSERT_EQUAL(ell.values.values[ 0], 10.25); 
    ASSERT_EQUAL(ell.column_indices.values[ 1],  2);  ASSERT_EQUAL(ell.values.values[ 1], 12.50);
    ASSERT_EQUAL(ell.column_indices.values[ 2],  0);  ASSERT_EQUAL(ell.values.values[ 2], 13.75);
    ASSERT_EQUAL(ell.column_indices.values[ 3],  1);  ASSERT_EQUAL(ell.values.values[ 3], 16.50);
                                                                                               
    ASSERT_EQUAL(ell.column_indices.values[ 4],  1);  ASSERT_EQUAL(ell.values.values[ 4], 11.00); 
    ASSERT_EQUAL(ell.column_indices.values[ 5],  X);  ASSERT_EQUAL(ell.values.values[ 5],  0.00); 
    ASSERT_EQUAL(ell.column_indices.values[ 6],  2);  ASSERT_EQUAL(ell.values.values[ 6], 14.00);
    ASSERT_EQUAL(ell.column_indices.values[ 7],  X);  ASSERT_EQUAL(ell.values.values[ 7],  0.00);
                                                                                               
    ASSERT_EQUAL(ell.column_indices.values[ 8],  X);  ASSERT_EQUAL(ell.values.values[ 8],  0.00);
    ASSERT_EQUAL(ell.column_indices.values[ 9],  X);  ASSERT_EQUAL(ell.values.values[ 9],  0.00);
    ASSERT_EQUAL(ell.column_indices.values[10],  3);  ASSERT_EQUAL(ell.values.values[10], 15.25);
    ASSERT_EQUAL(ell.column_indices.values[11],  X);  ASSERT_EQUAL(ell.values.values[11],  0.00);
    
    cusp::assert_is_valid_matrix(ell);
}
DECLARE_UNITTEST(TestConvertCsrToEllMatrixHost);

