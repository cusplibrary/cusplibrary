#include <unittest/unittest.h>
#include <cusp/convert.h>

#include <cusp/print.h>

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::csr_matrix<IndexType, ValueType, Space> & csr)
{
    csr.resize(4, 4, 7);

    csr.row_offsets[0] = 0;
    csr.row_offsets[1] = 2;
    csr.row_offsets[2] = 3;
    csr.row_offsets[3] = 6;
    csr.row_offsets[4] = 7;
    
    csr.column_indices[0] = 0;   csr.values[0] = 10; 
    csr.column_indices[1] = 1;   csr.values[1] = 11;
    csr.column_indices[2] = 2;   csr.values[2] = 12;
    csr.column_indices[3] = 0;   csr.values[3] = 13;
    csr.column_indices[4] = 2;   csr.values[4] = 14; 
    csr.column_indices[5] = 3;   csr.values[5] = 15;
    csr.column_indices[6] = 1;   csr.values[6] = 16;
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::coo_matrix<IndexType, ValueType, Space> & coo)
{
    coo.resize(4, 4, 7);

    coo.row_indices[0] = 0;  coo.column_indices[0] = 0;  coo.values[0] = 10; 
    coo.row_indices[1] = 0;  coo.column_indices[1] = 1;  coo.values[1] = 11;
    coo.row_indices[2] = 1;  coo.column_indices[2] = 2;  coo.values[2] = 12;
    coo.row_indices[3] = 2;  coo.column_indices[3] = 0;  coo.values[3] = 13;
    coo.row_indices[4] = 2;  coo.column_indices[4] = 2;  coo.values[4] = 14; 
    coo.row_indices[5] = 2;  coo.column_indices[5] = 3;  coo.values[5] = 15;
    coo.row_indices[6] = 3;  coo.column_indices[6] = 1;  coo.values[6] = 16;
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::dia_matrix<IndexType, ValueType, Space> & dia)
{
    dia.resize(4, 4, 7, 3, 4);

    dia.diagonal_offsets[0] = -2;
    dia.diagonal_offsets[1] =  0;
    dia.diagonal_offsets[2] =  1;

    dia.values[ 0] =  0; 
    dia.values[ 1] =  0; 
    dia.values[ 2] = 13; 
    dia.values[ 3] = 16; 
    dia.values[ 4] = 10; 
    dia.values[ 5] =  0; 
    dia.values[ 6] = 14; 
    dia.values[ 7] =  0; 
    dia.values[ 8] = 11; 
    dia.values[ 9] = 12; 
    dia.values[10] = 15; 
    dia.values[11] =  0; 
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::ell_matrix<IndexType, ValueType, Space> & ell)
{
    ell.resize(4, 4, 7, 3, 4);

    const int X = cusp::ell_matrix<IndexType, ValueType, Space>::invalid_index;

    ell.column_indices[ 0] =  0;  ell.values[ 0] = 10; 
    ell.column_indices[ 1] =  2;  ell.values[ 1] = 12;
    ell.column_indices[ 2] =  0;  ell.values[ 2] = 13;
    ell.column_indices[ 3] =  1;  ell.values[ 3] = 16;
    
    ell.column_indices[ 4] =  1;  ell.values[ 4] = 11; 
    ell.column_indices[ 5] =  X;  ell.values[ 5] =  0; 
    ell.column_indices[ 6] =  2;  ell.values[ 6] = 14;
    ell.column_indices[ 7] =  X;  ell.values[ 7] =  0;

    ell.column_indices[ 8] =  X;  ell.values[ 8] =  0;
    ell.column_indices[ 9] =  X;  ell.values[ 9] =  0;
    ell.column_indices[10] =  3;  ell.values[10] = 15;
    ell.column_indices[11] =  X;  ell.values[11] =  0;
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::hyb_matrix<IndexType, ValueType, Space> & hyb)
{
    hyb.resize(4, 4, 4, 3, 1, 4); 

    hyb.ell.column_indices[0] = 0;  hyb.ell.values[0] = 10; 
    hyb.ell.column_indices[1] = 2;  hyb.ell.values[1] = 12;
    hyb.ell.column_indices[2] = 0;  hyb.ell.values[2] = 13;
    hyb.ell.column_indices[3] = 1;  hyb.ell.values[3] = 16;

    hyb.coo.row_indices[0] = 0; hyb.coo.column_indices[0] = 1; hyb.coo.values[0] = 11; 
    hyb.coo.row_indices[1] = 2; hyb.coo.column_indices[1] = 2; hyb.coo.values[1] = 14;
    hyb.coo.row_indices[2] = 2; hyb.coo.column_indices[2] = 3; hyb.coo.values[2] = 15;
}

template <typename ValueType, typename Space, class Orientation>
void initialize_conversion_example(cusp::array2d<ValueType, Space, Orientation> & dense)
{
    dense.resize(4, 4);

    dense(0,0) = 10;  dense(0,1) = 11;  dense(0,2) =  0;  dense(0,3) =  0;
    dense(1,0) =  0;  dense(1,1) =  0;  dense(1,2) = 12;  dense(1,3) =  0;
    dense(2,0) = 13;  dense(2,1) =  0;  dense(2,2) = 14;  dense(2,3) = 15;
    dense(3,0) =  0;  dense(3,1) = 16;  dense(3,2) =  0;  dense(3,3) =  0;
}

template <typename MatrixType>
void verify_conversion_example(const MatrixType& matrix)
{
    typedef typename MatrixType::value_type ValueType;

    cusp::array2d<ValueType, cusp::host_memory> dense(matrix);
   
    ASSERT_EQUAL(dense.num_rows,    4);
    ASSERT_EQUAL(dense.num_cols,    4);
    ASSERT_EQUAL(dense.num_entries, 16);

    ASSERT_EQUAL(dense(0,0), 10);  
    ASSERT_EQUAL(dense(0,1), 11);  
    ASSERT_EQUAL(dense(0,2),  0);  
    ASSERT_EQUAL(dense(0,3),  0);
    ASSERT_EQUAL(dense(1,0),  0);  
    ASSERT_EQUAL(dense(1,1),  0);
    ASSERT_EQUAL(dense(1,2), 12);
    ASSERT_EQUAL(dense(1,3),  0);
    ASSERT_EQUAL(dense(2,0), 13);
    ASSERT_EQUAL(dense(2,1),  0);
    ASSERT_EQUAL(dense(2,2), 14);
    ASSERT_EQUAL(dense(2,3), 15);
    ASSERT_EQUAL(dense(3,0),  0);
    ASSERT_EQUAL(dense(3,1), 16);
    ASSERT_EQUAL(dense(3,2),  0);
    ASSERT_EQUAL(dense(3,3),  0);
}


template <class HostDestinationType, class HostSourceType>
void TestConversion(HostDestinationType dst, HostSourceType src)
{
    typedef typename HostSourceType::template rebind<cusp::device_memory>::type      DeviceSourceType;
    typedef typename HostDestinationType::template rebind<cusp::device_memory>::type DeviceDestinationType;

    {
        // test host->host
        HostSourceType      src;
        initialize_conversion_example(src);

        {
            HostDestinationType dst(src);
            verify_conversion_example(dst);
        }
        {
            HostDestinationType dst;
            dst = src;
            verify_conversion_example(dst);
        }
        {
            HostDestinationType dst;
            cusp::convert(dst, src);
            verify_conversion_example(dst);
        }
    }
    
    {
        // test host->device
        HostSourceType        src;
        DeviceDestinationType dst;
        initialize_conversion_example(src);
        cusp::convert(dst, src);
        verify_conversion_example(dst);
    }
    
    {
        // test device->host
        DeviceSourceType    src;
        HostDestinationType dst;
        initialize_conversion_example(src);
        cusp::convert(dst, src);
        verify_conversion_example(dst);
    }
    
    {
        // test device->device
        DeviceSourceType      src;
        DeviceDestinationType dst;
        initialize_conversion_example(src);
        cusp::convert(dst, src);
        verify_conversion_example(dst);
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
void TestConversionToCooMatrix(void)
{
    TestConversionTo(cusp::coo_matrix<int, float, cusp::host_memory>());
}
DECLARE_UNITTEST(TestConversionToCooMatrix);

void TestConversionToCsrMatrix(void)
{
    TestConversionTo(cusp::csr_matrix<int, float, cusp::host_memory>());
}
DECLARE_UNITTEST(TestConversionToCsrMatrix);

void TestConversionToDiaMatrix(void)
{
    TestConversionTo(cusp::dia_matrix<int, float, cusp::host_memory>());
}
DECLARE_UNITTEST(TestConversionToDiaMatrix);

void TestConversionToEllMatrix(void)
{
    TestConversionTo(cusp::ell_matrix<int, float, cusp::host_memory>());
}
DECLARE_UNITTEST(TestConversionToEllMatrix);

void TestConversionToHybMatrix(void)
{
    TestConversionTo(cusp::hyb_matrix<int, float, cusp::host_memory>());
}
DECLARE_UNITTEST(TestConversionToHybMatrix);

void TestConversionToArray(void)
{
    TestConversionTo(cusp::array2d<float, cusp::host_memory>());
}
DECLARE_UNITTEST(TestConversionToArray);

//////////////////////////////
// Special Conversion Tests //
//////////////////////////////
void TestConvertCsrToDiaMatrix(void)
{
    cusp::csr_matrix<int, float, cusp::host_memory> csr;
    cusp::dia_matrix<int, float, cusp::host_memory> dia;

    // initialize host matrix
    initialize_conversion_example(csr);

    ASSERT_THROWS(cusp::detail::host::convert(dia, csr, 1.0, 4), cusp::format_conversion_exception);

    cusp::detail::host::convert(dia, csr, 3.0, 4);

    // compare csr and dia
    ASSERT_EQUAL(dia.num_rows,    csr.num_rows);
    ASSERT_EQUAL(dia.num_cols,    csr.num_cols);
    ASSERT_EQUAL(dia.num_entries, csr.num_entries);
    ASSERT_EQUAL(dia.num_diagonals, 3);

    ASSERT_EQUAL(dia.diagonal_offsets[ 0],  -2);
    ASSERT_EQUAL(dia.diagonal_offsets[ 1],   0);
    ASSERT_EQUAL(dia.diagonal_offsets[ 2],   1);

    ASSERT_EQUAL(dia.values[ 0],  0);
    ASSERT_EQUAL(dia.values[ 1],  0);
    ASSERT_EQUAL(dia.values[ 2], 13);
    ASSERT_EQUAL(dia.values[ 3], 16);
    
    ASSERT_EQUAL(dia.values[ 4], 10);
    ASSERT_EQUAL(dia.values[ 5],  0);
    ASSERT_EQUAL(dia.values[ 6], 14);
    ASSERT_EQUAL(dia.values[ 7],  0);
    
    ASSERT_EQUAL(dia.values[ 8], 11);
    ASSERT_EQUAL(dia.values[ 9], 12);
    ASSERT_EQUAL(dia.values[10], 15);
    ASSERT_EQUAL(dia.values[11],  0);
}
DECLARE_UNITTEST(TestConvertCsrToDiaMatrix);

void TestConvertCsrToEllMatrix(void)
{
    cusp::csr_matrix<int, float, cusp::host_memory> csr;
    cusp::ell_matrix<int, float, cusp::host_memory> ell;

    // initialize host matrix
    initialize_conversion_example(csr);

    ASSERT_THROWS(cusp::detail::host::convert(ell, csr, 1.0, 4), cusp::format_conversion_exception);

    cusp::detail::host::convert(ell, csr, 3.0, 4);

    const int X = cusp::ell_matrix<int, float, cusp::host_memory>::invalid_index;

    // compare csr and dia
    ASSERT_EQUAL(ell.num_rows,    csr.num_rows);
    ASSERT_EQUAL(ell.num_cols,    csr.num_cols);
    ASSERT_EQUAL(ell.num_entries, csr.num_entries);
    ASSERT_EQUAL(ell.num_entries_per_row, 3);
    ASSERT_EQUAL(ell.column_indices[ 0],  0);  ASSERT_EQUAL(ell.values[ 0], 10); 
    ASSERT_EQUAL(ell.column_indices[ 1],  2);  ASSERT_EQUAL(ell.values[ 1], 12);
    ASSERT_EQUAL(ell.column_indices[ 2],  0);  ASSERT_EQUAL(ell.values[ 2], 13);
    ASSERT_EQUAL(ell.column_indices[ 3],  1);  ASSERT_EQUAL(ell.values[ 3], 16);
    
    ASSERT_EQUAL(ell.column_indices[ 4],  1);  ASSERT_EQUAL(ell.values[ 4], 11); 
    ASSERT_EQUAL(ell.column_indices[ 5],  X);  ASSERT_EQUAL(ell.values[ 5],  0); 
    ASSERT_EQUAL(ell.column_indices[ 6],  2);  ASSERT_EQUAL(ell.values[ 6], 14);
    ASSERT_EQUAL(ell.column_indices[ 7],  X);  ASSERT_EQUAL(ell.values[ 7],  0);

    ASSERT_EQUAL(ell.column_indices[ 8],  X);  ASSERT_EQUAL(ell.values[ 8],  0);
    ASSERT_EQUAL(ell.column_indices[ 9],  X);  ASSERT_EQUAL(ell.values[ 9],  0);
    ASSERT_EQUAL(ell.column_indices[10],  3);  ASSERT_EQUAL(ell.values[10], 15);
    ASSERT_EQUAL(ell.column_indices[11],  X);  ASSERT_EQUAL(ell.values[11],  0);
}
DECLARE_UNITTEST(TestConvertCsrToEllMatrix);

