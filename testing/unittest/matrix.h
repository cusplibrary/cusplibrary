#pragma once

// Macro to create host and device versions of a unit test
#define DECLARE_HOST_DEVICE_UNITTEST(VTEST)                    \
void VTEST##Host(void)   {  VTEST< cusp::host_memory   >(); }  \
void VTEST##Device(void) {  VTEST< cusp::device_memory >(); }  \
DECLARE_UNITTEST(VTEST##Host);                                 \
DECLARE_UNITTEST(VTEST##Device);

// Macro to create host and device versions of a unit test
#define DECLARE_HOST_SPARSE_MATRIX_UNITTEST(VTEST)                                                                                                                \
void VTEST##CooMatrixHost(void)   { VTEST< cusp::coo_matrix<int,float,cusp::host_memory> >();  VTEST< cusp::coo_matrix<long long,float,cusp::host_memory> >(); }  \
void VTEST##CsrMatrixHost(void)   { VTEST< cusp::csr_matrix<int,float,cusp::host_memory> >();  VTEST< cusp::csr_matrix<long long,float,cusp::host_memory> >(); }  \
void VTEST##DiaMatrixHost(void)   { VTEST< cusp::dia_matrix<int,float,cusp::host_memory> >();  VTEST< cusp::dia_matrix<long long,float,cusp::host_memory> >(); }  \
void VTEST##EllMatrixHost(void)   { VTEST< cusp::ell_matrix<int,float,cusp::host_memory> >();  VTEST< cusp::ell_matrix<long long,float,cusp::host_memory> >(); }  \
void VTEST##HybMatrixHost(void)   { VTEST< cusp::hyb_matrix<int,float,cusp::host_memory> >();  VTEST< cusp::hyb_matrix<long long,float,cusp::host_memory> >(); }  \
DECLARE_UNITTEST(VTEST##CooMatrixHost);                                                                                                                           \
DECLARE_UNITTEST(VTEST##CsrMatrixHost);                                                                                                                           \
DECLARE_UNITTEST(VTEST##DiaMatrixHost);                                                                                                                           \
DECLARE_UNITTEST(VTEST##EllMatrixHost);                                                                                                                           \
DECLARE_UNITTEST(VTEST##HybMatrixHost);                                                              

#define DECLARE_DEVICE_SPARSE_MATRIX_UNITTEST(VTEST)                                                                                                                 \
void VTEST##CooMatrixDevice(void) { VTEST< cusp::coo_matrix<int,float,cusp::device_memory> >(); VTEST< cusp::coo_matrix<long long,float,cusp::device_memory> >(); }  \
void VTEST##CsrMatrixDevice(void) { VTEST< cusp::csr_matrix<int,float,cusp::device_memory> >(); VTEST< cusp::csr_matrix<long long,float,cusp::device_memory> >(); }  \
void VTEST##DiaMatrixDevice(void) { VTEST< cusp::dia_matrix<int,float,cusp::device_memory> >(); VTEST< cusp::dia_matrix<long long,float,cusp::device_memory> >(); }  \
void VTEST##EllMatrixDevice(void) { VTEST< cusp::ell_matrix<int,float,cusp::device_memory> >(); VTEST< cusp::ell_matrix<long long,float,cusp::device_memory> >(); }  \
void VTEST##HybMatrixDevice(void) { VTEST< cusp::hyb_matrix<int,float,cusp::device_memory> >(); VTEST< cusp::hyb_matrix<long long,float,cusp::device_memory> >(); }  \
DECLARE_UNITTEST(VTEST##CooMatrixDevice);                                                                                                                            \
DECLARE_UNITTEST(VTEST##CsrMatrixDevice);                                                                                                                            \
DECLARE_UNITTEST(VTEST##DiaMatrixDevice);                                                                                                                            \
DECLARE_UNITTEST(VTEST##EllMatrixDevice);                                                                                                                            \
DECLARE_UNITTEST(VTEST##HybMatrixDevice);                                                            

#define DECLARE_SPARSE_MATRIX_UNITTEST(VTEST) \
DECLARE_HOST_SPARSE_MATRIX_UNITTEST(VTEST)    \
DECLARE_DEVICE_SPARSE_MATRIX_UNITTEST(VTEST)

