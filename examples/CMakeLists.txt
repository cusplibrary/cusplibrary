add_custom_target(all-examples)

function(add_example DIRNAME NAME)
  add_executable("${NAME}" "${DIRNAME}/${NAME}.cu")
  target_link_libraries("${NAME}" PUBLIC CCCL::CCCL)
  target_link_libraries("${NAME}" PUBLIC cusp)
  add_custom_target("run_${NAME}" COMMAND "${NAME}" VERBATIM)
  add_dependencies("run_${NAME}" "${NAME}")
  add_dependencies(all-examples "run_${NAME}")
endfunction()

add_example(Algorithms blas)
add_example(Algorithms multiply)
add_example(Algorithms transpose)

add_example(Gallery diffusion)
add_example(Gallery poisson)

# FIXME: add_example(Graph breadth_first_search)  # needs cudaTextureObject_t? CUDA v12 issue
# FIXME: add_example(Graph connected_components)  # needs cudaTextureObject_t? CUDA v12 issue
add_example(Graph hilbert_curve)
add_example(Graph maximal_independent_set)
# FIXME: add_example(Graph pseudo_peripheral)  # needs cudaTextureObject_t? CUDA v12 issue
# FIXME: add_example(Graph symmetric_rcm)  # needs cudaTextureObject_t? CUDA v12 issue
add_example(Graph vertex_coloring)

add_example(InputOutput matrix_market)

# FIXME: add_example(LinearOperator block_matrix)
# FIXME: add_example(LinearOperator stencil)

add_example(MatrixAssembly unordered_triplets)

add_example(MatrixFormats coo)
add_example(MatrixFormats csr)
# FIXME: add_example(MatrixFormats dia)
add_example(MatrixFormats ell)
# FIXME: add_example(MatrixFormats hyb)

add_example(Monitors monitor)
add_example(Monitors verbose_monitor)

# FIXME: add_example(OpenGL spy)

# FIXME: add_example(Preconditioners ainv)
# FIXME: add_example(Preconditioners custom_amg)
# FIXME: add_example(Preconditioners diagonal)
# FIXME: add_example(Preconditioners smoothed_aggregation)
# FIXME: file(COPY ${CMAKE_CURRENT_LIST_DIR}/Preconditioners/A.mtx DESTINATION ${CMAKE_BINARY_DIR}/examples/Preconditioners/A.mtx)

add_example(Solvers bicgstab)
add_example(Solvers bicgstab_m)
add_example(Solvers cg)
add_example(Solvers cg_m)
add_example(Solvers cr)
add_example(Solvers gmres)

add_example(Version version)

add_example(Views array1d)
add_example(Views array2d_raw)
add_example(Views cg_raw)
add_example(Views csr_raw)
add_example(Views csr_view)
