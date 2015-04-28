EnsureSConsVersion(1, 2)

import os

import inspect
import platform
import subprocess
from distutils.version import StrictVersion

def get_cuda_paths():
    """Determines CUDA {bin,lib,include} paths

    returns (bin_path,lib_path,inc_path)
    """

    # determine defaults
    if os.name == 'nt':
        bin_path = 'C:/CUDA/bin'
        lib_path = 'C:/CUDA/lib'
        inc_path = 'C:/CUDA/include'
    elif os.name == 'posix':
        bin_path = '/usr/local/cuda/bin'
        lib_path = '/usr/local/cuda/lib'
        inc_path = '/usr/local/cuda/include'
    else:
        raise ValueError, 'Error: unknown OS.  Where is nvcc installed?'

    if platform.machine()[-2:] == '64' and platform.platform()[:6] != 'Darwin':
        lib_path += '64'

    # override with environement variables
    if 'CUDA_BIN_PATH' in os.environ:
        bin_path = os.path.abspath(os.environ['CUDA_BIN_PATH'])
    if 'CUDA_LIB_PATH' in os.environ:
        lib_path = os.path.abspath(os.environ['CUDA_LIB_PATH'])
    if 'CUDA_INC_PATH' in os.environ:
        inc_path = os.path.abspath(os.environ['CUDA_INC_PATH'])

    return (bin_path, lib_path, inc_path)


def get_mkl_paths():
    """Determines MKL {lib,include} paths

    returns (lib_path,inc_path)
    """

    arch64 = False
    if platform.machine()[-2:] == '64':
        arch64 = True

    if 'MKLROOT' not in os.environ:
        raise ValueError, "MKLROOT is not an environment variable"

    # determine defaults
    if os.name == 'nt':
        raise ValueError, "Intel MKL support for Windows not implemented."
    elif os.name == 'posix':
        lib_base = os.environ['MKLROOT'] + '/lib'
        dirs = os.listdir(lib_base)
        for dir in dirs:
            # select 64/32 bit MKL library path based on architecture
            if arch64 == True and dir.find('64') > -1:
                lib_path = lib_base + '/' + dir
                break
            elif arch64 == False and dir.find('64') == -1:
                lib_path = lib_base + '/' + dir
                break

        if lib_path == lib_base:
            raise ValueError, 'Could not find MKL library directory which matches the arctitecture.'

        inc_path = os.environ['MKLROOT'] + '/include'
    else:
        raise ValueError, 'Error: unknown OS.  Where is nvcc installed?'

    return (lib_path, inc_path)


def getTools():
    result = []
    if os.name == 'nt':
        result = ['default', 'msvc']
    elif os.name == 'posix':
        result = ['default', 'gcc']
    else:
        result = ['default']
    return result


OldEnvironment = Environment


# this dictionary maps the name of a compiler program to a dictionary mapping the name of
# a compiler switch of interest to the specific switch implementing the feature
gCompilerOptions = {
        'gcc': {'warn_all': '-Wall', 'warn_errors': '-Werror', 'optimization': '-O3', 'debug': '-g',  'exception_handling': '',      'omp': '-fopenmp', 'coverage':['-O0', '-coverage']},
        'g++': {'warn_all': '-Wall', 'warn_errors': '-Werror', 'optimization': '-O3', 'debug': '-g',  'exception_handling': '',      'omp': '-fopenmp', 'coverage':['-O0', '-coverage']},
        'cl': {'warn_all': '/Wall', 'warn_errors': '/WX',     'optimization': '/Ox', 'debug': ['/Zi', '-D_DEBUG', '/MTd'], 'exception_handling': '/EHsc', 'omp': '/openmp', 'coverage':''}
}


# this dictionary maps the name of a linker program to a dictionary mapping the name of
# a linker switch of interest to the specific switch implementing the feature
gLinkerOptions = {
        'gcc': {'debug': ''},
        'g++': {'debug': ''},
        'link': {'debug': '/debug'}
}


def getCFLAGS(mode, backend, warn, warnings_as_errors, hostspblas, CC):
    result = []
    if mode == 'release':
        # turn on optimization
        result.append(gCompilerOptions[CC]['optimization'])
    elif mode == 'debug':
        # turn on debug mode
        result.append(gCompilerOptions[CC]['debug'])
        result.append('-DTHRUST_DEBUG')
    elif mode == 'coverage':
        result.append(gCompilerOptions[CC]['debug'])
        result.append(gCompilerOptions[CC]['coverage'])
        result.append('-DTHRUST_DEBUG')

    # force 32b code on darwin
    # if platform.platform()[:6] == 'Darwin':
    #   result.append('-m32')

    if CC == 'cl':
        result.append('/bigobj')

    # generate omp code
    if backend == 'omp':
        result.append(gCompilerOptions[CC]['omp'])

    if warn:
        # turn on all warnings
        result.append(gCompilerOptions[CC]['warn_all'])

    if warnings_as_errors:
        # treat warnings as errors
        result.append(gCompilerOptions[CC]['warn_errors'])

    # generate hostspblas code
    if hostspblas == 'mkl':
        result.append('-DINTEL_MKL_SPBLAS')

    return result


def getCXXFLAGS(mode, backend, warn, warnings_as_errors, hostspblas, CXX):
    result = []
    if mode == 'release':
        # turn on optimization
        result.append(gCompilerOptions[CXX]['optimization'])
    elif mode == 'debug':
        # turn on debug mode
        result.append(gCompilerOptions[CXX]['debug'])
    elif mode == 'coverage':
        result.append(gCompilerOptions[CXX]['debug'])
        result.append(gCompilerOptions[CXX]['coverage'])
        result.append('-DTHRUST_DEBUG')

    # enable exception handling
    result.append(gCompilerOptions[CXX]['exception_handling'])

    # force 32b code on darwin
    # if platform.platform()[:6] == 'Darwin':
    #   result.append('-m32')

    # generate omp code
    if backend == 'omp':
        result.append(gCompilerOptions[CXX]['omp'])

    if warn:
        # turn on all warnings
        result.append(gCompilerOptions[CXX]['warn_all'])

    if warnings_as_errors:
        # treat warnings as errors
        result.append(gCompilerOptions[CXX]['warn_errors'])

    # generate hostspblas code
    if hostspblas == 'mkl':
        result.append('-DINTEL_MKL_SPBLAS')

    return result


def getNVCCFLAGS(mode, backend, arch):
    result = ['-arch=' + arch]
    if mode == 'debug':
        # turn on debug mode
        # XXX make this work when we've debugged nvcc -G
        # result.append('-G')
        pass
    return result


def getLINKFLAGS(mode, backend, hostspblas, LINK):
    result = []
    if mode == 'debug':
        # turn on debug mode
        result.append(gLinkerOptions[LINK]['debug'])

    # force 32b code on darwin
    # if platform.platform()[:6] == 'Darwin':
    #   result.append('-m32')

    # XXX make this portable
    if backend == 'ocelot':
        result.append(os.popen('OcelotConfig -l').read().split())

    if hostspblas == 'mkl':
        result.append('-fopenmp')

    return result


def Environment():
    # allow the user discretion to choose the MSVC version
    vars = Variables()
    if os.name == 'nt':
        vars.Add(EnumVariable('MSVC_VERSION', 'MS Visual C++ version',
                 None, allowed_values=('8.0', '9.0', '10.0')))

    # add a variable to handle the device backend
    compiler_variable = EnumVariable('compiler', 'The compiler to use', 'nvcc',
                                     allowed_values=('nvcc', 'g++'))
    vars.Add(compiler_variable)

    # add a variable to handle the device backend
    backend_variable = EnumVariable(
        'backend', 'The parallel device backend to target', 'cuda',
        allowed_values=('cuda', 'omp', 'ocelot'))
    vars.Add(backend_variable)

    # add a variable to handle RELEASE/DEBUG mode
    vars.Add(EnumVariable('mode', 'Release versus debug mode', 'release',
                          allowed_values=('release', 'debug', 'coverage')))

    # add a variable to handle compute capability
    vars.Add(
        EnumVariable('arch', 'Compute capability code generation', 'sm_20',
                     allowed_values=('sm_10', 'sm_11', 'sm_12', 'sm_13', 'sm_20', 'sm_21', 'sm_30', 'sm_35')))

    # add a variable to handle warnings
    if os.name == 'posix':
        vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', 1))
    else:
        vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', 0))

    # add a variable to treat warnings as errors
    vars.Add(BoolVariable('Werror', 'Treat warnings as errors', 0))

    # add a variable to filter source files by a regex
    vars.Add('tests', help='Filter test files using a regex')

    # add a variable to execute tests for a single file
    vars.Add('single_test', help='Test a single file')

    # add a variable to handle the host sparse BLAS backend
    hostspblas_variable = EnumVariable(
        'hostspblas', 'Host sparse math library', 'cusp',
        allowed_values=('cusp', 'mkl'))
    vars.Add(hostspblas_variable)

    # add a variable to handle the host BLAS backend
    hostblas_variable = EnumVariable('hostblas', 'Host BLAS library', 'thrust',
                                     allowed_values=('thrust', 'cblas'))
    vars.Add(hostblas_variable)

    # add a variable to handle the device BLSA backend
    deviceblas_variable = EnumVariable(
        'deviceblas', 'Device BLAS library', 'thrust',
        allowed_values=('thrust', 'cblas', 'cublas'))
    vars.Add(deviceblas_variable)

    # create an Environment
    env = OldEnvironment(tools=getTools(), variables=vars)

    # get the absolute path to the directory containing
    # this source file
    thisFile = inspect.getabsfile(Environment)
    thisDir = os.path.dirname(thisFile)

    compiler_define = env['compiler']

    # enable nvcc
    env.Tool(compiler_define, toolpath=[os.path.join(thisDir)])

    # get the preprocessor define to use for the backend
    backend_define = {'cuda': 'THRUST_DEVICE_SYSTEM_CUDA', 'omp':
                      'THRUST_DEVICE_SYSTEM_OMP', 'ocelot': 'THRUST_DEVICE_SYSTEM_CUDA'}[env['backend']]
    env.Append(CFLAGS=['-DTHRUST_DEVICE_SYSTEM=%s' % backend_define])
    env.Append(CXXFLAGS=['-DTHRUST_DEVICE_SYSTEM=%s' % backend_define])

    # get the preprocessor define to use for the device BLAS backend
    device_blas_backend_define = {'thrust': 'CUSP_DEVICE_BLAS_THRUST', 'cblas':
                                  'CUSP_DEVICE_BLAS_CBLAS', 'cublas': 'CUSP_DEVICE_BLAS_CUBLAS'}[env['deviceblas']]
    env.Append(
        CFLAGS=['-DCUSP_DEVICE_BLAS_SYSTEM=%s' % device_blas_backend_define])
    env.Append(
        CXXFLAGS=['-DCUSP_DEVICE_BLAS_SYSTEM=%s' % device_blas_backend_define])

    # get the preprocessor define to use for the host BLAS backend
    host_blas_backend_define = {
        'thrust': 'CUSP_HOST_BLAS_THRUST', 'cblas': 'CUSP_HOST_BLAS_CBLAS'}[env['hostblas']]
    env.Append(
        CFLAGS=['-DCUSP_HOST_BLAS_SYSTEM=%s' % host_blas_backend_define])
    env.Append(
        CXXFLAGS=['-DCUSP_HOST_BLAS_SYSTEM=%s' % host_blas_backend_define])

    # get C compiler switches
    env.Append(CFLAGS=getCFLAGS(env['mode'], env['backend'], env[
               'Wall'], env['Werror'], env['hostspblas'], env.subst('$CC')))

    # get CXX compiler switches
    env.Append(CXXFLAGS=getCXXFLAGS(env['mode'], env['backend'], env[
               'Wall'], env['Werror'], env['hostspblas'], env.subst('$CXX')))

    compile_flag_prefix = ''
    if compiler_define == 'nvcc':

        compile_flag_prefix = '-Xcompiler'

        # get NVCC compiler switches
        env.Append(NVCCFLAGS=getNVCCFLAGS(
            env['mode'], env['backend'], env['arch']))

        # get CUDA paths
        (cuda_exe_path, cuda_lib_path, cuda_inc_path) = get_cuda_paths()
        env.Append(LIBPATH=[cuda_lib_path])
        env.Append(CPPPATH=[cuda_inc_path])

        env.Append(LIBS=['cudart'])

    else:
        env.Append(NVCCFLAGS = ["-x", "c++"])

        GCC_VERSION = subprocess.check_output([env['CXX'], '-dumpversion'])
        if StrictVersion(GCC_VERSION) >= StrictVersion("4.8.0") :
            env.Append(NVCCFLAGS = ["-Wno-unused-local-typedefs"])

        if 'THRUST_PATH' not in os.environ :
            raise ValueError("Building without nvcc requires THRUST_PATH environment variable!")

    # hack to silence unknown pragma warnings
    # env.Append(NVCCFLAGS = [compile_flag_prefix, '-Wno-unknown-pragmas'])

    # get linker switches
    env.Append(LINKFLAGS=getLINKFLAGS(
        env['mode'], env['backend'], env['hostspblas'], env.subst('$LINK')))

    # link against backend-specific runtimes
    # XXX we shouldn't have to link against cudart unless we're using the
    #     cuda runtime, but cudafe inserts some dependencies when compiling .cu files
    # XXX ideally this gets handled in nvcc.py if possible
    if os.name != 'nt':
        env.Append(LIBS=['stdc++', 'm'])

    if env['mode'] == 'coverage':
        env.Append(LIBS=['gcov'])

    if env['backend'] == 'ocelot':
        if os.name == 'posix':
            env.Append(LIBPATH=['/usr/local/lib'])
        else:
            raise ValueError, "Unknown OS.  What is the Ocelot library path?"
    elif env['backend'] == 'omp':
        if os.name == 'posix':
            env.Append(LIBS=['gomp'])
        elif os.name == 'nt':
            env.Append(LIBS=['VCOMP'])
        else:
            raise ValueError, "Unknown OS.  What is the name of the OpenMP library?"

    if env['deviceblas'] == 'cublas':
        env.Append(LIBS=['cublas'])

    if env['hostspblas'] == 'mkl':
        intel_lib = 'mkl_intel'
        if platform.machine()[-2:] == '64':
            intel_lib += '_lp64'

        (mkl_lib_path, mkl_inc_path) = get_mkl_paths()
        env.Append(CPPPATH=[mkl_inc_path])
        env.Append(LIBPATH=[mkl_lib_path])
        env.Append(LIBS=['mkl_core', 'mkl_gnu_thread', intel_lib])

    # set thrust include path
    # this needs to come before the CUDA include path appended above,
    # which may include a different version of thrust
    env.Prepend(CPPPATH=os.path.dirname(thisDir))

    if 'THRUST_PATH' in os.environ:
        env.Prepend(CPPPATH=[os.path.abspath(os.environ['THRUST_PATH'])])

    # import the LD_LIBRARY_PATH so we can run commands which depend
    # on shared libraries
    # XXX we should probably just copy the entire environment
    if os.name == 'posix':
        if ('DYLD_LIBRARY_PATH' in os.environ) and (env['PLATFORM'] == "darwin"):
            env['ENV']['DYLD_LIBRARY_PATH'] = os.environ['DYLD_LIBRARY_PATH']
        elif 'LD_LIBRARY_PATH' in os.environ:
            env['ENV']['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']

    # generate help text
    Help(vars.GenerateHelpText(env))

    return env
