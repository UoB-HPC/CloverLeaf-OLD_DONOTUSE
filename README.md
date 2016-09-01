# CloverLeaf 1.4

This is a C/C++ version of Cloverleaf ported to OpenMP, Kokkos, Ompss, OpenCL, and CUDA.




## Release Notes

### Version 1.4

This release includes:
*   A rewrite of all the host code into C/C++.
*   Re-porting to CUDA, OpenCL, Kokkos, OpenMP, and OmpSs.

### Version 1.3

CloverLeaf 1.3 contains a number of optimisations over previous releases.
These include a number of loop fusion optimisations and the use of scalar variables over work arrays.
Overall this improves cache efficiency.

This version also contains some support for explicit tiling.
This is activated through the two input deck parameters:
*   `tiles_per_chunk`   To specify how many tiles per MPI ranks there are.
*   `tiles_per_problem` To specify how many global tiles there are, this is rounded down to be an even number per MPI rank.

## Performance

![Overall Performance](https://github.com/jamuus/CloverLeaf/blob/master/img/overallPerf.png)
![CloverLeaf scaling on 2x ThunderX ARM chips.](https://github.com/jamuus/CloverLeaf/blob/master/img/thunderxscaling.png)

## Code Structure

The host code was written as a direct mapping from the serial version of Cloverleaf 1.3. During development minor modifications were made including a simplification of the file parsing, and replacing the original kernel invocations with adaptor invocations.

All source code is now located in `src/`. Kernels describing the computation are located in the `kernels` sub directory. Adaptors for each kernel are located in the `adaptors` sub directory.

Adaptors abstract what computation needs performing from how/where it is executed. The content of an adaptor is determined at compile time using preprocessor macros prefixed by `USE_`. e.g. `USE_OPENMP`.

Common data structures and variables are defined in `definitions_c.h`. Library specific definitions are included using its macro. These definitions are located in `$LIBdefs.h` and define what data type the fields should be as well as how to access them.

Types `field_2d_t` and `field_1d_t` are used by kernels as the parameter types for each field. E.g. for OpenMP these are `double` pointers with `restrict` qualifiers to help the compiler optimize. Field types prefixed by const are used for parameters in kernels that don't assign to the field, extra qualifiers can be used here to aid the compiler e.g. const. Also defined is the macro `kernelquad` which allows adding qualifiers to kernel definitions e.g. `__DEVICE__` for CUDA kernels.

Allocation is different for each framework and so separate allocation logic is defined in file `allocate_$LIB.c`.


## Compilation

There is configuration in the Makefile for GCC, Intel's compiler, and Clang. Choosing a compiler and framework is done as follows:

    make COMPILER=GNU USE_OPENMP=1

The following table shows known working combinations.

|        | GNU | Intel | Clang | Mercurium |
|--------|:---:|:-----:|:-----:|:---------:|
| OpenMP |  ✓  |   ✓   |       |           |
| Kokkos |  ✓  |   ✓   |       |           |
| OmpSs  |     |       |       |     ✓     |
| OpenCL |     |   ✓   |       |           |
| CUDA   |     |   ✓   |   ✓   |           |

### Kokkos

To compile for Kokkos make sure environment variable `KOKKOS_PATH` is set.

<!-- 

|        | MPI HX | Tile HX |
|--------|:------:|:-------:|
| OpenMP |  ✓     |   ✓     |
| Kokkos |  ✓     |         |
| OmpSs  |  ✓     |   ✓     |
| OpenCL |        |         |
| CUDA   |        |         |

 -->


## OpenMP

Parallelisation is implemented by giving each thread a portion of the rows to work on. This is acheived using non collapse OpenMP for pragmas on the outer loop for each kernel. The macro `DOUBLEFOR` encapsulates this behaviour. The pragma `ivdep` is added to the inner loop to notify compatible compilers that the loops are independant.


## Kokkos

Kokkos attempts to abstract where the code will be run and the project follows this abstraction by defining a set of fields for the device that will execute the kernels, and a set for the host. Device buffer pointers are prepended with `d_` in `field_t`. The host buffers are used for initialisation and make getting the data back from the device trivial.

Macro `kernelqual` is set to make each kernel a function template allowing the kernels to be called on any type of `View`.

In `adaptors/kokkos` are source files containing the Kokkos functors defined for each kernel.


## OpenCL

Initialising an OpenCL device is performed just after calculating chunk size and just before allocating buffers in `start.c`. This allows parameters to be defined as compile time constants in the OpenCL kernel files.

The kernel code is defined in files in `adaptors/opencl`. As with the Kokkos version, there exist device and host buffers for each field. Workgroup sizes are defined in `opencldefs.h`. OpenCL specific definitions is split into two files as the definitions for accessing arrays is required in the device code but not all the definitions for the host code is desired to be included there.

## CUDA

CUDA allows host code and device code to reside in the same file so the CUDA kernels are currently situated next to the adaptor code.

The implementation is similar to the OpenCL version where there exist both host and device buffer for each field.

## Todo

*   Visit file output hasn't been ported.
*   MPI buffer packing for halo exchange isn't implemented for all libraries
