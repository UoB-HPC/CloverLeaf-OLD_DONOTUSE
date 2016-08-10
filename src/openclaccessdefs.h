#define FTNREF1D(i_index,i_lb) \
    ((i_index)-(i_lb))
#define FTNREF2D(i_index, j_index, i_size, i_lb, j_lb) \
    ((i_size) * (j_index - (j_lb)) + \
        (i_index) - (i_lb))


#if !defined(USE_OPENCL)
__constant double g_small = 1.0e-16,
                  g_big  = 1.0e+21;
__constant double dtc_safe = 0.7,
                  dtu_safe = 0.5,
                  dtv_safe = 0.5,
                  dtdiv_safe = 0.7;
#ifndef MAX
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) >= (b) ? (b) : (a))
#endif

#endif


#define T1ACCESS(d, i, j)         d[FTNREF2D(i, j, x_max + 4, x_min - 2, y_min - 2)]
#define T2ACCESS(d, i, j)         d[FTNREF2D(i, j, x_max + 5, x_min - 2, y_min - 2)]

#define DENSITY0(d, i, j)      T1ACCESS(d, i, j)
#define DENSITY1(d, i, j)      T1ACCESS(d, i, j)

#define ENERGY0(d, i, j)       T1ACCESS(d, i, j)
#define ENERGY1(d, i, j)       T1ACCESS(d, i, j)
#define PRESSURE(d, i, j)      T1ACCESS(d, i, j)
#define VISCOSITY(d, i, j)     T1ACCESS(d, i, j)
#define SOUNDSPEED(d, i, j)    T1ACCESS(d, i, j)

#define VEL(d, i, j)           T2ACCESS(d, i, j)
#define XVEL0(d, i, j)         VEL(d, i, j)
#define XVEL1(d, i, j)         VEL(d, i, j)
#define YVEL0(d, i, j)         VEL(d, i, j)
#define YVEL1(d, i, j)         VEL(d, i, j)

#define VOL_FLUX_X(d, i, j)    T2ACCESS(d, i, j)
#define MASS_FLUX_X(d, i, j)   T2ACCESS(d, i, j)
#define VOL_FLUX_Y(d, i, j)    T1ACCESS(d, i, j)
#define MASS_FLUX_Y(d, i, j)   T1ACCESS(d, i, j)

#define VOLUME(d, i, j)        T1ACCESS(d, i, j)
#define XAREA(d, i, j)         T2ACCESS(d, i, j)
#define YAREA(d, i, j)         T1ACCESS(d, i, j)

#define WORK_ARRAY(d, i, j)    T2ACCESS(d, i, j)

#define FIELD_1D(d, i, j)      d[FTNREF1D(i, j)]


#if defined(USE_OPENCL)

#define const_field_2d_t     const double* __restrict__
#define field_2d_t           double* __restrict__

#define const_field_1d_t     const double* __restrict__
#define field_1d_t           double* __restrict__

#define flag_t               int*

#else // Inside opencl code

#define const_field_2d_t     const global double* __restrict__
#define field_2d_t           global double* __restrict__

#define const_field_1d_t     const global double* __restrict__
#define field_1d_t           global double* __restrict__

#define flag_t               constant int*

#define kernelqual

#endif