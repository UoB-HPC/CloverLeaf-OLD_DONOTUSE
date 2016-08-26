#include <cuda_runtime.h>
#include <stdlib.h>

struct field_type {
    double* d_density0;
    double* d_density1;
    double* d_energy0;
    double* d_energy1;
    double* d_pressure;
    double* d_viscosity;
    double* d_soundspeed;
    double* d_xvel0;
    double* d_xvel1;
    double* d_yvel0;
    double* d_yvel1;
    double* d_vol_flux_x;
    double* d_mass_flux_x;
    double* d_vol_flux_y;
    double* d_mass_flux_y;
    //node_flux; stepbymass; volume_change; pre_vo
    double* d_work_array1;
    //node_mass_post; post_vol
    double* d_work_array2;
    //node_mass_pre; pre_mass
    double* d_work_array3;
    //advec_vel; post_mass
    double* d_work_array4;
    //mom_flux; advec_vol
    double* d_work_array5;
    //pre_vol; post_ener
    double* d_work_array6;
    //post_vol; ener_flux
    double* d_work_array7;
    double* d_cellx;
    double* d_celly;
    double* d_vertexx;
    double* d_vertexy;
    double* d_celldx;
    double* d_celldy;
    double* d_vertexdx;
    double* d_vertexdy;
    double* d_volume;
    double* d_xarea;
    double* d_yarea;

    double* density0;
    double* density1;
    double* energy0;
    double* energy1;
    double* pressure;
    double* viscosity;
    double* soundspeed;
    double* xvel0;
    double* xvel1;
    double* yvel0;
    double* yvel1;
    double* vol_flux_x;
    double* mass_flux_x;
    double* vol_flux_y;
    double* mass_flux_y;
    //node_flux; stepbymass; volume_change; pre_vo
    double* work_array1;
    //node_mass_post; post_vol
    double* work_array2;
    //node_mass_pre; pre_mass
    double* work_array3;
    //advec_vel; post_mass
    double* work_array4;
    //mom_flux; advec_vol
    double* work_array5;
    //pre_vol; post_ener
    double* work_array6;
    //post_vol; ener_flux
    double* work_array7;
    double* cellx;
    double* celly;
    double* vertexx;
    double* vertexy;
    double* celldx;
    double* celldy;
    double* vertexdx;
    double* vertexdy;
    double* volume;
    double* xarea;
    double* yarea;

    int density0_size;
    int density1_size;
    int energy0_size;
    int energy1_size;
    int pressure_size;
    int viscosity_size;
    int soundspeed_size;
    int xvel0_size;
    int xvel1_size;
    int yvel0_size;
    int yvel1_size;
    int vol_flux_x_size;
    int mass_flux_x_size;
    int vol_flux_y_size;
    int mass_flux_y_size;
    //node_flux; stepbymass; volume_change; pre_vo
    int work_array1_size;
    //node_mass_post; post_vol
    int work_array2_size;
    //node_mass_pre; pre_mass
    int work_array3_size;
    //advec_vel; post_mass
    int work_array4_size;
    //mom_flux; advec_vol
    int work_array5_size;
    //pre_vol; post_ener
    int work_array6_size;
    //post_vol; ener_flux
    int work_array7_size;
    int cellx_size;
    int celly_size;
    int vertexx_size;
    int vertexy_size;
    int celldx_size;
    int celldy_size;
    int vertexdx_size;
    int vertexdy_size;
    int volume_size;
    int xarea_size;
    int yarea_size;
};

#define DOUBLEFOR(k_from, k_to, j_from, j_to, body) \
    for(int k = (k_from); k <= (k_to); k++) { \
        _Pragma("ivdep") \
        for(int j = (j_from); j <= (j_to); j++) { \
            body; \
        } \
    }


#define FTNREF1D(i_index,i_lb) \
    ((i_index)-(i_lb))
#define FTNREF2D(i_index, j_index, i_size, i_lb, j_lb) \
    ((i_size) * (j_index - (j_lb)) + \
        (i_index) - (i_lb))

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


#define const_field_2d_t     const double* __restrict
#define field_2d_t           double* __restrict

#define const_field_1d_t     const double* __restrict
#define field_1d_t           double* __restrict

#define flag_t               int*

#define kernelqual   __device__ __host__ inline

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#include <math.h>
inline dim3 numBlocks(dim3 globalSize, dim3 threadsPerBlock)
{
    return dim3(ceil(globalSize.x / (double)threadsPerBlock.x),
                ceil(globalSize.y / (double)threadsPerBlock.y));
}

#define accelerate_blocksize      dim3(256,1)

#define advec_cell_x1_blocksize   dim3(256,1)
#define advec_cell_x2_blocksize   dim3(256,1)
#define advec_cell_x3_blocksize   dim3(256,1)
#define advec_cell_y1_blocksize   dim3(256,1)
#define advec_cell_y2_blocksize   dim3(256,1)
#define advec_cell_y3_blocksize   dim3(256,1)

#define advec_mom_ms1_blocksize   dim3(256,1)
#define advec_mom_ms2_blocksize   dim3(256,1)
#define advec_mom_ms3_blocksize   dim3(256,1)
#define advec_mom_ms4_blocksize   dim3(256,1)
#define advec_mom_x1_blocksize    dim3(256,1)
#define advec_mom_x2_blocksize    dim3(256,1)
#define advec_mom_x3_blocksize    dim3(256,1)
#define advec_mom_x4_blocksize    dim3(256,1)
#define advec_mom_y1_blocksize    dim3(256,1)
#define advec_mom_y2_blocksize    dim3(256,1)
#define advec_mom_y3_blocksize    dim3(256,1)
#define advec_mom_y4_blocksize    dim3(256,1)

#define dtmin_blocksize           dim3(256,1)
#define field_summary_blocksize   dim3(256,1)

#define flux_calc_x_blocksize     dim3(256,1)
#define flux_calc_y_blocksize     dim3(256,1)

#define ideal_gas_blocksize       dim3(256,1)

#define pdv_kernel_blocksize      dim3(256,1)

#define reset_field_blocksize     dim3(256,1)

#define revert_blocksize          dim3(256,1)

#define update_halo_blocksize     dim3(256,1)

#define viscosity_blocksize       dim3(256,1)
