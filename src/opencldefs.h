#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "cl.hpp"

extern cl::Context        openclContext;
extern cl::CommandQueue   openclQueue;
extern cl::Program        openclProgram;
const char* getErrorString(cl_int);
#define checkOclErr(err) {if ((err) != CL_SUCCESS){fprintf(stderr,"Line %d: %s\n", __LINE__, getErrorString(err));exit(1);}}

struct field_type {
    cl::Buffer* d_density0;
    cl::Buffer* d_density1;
    cl::Buffer* d_energy0;
    cl::Buffer* d_energy1;
    cl::Buffer* d_pressure;
    cl::Buffer* d_viscosity;
    cl::Buffer* d_soundspeed;
    cl::Buffer* d_xvel0;
    cl::Buffer* d_xvel1;
    cl::Buffer* d_yvel0;
    cl::Buffer* d_yvel1;
    cl::Buffer* d_vol_flux_x;
    cl::Buffer* d_mass_flux_x;
    cl::Buffer* d_vol_flux_y;
    cl::Buffer* d_mass_flux_y;
    //node_flux; stepbymass; volume_change; pre_vo
    cl::Buffer* d_work_array1;
    //node_mass_post; post_vol
    cl::Buffer* d_work_array2;
    //node_mass_pre; pre_mass
    cl::Buffer* d_work_array3;
    //advec_vel; post_mass
    cl::Buffer* d_work_array4;
    //mom_flux; advec_vol
    cl::Buffer* d_work_array5;
    //pre_vol; post_ener
    cl::Buffer* d_work_array6;
    //post_vol; ener_flux
    cl::Buffer* d_work_array7;
    cl::Buffer* d_cellx;
    cl::Buffer* d_celly;
    cl::Buffer* d_vertexx;
    cl::Buffer* d_vertexy;
    cl::Buffer* d_celldx;
    cl::Buffer* d_celldy;
    cl::Buffer* d_vertexdx;
    cl::Buffer* d_vertexdy;
    cl::Buffer* d_volume;
    cl::Buffer* d_xarea;
    cl::Buffer* d_yarea;

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

#define mapoclmem(devbuf, hostbuf, size, rw) \
    hostbuf = (double*)openclQueue.enqueueMapBuffer(\
        *(devbuf), \
        CL_TRUE, \
        (rw), \
        0, \
        sizeof(double) * size);

#define unmapoclmem(devbuf, hostbuf) openclQueue.enqueueUnmapMemObject(*(devbuf), (hostbuf));

#define DOUBLEFOR(k_from, k_to, j_from, j_to, body) \
    for(int k = (k_from); k <= (k_to); k++) { \
        _Pragma("ivdep") \
        for(int j = (j_from); j <= (j_to); j++) { \
            body; \
        } \
    }



#define acclerate_local_size       cl::NullRange

#define advec_cell_x1_local_size   cl::NullRange
#define advec_cell_x2_local_size   cl::NullRange
#define advec_cell_x3_local_size   cl::NullRange
#define advec_cell_y1_local_size   cl::NullRange
#define advec_cell_y2_local_size   cl::NullRange
#define advec_cell_y3_local_size   cl::NullRange

#define advec_mom_ms1_local_size   cl::NullRange
#define advec_mom_ms2_local_size   cl::NullRange
#define advec_mom_ms3_local_size   cl::NullRange
#define advec_mom_ms4_local_size   cl::NullRange
#define advec_mom_x1_local_size    cl::NullRange
#define advec_mom_x2_local_size    cl::NullRange
#define advec_mom_x3_local_size    cl::NullRange
#define advec_mom_x4_local_size    cl::NullRange
#define advec_mom_y1_local_size    cl::NullRange
#define advec_mom_y2_local_size    cl::NullRange
#define advec_mom_y3_local_size    cl::NullRange
#define advec_mom_y4_local_size    cl::NullRange

#define dtmin_local_size           cl::NullRange

#define flux_calc_x_local_size     cl::NullRange
#define flux_calc_y_local_size     cl::NullRange

#define ideal_gas_local_size       cl::NullRange

#define pdv_kernel_local_size      cl::NullRange

#define reset_field_local_size     cl::NullRange

#define revert_local_size          cl::NullRange

#define update_halo_local_size     cl::NullRange

#define update_halo_local_size     cl::NullRange


#include "openclaccessdefs.h"
