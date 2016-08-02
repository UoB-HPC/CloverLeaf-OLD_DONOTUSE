#include "cl.hpp"

extern cl::Context        openclContext;
extern cl::CommandQueue   openclQueue;
extern cl::Program        openclProgram;

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
};


#define DOUBLEFOR(k_from, k_to, j_from, j_to, body) \
    _Pragma("omp for") \
    for(int k = (k_from); k <= (k_to); k++) { \
        _Pragma("ivdep") \
        for(int j = (j_from); j <= (j_to); j++) { \
            body; \
        } \
    }

#include "openclaccessdefs.h"