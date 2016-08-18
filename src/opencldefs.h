#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "cl.hpp"

extern cl::Context        openclContext;
extern cl::CommandQueue   openclQueue;
extern cl::Program        openclProgram;

inline const char* getErrorString(cl_int error)
{
    switch (error) {
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

#define checkOclErr(err) {if ((err) != CL_SUCCESS){fprintf(stderr,"%s:%d: %s\n", __FILE__, __LINE__, getErrorString(err));exit(1);}}

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
    { \
        cl_int err; \
        hostbuf = (double*)openclQueue.enqueueMapBuffer(\
            *(devbuf), \
            CL_TRUE, \
            (rw), \
            0, \
            sizeof(double) * size, NULL, NULL, &err); \
        checkOclErr(err); \
    }

#define unmapoclmem(devbuf, hostbuf) \
    openclQueue.enqueueUnmapMemObject(*(devbuf), (hostbuf));

#define DOUBLEFOR(k_from, k_to, j_from, j_to, body) \
    for(int k = (k_from); k <= (k_to); k++) { \
        _Pragma("ivdep") \
        for(int j = (j_from); j <= (j_to); j++) { \
            body; \
        } \
    }

#define kernelqual inline
#define local_t    local double*

inline int roundUp(int numToRound, int multiple)
{
    if (multiple == 0)
        return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}

inline cl::NDRange calcGlobalSize(
    cl::NDRange global,
    cl::NDRange local)
{
    if (local == cl::NullRange)
        return global;

    return cl::NDRange(
               roundUp(global[0], local[0]),
               roundUp(global[1], local[1])
           );
}

#define acclerate_local_size       cl::NDRange(256,1)

#define advec_cell_x1_local_size   cl::NDRange(256,1)
#define advec_cell_x2_local_size   cl::NDRange(256,1)
#define advec_cell_x3_local_size   cl::NDRange(256,1)
#define advec_cell_y1_local_size   cl::NDRange(256,1)
#define advec_cell_y2_local_size   cl::NDRange(256,1)
#define advec_cell_y3_local_size   cl::NDRange(256,1)

#define advec_mom_ms1_local_size   cl::NDRange(256,1)
#define advec_mom_ms2_local_size   cl::NDRange(256,1)
#define advec_mom_ms3_local_size   cl::NDRange(256,1)
#define advec_mom_ms4_local_size   cl::NDRange(256,1)
#define advec_mom_x1_local_size    cl::NDRange(256,1)
#define advec_mom_x2_local_size    cl::NDRange(256,1)
#define advec_mom_x3_local_size    cl::NDRange(256,1)
#define advec_mom_x4_local_size    cl::NDRange(256,1)
#define advec_mom_y1_local_size    cl::NDRange(256,1)
#define advec_mom_y2_local_size    cl::NDRange(256,1)
#define advec_mom_y3_local_size    cl::NDRange(256,1)
#define advec_mom_y4_local_size    cl::NDRange(256,1)

#define dtmin_local_size           cl::NDRange(256,1)
#define field_summary_local_size   cl::NDRange(256,1)

#define flux_calc_x_local_size     cl::NDRange(256,1)
#define flux_calc_y_local_size     cl::NDRange(256,1)

#define ideal_gas_local_size       cl::NDRange(256,1)

#define pdv_kernel_local_size      cl::NDRange(256,1)

#define reset_field_local_size     cl::NDRange(256,1)

#define revert_local_size          cl::NDRange(256,1)

#define update_halo_1_local_size   cl::NDRange(256,1)
#define update_halo_2_local_size   cl::NDRange(256,1)

#define viscosity_local_size       cl::NDRange(256,1)


#include "openclaccessdefs.h"
