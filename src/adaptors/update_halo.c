
#include "../kernels/update_halo_kernel_c.c"

#if defined(USE_OPENMP) || defined(USE_OMPSS) || defined(USE_KOKKOS)

void update_local_halo(struct tile_type tile, int* chunk_neighbours, int* fields, int depth)
{
    int x_min = tile.t_xmin,
        x_max = tile.t_xmax,
        y_min = tile.t_ymin,
        y_max = tile.t_ymax;

    for (int j = x_min - depth; j <= x_max + depth; j++) {
#pragma ivdep
        for (int k = 1; k <= depth; k++) {
            update_halo_kernel_1(
                j, k,
                tile.t_xmin,
                tile.t_xmax,
                tile.t_ymin,
                tile.t_ymax,
                chunk_neighbours,
                tile.tile_neighbours,
                tile.field.density0,
                tile.field.density1,
                tile.field.energy0,
                tile.field.energy1,
                tile.field.pressure,
                tile.field.viscosity,
                tile.field.soundspeed,
                tile.field.xvel0,
                tile.field.yvel0,
                tile.field.xvel1,
                tile.field.yvel1,
                tile.field.vol_flux_x,
                tile.field.mass_flux_x,
                tile.field.vol_flux_y,
                tile.field.mass_flux_y,
                fields,
                depth);
        }
    }
    for (int k = y_min - depth; k <= y_max + depth; k++) {
#pragma ivdep
        for (int j = 1; j <= depth; j++) {
            update_halo_kernel_2(
                j, k,
                tile.t_xmin,
                tile.t_xmax,
                tile.t_ymin,
                tile.t_ymax,
                chunk_neighbours,
                tile.tile_neighbours,
                tile.field.density0,
                tile.field.density1,
                tile.field.energy0,
                tile.field.energy1,
                tile.field.pressure,
                tile.field.viscosity,
                tile.field.soundspeed,
                tile.field.xvel0,
                tile.field.yvel0,
                tile.field.xvel1,
                tile.field.yvel1,
                tile.field.vol_flux_x,
                tile.field.mass_flux_x,
                tile.field.vol_flux_y,
                tile.field.mass_flux_y,
                fields,
                depth);
        }
    }
}
#endif


#if defined(USE_OPENCL)


const char* getErrorString(cl_int error)
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

#define printerr(err) {if ((err) != CL_SUCCESS){fprintf(stderr,"%d: %s\n", __LINE__, getErrorString(err));exit(1);}}

void update_local_halo(struct tile_type tile, int* chunk_neighbours, int* fields, int depth)
{
    int x_min = tile.t_xmin,
        x_max = tile.t_xmax,
        y_min = tile.t_ymin,
        y_max = tile.t_ymax;

    cl::Buffer chunk_neighbours_cl(
        openclContext, CL_MEM_COPY_HOST_PTR,
        sizeof(int) * 4, chunk_neighbours);

    cl::Buffer tile_neighbours_cl(
        openclContext, CL_MEM_COPY_HOST_PTR,
        sizeof(int) * 4, tile.tile_neighbours);

    cl::Buffer fields_cl(
        openclContext, CL_MEM_COPY_HOST_PTR,
        sizeof(int) * 15, fields);

    cl::Kernel update_halo_1(openclProgram, "update_halo_1_kernel");
    update_halo_1.setArg(0,  x_min);
    update_halo_1.setArg(1,  x_max);
    update_halo_1.setArg(2,  y_min);
    update_halo_1.setArg(3,  y_max);


    printerr(update_halo_1.setArg(4, chunk_neighbours_cl));
    printerr(update_halo_1.setArg(5, tile_neighbours_cl));
    printerr(update_halo_1.setArg(6,  *tile.field.d_density0));
    printerr(update_halo_1.setArg(7,  *tile.field.d_density1));
    printerr(update_halo_1.setArg(8,  *tile.field.d_energy0));
    printerr(update_halo_1.setArg(9,  *tile.field.d_energy1));
    printerr(update_halo_1.setArg(10, *tile.field.d_pressure));
    printerr(update_halo_1.setArg(11, *tile.field.d_viscosity));
    printerr(update_halo_1.setArg(12, *tile.field.d_soundspeed));
    printerr(update_halo_1.setArg(13, *tile.field.d_xvel0));
    printerr(update_halo_1.setArg(14, *tile.field.d_yvel0));
    printerr(update_halo_1.setArg(15, *tile.field.d_xvel1));
    printerr(update_halo_1.setArg(16, *tile.field.d_yvel1));
    printerr(update_halo_1.setArg(17, *tile.field.d_vol_flux_x));
    printerr(update_halo_1.setArg(18, *tile.field.d_mass_flux_x));
    printerr(update_halo_1.setArg(19, *tile.field.d_vol_flux_y));
    printerr(update_halo_1.setArg(20, *tile.field.d_mass_flux_y));
    printerr(update_halo_1.setArg(21, fields_cl));
    printerr(update_halo_1.setArg(22, depth));


    printerr(openclQueue.enqueueNDRangeKernel(
                 update_halo_1, cl::NullRange,
                 cl::NDRange((x_max + depth) - (x_min - depth) + 1, depth - 1 + 1),
                 cl::NullRange));
//     for (int j = x_min - depth; j <= x_max + depth; j++) {
// #pragma ivdep
//         for (int k = 1; k <= depth; k++) {
//             update_halo_kernel_1(
//                 j, k,
//                 tile.t_xmin,
//                 tile.t_xmax,
//                 tile.t_ymin,
//                 tile.t_ymax,
//                 chunk_neighbours,
//                 tile.tile_neighbours,
//                 tile.field.density0,
//                 tile.field.density1,
//                 tile.field.energy0,
//                 tile.field.energy1,
//                 tile.field.pressure,
//                 tile.field.viscosity,
//                 tile.field.soundspeed,
//                 tile.field.xvel0,
//                 tile.field.yvel0,
//                 tile.field.xvel1,
//                 tile.field.yvel1,
//                 tile.field.vol_flux_x,
//                 tile.field.mass_flux_x,
//                 tile.field.vol_flux_y,
//                 tile.field.mass_flux_y,
//                 fields,
//                 depth);
//         }
//     }

    cl::Kernel update_halo_2(openclProgram, "update_halo_2_kernel");
    update_halo_2.setArg(0,  x_min);
    update_halo_2.setArg(1,  x_max);
    update_halo_2.setArg(2,  y_min);
    update_halo_2.setArg(3,  y_max);


    printerr(update_halo_2.setArg(4, chunk_neighbours_cl));
    printerr(update_halo_2.setArg(5, tile_neighbours_cl));
    printerr(update_halo_2.setArg(6,  *tile.field.d_density0));
    printerr(update_halo_2.setArg(7,  *tile.field.d_density1));
    printerr(update_halo_2.setArg(8,  *tile.field.d_energy0));
    printerr(update_halo_2.setArg(9,  *tile.field.d_energy1));
    printerr(update_halo_2.setArg(10, *tile.field.d_pressure));
    printerr(update_halo_2.setArg(11, *tile.field.d_viscosity));
    printerr(update_halo_2.setArg(12, *tile.field.d_soundspeed));
    printerr(update_halo_2.setArg(13, *tile.field.d_xvel0));
    printerr(update_halo_2.setArg(14, *tile.field.d_yvel0));
    printerr(update_halo_2.setArg(15, *tile.field.d_xvel1));
    printerr(update_halo_2.setArg(16, *tile.field.d_yvel1));
    printerr(update_halo_2.setArg(17, *tile.field.d_vol_flux_x));
    printerr(update_halo_2.setArg(18, *tile.field.d_mass_flux_x));
    printerr(update_halo_2.setArg(19, *tile.field.d_vol_flux_y));
    printerr(update_halo_2.setArg(20, *tile.field.d_mass_flux_y));
    printerr(update_halo_2.setArg(21, fields_cl));
    printerr(update_halo_2.setArg(22, depth));


    printerr(openclQueue.enqueueNDRangeKernel(
                 update_halo_2, cl::NullRange,
                 cl::NDRange(depth - 1 + 1, (y_max + depth) - (y_min - depth) + 1),
                 cl::NullRange));
    openclQueue.finish();
//     for (int k = y_min - depth; k <= y_max + depth; k++) {
// #pragma ivdep
//         for (int j = 1; j <= depth; j++) {
//             update_halo_kernel_2(
//                 j, k,
//                 tile.t_xmin,
//                 tile.t_xmax,
//                 tile.t_ymin,
//                 tile.t_ymax,
//                 chunk_neighbours,
//                 tile.tile_neighbours,
//                 tile.field.density0,
//                 tile.field.density1,
//                 tile.field.energy0,
//                 tile.field.energy1,
//                 tile.field.pressure,
//                 tile.field.viscosity,
//                 tile.field.soundspeed,
//                 tile.field.xvel0,
//                 tile.field.yvel0,
//                 tile.field.xvel1,
//                 tile.field.yvel1,
//                 tile.field.vol_flux_x,
//                 tile.field.mass_flux_x,
//                 tile.field.vol_flux_y,
//                 tile.field.mass_flux_y,
//                 fields,
//                 depth);
//         }
//     }
}
#endif


