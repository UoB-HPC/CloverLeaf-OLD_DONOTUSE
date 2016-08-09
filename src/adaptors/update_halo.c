
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

#include "../definitions_c.h"

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
        sizeof(int) * NUM_FIELDS, fields);

    cl::Kernel update_halo_1(openclProgram, "update_halo_1_kernel");
    update_halo_1.setArg(0,  x_min);
    update_halo_1.setArg(1,  x_max);
    update_halo_1.setArg(2,  y_min);
    update_halo_1.setArg(3,  y_max);


    checkOclErr(update_halo_1.setArg(4, chunk_neighbours_cl));
    checkOclErr(update_halo_1.setArg(5, tile_neighbours_cl));
    checkOclErr(update_halo_1.setArg(6,  *tile.field.d_density0));
    checkOclErr(update_halo_1.setArg(7,  *tile.field.d_density1));
    checkOclErr(update_halo_1.setArg(8,  *tile.field.d_energy0));
    checkOclErr(update_halo_1.setArg(9,  *tile.field.d_energy1));
    checkOclErr(update_halo_1.setArg(10, *tile.field.d_pressure));
    checkOclErr(update_halo_1.setArg(11, *tile.field.d_viscosity));
    checkOclErr(update_halo_1.setArg(12, *tile.field.d_soundspeed));
    checkOclErr(update_halo_1.setArg(13, *tile.field.d_xvel0));
    checkOclErr(update_halo_1.setArg(14, *tile.field.d_yvel0));
    checkOclErr(update_halo_1.setArg(15, *tile.field.d_xvel1));
    checkOclErr(update_halo_1.setArg(16, *tile.field.d_yvel1));
    checkOclErr(update_halo_1.setArg(17, *tile.field.d_vol_flux_x));
    checkOclErr(update_halo_1.setArg(18, *tile.field.d_mass_flux_x));
    checkOclErr(update_halo_1.setArg(19, *tile.field.d_vol_flux_y));
    checkOclErr(update_halo_1.setArg(20, *tile.field.d_mass_flux_y));
    checkOclErr(update_halo_1.setArg(21, fields_cl));
    checkOclErr(update_halo_1.setArg(22, depth));


    checkOclErr(openclQueue.enqueueNDRangeKernel(
                    update_halo_1,
                    cl::NullRange,
                    cl::NDRange((x_max + depth) - (x_min - depth) + 1, depth - 1 + 1),
                    cl::NullRange));

    cl::Kernel update_halo_2(openclProgram, "update_halo_2_kernel");
    update_halo_2.setArg(0,  x_min);
    update_halo_2.setArg(1,  x_max);
    update_halo_2.setArg(2,  y_min);
    update_halo_2.setArg(3,  y_max);


    checkOclErr(update_halo_2.setArg(4, chunk_neighbours_cl));
    checkOclErr(update_halo_2.setArg(5, tile_neighbours_cl));
    checkOclErr(update_halo_2.setArg(6,  *tile.field.d_density0));
    checkOclErr(update_halo_2.setArg(7,  *tile.field.d_density1));
    checkOclErr(update_halo_2.setArg(8,  *tile.field.d_energy0));
    checkOclErr(update_halo_2.setArg(9,  *tile.field.d_energy1));
    checkOclErr(update_halo_2.setArg(10, *tile.field.d_pressure));
    checkOclErr(update_halo_2.setArg(11, *tile.field.d_viscosity));
    checkOclErr(update_halo_2.setArg(12, *tile.field.d_soundspeed));
    checkOclErr(update_halo_2.setArg(13, *tile.field.d_xvel0));
    checkOclErr(update_halo_2.setArg(14, *tile.field.d_yvel0));
    checkOclErr(update_halo_2.setArg(15, *tile.field.d_xvel1));
    checkOclErr(update_halo_2.setArg(16, *tile.field.d_yvel1));
    checkOclErr(update_halo_2.setArg(17, *tile.field.d_vol_flux_x));
    checkOclErr(update_halo_2.setArg(18, *tile.field.d_mass_flux_x));
    checkOclErr(update_halo_2.setArg(19, *tile.field.d_vol_flux_y));
    checkOclErr(update_halo_2.setArg(20, *tile.field.d_mass_flux_y));
    checkOclErr(update_halo_2.setArg(21, fields_cl));
    checkOclErr(update_halo_2.setArg(22, depth));


    checkOclErr(openclQueue.enqueueNDRangeKernel(
                    update_halo_2,
                    cl::NullRange,
                    cl::NDRange(depth - 1 + 1, (y_max + depth) - (y_min - depth) + 1),
                    cl::NullRange));
    if (profiler_on)
        openclQueue.finish();
}
#endif


