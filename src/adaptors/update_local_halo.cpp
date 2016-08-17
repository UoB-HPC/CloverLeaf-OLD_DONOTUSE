
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

#if defined(USE_CUDA)


__global__ void update_halo_1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    int* chunk_neighbours,
    int* tile_neighbours,
    field_2d_t density0,
    field_2d_t density1,
    field_2d_t energy0,
    field_2d_t energy1,
    field_2d_t pressure,
    field_2d_t viscosity,
    field_2d_t soundspeed,
    field_2d_t xvel0,
    field_2d_t yvel0,
    field_2d_t xvel1,
    field_2d_t yvel1,
    field_2d_t vol_flux_x,
    field_2d_t mass_flux_x,
    field_2d_t vol_flux_y,
    field_2d_t mass_flux_y,
    int* fields,
    int depth)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + (x_min - depth);
    int k = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (j <= x_max + depth && k <= depth)
        update_halo_kernel_1(
            j, k,
            x_min,
            x_max,
            y_min,
            y_max,
            chunk_neighbours,
            tile_neighbours,
            density0,
            density1,
            energy0,
            energy1,
            pressure,
            viscosity,
            soundspeed,
            xvel0,
            yvel0,
            xvel1,
            yvel1,
            vol_flux_x,
            mass_flux_x,
            vol_flux_y,
            mass_flux_y,
            fields,
            depth);
}
__global__ void update_halo_2_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    int* chunk_neighbours,
    int* tile_neighbours,
    field_2d_t density0,
    field_2d_t density1,
    field_2d_t energy0,
    field_2d_t energy1,
    field_2d_t pressure,
    field_2d_t viscosity,
    field_2d_t soundspeed,
    field_2d_t xvel0,
    field_2d_t yvel0,
    field_2d_t xvel1,
    field_2d_t yvel1,
    field_2d_t vol_flux_x,
    field_2d_t mass_flux_x,
    field_2d_t vol_flux_y,
    field_2d_t mass_flux_y,
    int* fields,
    int depth)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int k = threadIdx.y + blockIdx.y * blockDim.y + (y_min - depth);

    if (j <= depth && k <= y_max + depth)
        update_halo_kernel_2(
            j, k,
            x_min,
            x_max,
            y_min,
            y_max,
            chunk_neighbours,
            tile_neighbours,
            density0,
            density1,
            energy0,
            energy1,
            pressure,
            viscosity,
            soundspeed,
            xvel0,
            yvel0,
            xvel1,
            yvel1,
            vol_flux_x,
            mass_flux_x,
            vol_flux_y,
            mass_flux_y,
            fields,
            depth);
}


void update_local_halo(struct tile_type tile, int* chunk_neighbours, int* fields, int depth)
{
    int x_min = tile.t_xmin,
        x_max = tile.t_xmax,
        y_min = tile.t_ymin,
        y_max = tile.t_ymax;

    int* d_chunk_neighbours = NULL;
    gpuErrchk(cudaMalloc(&d_chunk_neighbours, sizeof(int) * 4));

    gpuErrchk(cudaMemcpy(
                  d_chunk_neighbours,
                  chunk_neighbours,
                  4 * sizeof(int),
                  cudaMemcpyHostToDevice));

    int* d_tile_neighbours = NULL;
    gpuErrchk(cudaMalloc(&d_tile_neighbours, sizeof(int) * 4));

    gpuErrchk(cudaMemcpy(
                  d_tile_neighbours,
                  tile.tile_neighbours,
                  4 * sizeof(int),
                  cudaMemcpyHostToDevice));

    int* d_fields = NULL;
    gpuErrchk(cudaMalloc(&d_fields, sizeof(int) * NUM_FIELDS));

    gpuErrchk(cudaMemcpy(
                  d_fields,
                  fields,
                  NUM_FIELDS * sizeof(int),
                  cudaMemcpyHostToDevice));

    dim3 size = numBlocks(
                    dim3((x_max + depth) - (x_min - depth) + 1,
                         (depth - 1) + 1),
                    update_halo_blocksize);
    update_halo_1_kernel <<< size, update_halo_blocksize >>> (
        x_min, x_max,
        y_min, y_max,
        d_chunk_neighbours,
        d_tile_neighbours,
        tile.field.d_density0,
        tile.field.d_density1,
        tile.field.d_energy0,
        tile.field.d_energy1,
        tile.field.d_pressure,
        tile.field.d_viscosity,
        tile.field.d_soundspeed,
        tile.field.d_xvel0,
        tile.field.d_yvel0,
        tile.field.d_xvel1,
        tile.field.d_yvel1,
        tile.field.d_vol_flux_x,
        tile.field.d_mass_flux_x,
        tile.field.d_vol_flux_y,
        tile.field.d_mass_flux_y,
        d_fields,
        depth);

    dim3 size2 = numBlocks(
                     dim3((depth - 1) + 1,
                          (y_max + depth) - (y_min - depth) + 1),
                     update_halo_blocksize);
    update_halo_2_kernel <<< size2, update_halo_blocksize >>> (
        x_min, x_max,
        y_min, y_max,
        d_chunk_neighbours,
        d_tile_neighbours,
        tile.field.d_density0,
        tile.field.d_density1,
        tile.field.d_energy0,
        tile.field.d_energy1,
        tile.field.d_pressure,
        tile.field.d_viscosity,
        tile.field.d_soundspeed,
        tile.field.d_xvel0,
        tile.field.d_yvel0,
        tile.field.d_xvel1,
        tile.field.d_yvel1,
        tile.field.d_vol_flux_x,
        tile.field.d_mass_flux_x,
        tile.field.d_vol_flux_y,
        tile.field.d_mass_flux_y,
        d_fields,
        depth);

    if (profiler_on)
        cudaDeviceSynchronize();
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
                    calcGlobalSize(cl::NDRange((x_max + depth) - (x_min - depth) + 1, depth - 1 + 1),
                                   update_halo_1_local_size),
                    update_halo_1_local_size));

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
                    calcGlobalSize(cl::NDRange(depth - 1 + 1, (y_max + depth) - (y_min - depth) + 1),
                                   update_halo_2_local_size),
                    update_halo_2_local_size));
    if (profiler_on)
        openclQueue.finish();
}
#endif


