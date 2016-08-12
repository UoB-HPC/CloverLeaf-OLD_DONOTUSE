

#if defined(USE_KOKKOS)
#include "../kernels/generate_chunk_kernel_c.c"
void generate_chunk(
    int tile,
    struct chunk_type chunk,
    double* state_density,
    double* state_energy,
    double* state_xvel,
    double* state_yvel,
    double* state_xmin,
    double* state_xmax,
    double* state_ymin,
    double* state_ymax,
    double* state_radius,
    int* state_geometry)
{
    for (int k = chunk.tiles[tile].t_ymin - 2; k <= chunk.tiles[tile].t_ymax + 2; k++) {
        for (int j = chunk.tiles[tile].t_xmin - 2; j <= chunk.tiles[tile].t_xmax + 2; j++) {
            generate_chunk_1_kernel(
                j, k,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                state_energy,
                state_density,
                state_xvel,
                state_yvel
            );
        }
    }

    /* State 1 is always the background state */
    for (int state = 2; state <= number_of_states; state++) {
        double x_cent = state_xmin[FTNREF1D(state, 1)];
        double y_cent = state_ymin[FTNREF1D(state, 1)];

        for (int k = chunk.tiles[tile].t_ymin - 2; k <= chunk.tiles[tile].t_ymax + 2; k++) {
            for (int j = chunk.tiles[tile].t_xmin - 2; j <= chunk.tiles[tile].t_xmax + 2; j++) {
                generate_chunk_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    x_cent, y_cent,
                    chunk.tiles[tile].field.vertexx,
                    chunk.tiles[tile].field.vertexy,
                    chunk.tiles[tile].field.cellx,
                    chunk.tiles[tile].field.celly,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.energy0,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.yvel0,
                    number_of_states,
                    state,
                    state_density,
                    state_energy,
                    state_xvel,
                    state_yvel,
                    state_xmin,
                    state_xmax,
                    state_ymin,
                    state_ymax,
                    state_radius,
                    state_geometry);
            };
        }
    }
}
#endif


#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include "../kernels/generate_chunk_kernel_c.c"
void generate_chunk(
    int tile,
    struct chunk_type chunk,
    double* state_density,
    double* state_energy,
    double* state_xvel,
    double* state_yvel,
    double* state_xmin,
    double* state_xmax,
    double* state_ymin,
    double* state_ymax,
    double* state_radius,
    int* state_geometry)
{
    #pragma omp parallel
    {

        DOUBLEFOR(chunk.tiles[tile].t_ymin - 2,
        chunk.tiles[tile].t_ymax + 2,
        chunk.tiles[tile].t_xmin - 2,
        chunk.tiles[tile].t_xmax + 2, {
            generate_chunk_1_kernel(
                j, k,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                state_energy,
                state_density,
                state_xvel,
                state_yvel
            );
        });

        /* State 1 is always the background state */
        for (int state = 2; state <= number_of_states; state++)
        {
            double x_cent = state_xmin[FTNREF1D(state, 1)];
            double y_cent = state_ymin[FTNREF1D(state, 1)];


            DOUBLEFOR(chunk.tiles[tile].t_ymin - 2,
            chunk.tiles[tile].t_ymax + 2,
            chunk.tiles[tile].t_xmin - 2,
            chunk.tiles[tile].t_xmax + 2, {
                generate_chunk_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    x_cent, y_cent,
                    chunk.tiles[tile].field.vertexx,
                    chunk.tiles[tile].field.vertexy,
                    chunk.tiles[tile].field.cellx,
                    chunk.tiles[tile].field.celly,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.energy0,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.yvel0,
                    number_of_states,
                    state,
                    state_density,
                    state_energy,
                    state_xvel,
                    state_yvel,
                    state_xmin,
                    state_xmax,
                    state_ymin,
                    state_ymax,
                    state_radius,
                    state_geometry);
            };
        });
    }
}
#endif

#if defined(USE_CUDA)
#include "../kernels/generate_chunk_kernel_c.c"

void generate_chunk(
    int tile,
    struct chunk_type chunk,
    double* state_density,
    double* state_energy,
    double* state_xvel,
    double* state_yvel,
    double* state_xmin,
    double* state_xmax,
    double* state_ymin,
    double* state_ymax,
    double* state_radius,
    int* state_geometry)
{
    DOUBLEFOR(chunk.tiles[tile].t_ymin - 2,
              chunk.tiles[tile].t_ymax + 2,
              chunk.tiles[tile].t_xmin - 2,
    chunk.tiles[tile].t_xmax + 2, {
        generate_chunk_1_kernel(
            j, k,
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.energy0,
            chunk.tiles[tile].field.density0,
            chunk.tiles[tile].field.xvel0,
            chunk.tiles[tile].field.yvel0,
            state_energy,
            state_density,
            state_xvel,
            state_yvel
        );
    });

    /* State 1 is always the background state */
    for (int state = 2; state <= number_of_states; state++) {
        double x_cent = state_xmin[FTNREF1D(state, 1)];
        double y_cent = state_ymin[FTNREF1D(state, 1)];


        DOUBLEFOR(chunk.tiles[tile].t_ymin - 2,
                  chunk.tiles[tile].t_ymax + 2,
                  chunk.tiles[tile].t_xmin - 2,
        chunk.tiles[tile].t_xmax + 2, {
            generate_chunk_kernel_c_(
                j, k,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                x_cent, y_cent,
                chunk.tiles[tile].field.vertexx,
                chunk.tiles[tile].field.vertexy,
                chunk.tiles[tile].field.cellx,
                chunk.tiles[tile].field.celly,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                number_of_states,
                state,
                state_density,
                state_energy,
                state_xvel,
                state_yvel,
                state_xmin,
                state_xmax,
                state_ymin,
                state_ymax,
                state_radius,
                state_geometry);
        };
    });

    gpuErrchk(cudaMemcpy(
                  chunk.tiles[tile].field.d_vertexx,
                  chunk.tiles[tile].field.vertexx,
                  chunk.tiles[tile].field.vertexx_size * sizeof(double),
                  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
                  chunk.tiles[tile].field.d_vertexy,
                  chunk.tiles[tile].field.vertexy,
                  chunk.tiles[tile].field.vertexy_size * sizeof(double),
                  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
                  chunk.tiles[tile].field.d_cellx,
                  chunk.tiles[tile].field.cellx,
                  chunk.tiles[tile].field.cellx_size * sizeof(double),
                  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
                  chunk.tiles[tile].field.d_celly,
                  chunk.tiles[tile].field.celly,
                  chunk.tiles[tile].field.celly_size * sizeof(double),
                  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
                  chunk.tiles[tile].field.d_density0,
                  chunk.tiles[tile].field.density0,
                  chunk.tiles[tile].field.density0_size * sizeof(double),
                  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
                  chunk.tiles[tile].field.d_energy0,
                  chunk.tiles[tile].field.energy0,
                  chunk.tiles[tile].field.energy0_size * sizeof(double),
                  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
                  chunk.tiles[tile].field.d_xvel0,
                  chunk.tiles[tile].field.xvel0,
                  chunk.tiles[tile].field.xvel0_size * sizeof(double),
                  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
                  chunk.tiles[tile].field.d_yvel0,
                  chunk.tiles[tile].field.yvel0,
                  chunk.tiles[tile].field.yvel0_size * sizeof(double),
                  cudaMemcpyHostToDevice));

    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_density0,
    //               chunk.tiles[tile].field.density0,
    //               chunk.tiles[tile].field.density0_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_density1,
    //               chunk.tiles[tile].field.density1,
    //               chunk.tiles[tile].field.density1_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_energy0,
    //               chunk.tiles[tile].field.energy0,
    //               chunk.tiles[tile].field.energy0_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_energy1,
    //               chunk.tiles[tile].field.energy1,
    //               chunk.tiles[tile].field.energy1_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_pressure,
    //               chunk.tiles[tile].field.pressure,
    //               chunk.tiles[tile].field.pressure_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_viscosity,
    //               chunk.tiles[tile].field.viscosity,
    //               chunk.tiles[tile].field.viscosity_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_soundspeed,
    //               chunk.tiles[tile].field.soundspeed,
    //               chunk.tiles[tile].field.soundspeed_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_xvel0,
    //               chunk.tiles[tile].field.xvel0,
    //               chunk.tiles[tile].field.xvel0_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_xvel1,
    //               chunk.tiles[tile].field.xvel1,
    //               chunk.tiles[tile].field.xvel1_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_yvel0,
    //               chunk.tiles[tile].field.yvel0,
    //               chunk.tiles[tile].field.yvel0_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_yvel1,
    //               chunk.tiles[tile].field.yvel1,
    //               chunk.tiles[tile].field.yvel1_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_vol_flux_x,
    //               chunk.tiles[tile].field.vol_flux_x,
    //               chunk.tiles[tile].field.vol_flux_x_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_mass_flux_x,
    //               chunk.tiles[tile].field.mass_flux_x,
    //               chunk.tiles[tile].field.mass_flux_x_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_vol_flux_y,
    //               chunk.tiles[tile].field.vol_flux_y,
    //               chunk.tiles[tile].field.vol_flux_y_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_mass_flux_y,
    //               chunk.tiles[tile].field.mass_flux_y,
    //               chunk.tiles[tile].field.mass_flux_y_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // //node_flux; stepbymass; volume_change; pre_vo
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_work_array1,
    //               chunk.tiles[tile].field.work_array1,
    //               chunk.tiles[tile].field.work_array1_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // //node_mass_post; post_vol
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_work_array2,
    //               chunk.tiles[tile].field.work_array2,
    //               chunk.tiles[tile].field.work_array2_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // //node_mass_pre; pre_mass
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_work_array3,
    //               chunk.tiles[tile].field.work_array3,
    //               chunk.tiles[tile].field.work_array3_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // //advec_vel; post_mass
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_work_array4,
    //               chunk.tiles[tile].field.work_array4,
    //               chunk.tiles[tile].field.work_array4_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // //mom_flux; advec_vol
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_work_array5,
    //               chunk.tiles[tile].field.work_array5,
    //               chunk.tiles[tile].field.work_array5_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // //pre_vol; post_ener
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_work_array6,
    //               chunk.tiles[tile].field.work_array6,
    //               chunk.tiles[tile].field.work_array6_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // //post_vol; ener_flux
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_work_array7,
    //               chunk.tiles[tile].field.work_array7,
    //               chunk.tiles[tile].field.work_array7_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_cellx,
    //               chunk.tiles[tile].field.cellx,
    //               chunk.tiles[tile].field.cellx_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_celly,
    //               chunk.tiles[tile].field.celly,
    //               chunk.tiles[tile].field.celly_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_vertexx,
    //               chunk.tiles[tile].field.vertexx,
    //               chunk.tiles[tile].field.vertexx_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_vertexy,
    //               chunk.tiles[tile].field.vertexy,
    //               chunk.tiles[tile].field.vertexy_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_celldx,
    //               chunk.tiles[tile].field.celldx,
    //               chunk.tiles[tile].field.celldx_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_celldy,
    //               chunk.tiles[tile].field.celldy,
    //               chunk.tiles[tile].field.celldy_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_vertexdx,
    //               chunk.tiles[tile].field.vertexdx,
    //               chunk.tiles[tile].field.vertexdx_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_vertexdy,
    //               chunk.tiles[tile].field.vertexdy,
    //               chunk.tiles[tile].field.vertexdy_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_volume,
    //               chunk.tiles[tile].field.volume,
    //               chunk.tiles[tile].field.volume_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_xarea,
    //               chunk.tiles[tile].field.xarea,
    //               chunk.tiles[tile].field.xarea_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(
    //               chunk.tiles[tile].field.d_yarea,
    //               chunk.tiles[tile].field.yarea,
    //               chunk.tiles[tile].field.yarea_size * sizeof(double),
    //               cudaMemcpyHostToDevice));
}
#endif

#if defined(USE_OPENCL)
#include "../kernels/generate_chunk_kernel_c.c"
#include "../definitions_c.h"

void generate_chunk(
    int tile,
    struct chunk_type chunk,
    double* state_density,
    double* state_energy,
    double* state_xvel,
    double* state_yvel,
    double* state_xmin,
    double* state_xmax,
    double* state_ymin,
    double* state_ymax,
    double* state_radius,
    int* state_geometry)
{
    mapoclmem(chunk.tiles[tile].field.d_energy0,
              chunk.tiles[tile].field.energy0,
              chunk.tiles[tile].field.energy0_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_density0,
              chunk.tiles[tile].field.density0,
              chunk.tiles[tile].field.density0_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_xvel0,
              chunk.tiles[tile].field.xvel0,
              chunk.tiles[tile].field.xvel0_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_yvel0,
              chunk.tiles[tile].field.yvel0,
              chunk.tiles[tile].field.yvel0_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_vertexx,
              chunk.tiles[tile].field.vertexx,
              chunk.tiles[tile].field.vertexx_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_vertexy,
              chunk.tiles[tile].field.vertexy,
              chunk.tiles[tile].field.vertexy_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_cellx,
              chunk.tiles[tile].field.cellx,
              chunk.tiles[tile].field.cellx_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_celly,
              chunk.tiles[tile].field.celly,
              chunk.tiles[tile].field.celly_size,
              CL_MAP_WRITE);

    DOUBLEFOR(
        chunk.tiles[tile].t_ymin - 2,
        chunk.tiles[tile].t_ymax + 2,
        chunk.tiles[tile].t_xmin - 2,
    chunk.tiles[tile].t_xmax + 2, {
        generate_chunk_1_kernel(
            j, k,
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.energy0,
            chunk.tiles[tile].field.density0,
            chunk.tiles[tile].field.xvel0,
            chunk.tiles[tile].field.yvel0,
            state_energy,
            state_density,
            state_xvel,
            state_yvel
        );
    });

    /* State 1 is always the background state */
    for (int state = 2; state <= number_of_states; state++) {
        double x_cent = state_xmin[FTNREF1D(state, 1)];
        double y_cent = state_ymin[FTNREF1D(state, 1)];


        DOUBLEFOR(chunk.tiles[tile].t_ymin - 2,
                  chunk.tiles[tile].t_ymax + 2,
                  chunk.tiles[tile].t_xmin - 2,
        chunk.tiles[tile].t_xmax + 2, {
            generate_chunk_kernel_c_(
                j, k,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                x_cent, y_cent,
                chunk.tiles[tile].field.vertexx,
                chunk.tiles[tile].field.vertexy,
                chunk.tiles[tile].field.cellx,
                chunk.tiles[tile].field.celly,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                number_of_states,
                state,
                state_density,
                state_energy,
                state_xvel,
                state_yvel,
                state_xmin,
                state_xmax,
                state_ymin,
                state_ymax,
                state_radius,
                state_geometry);
        };
    });

    unmapoclmem(chunk.tiles[tile].field.d_energy0,
                chunk.tiles[tile].field.energy0);
    unmapoclmem(chunk.tiles[tile].field.d_density0,
                chunk.tiles[tile].field.density0);
    unmapoclmem(chunk.tiles[tile].field.d_xvel0,
                chunk.tiles[tile].field.xvel0);
    unmapoclmem(chunk.tiles[tile].field.d_yvel0,
                chunk.tiles[tile].field.yvel0);
    unmapoclmem(chunk.tiles[tile].field.d_vertexx,
                chunk.tiles[tile].field.vertexx);
    unmapoclmem(chunk.tiles[tile].field.d_vertexy,
                chunk.tiles[tile].field.vertexy);
    unmapoclmem(chunk.tiles[tile].field.d_cellx,
                chunk.tiles[tile].field.cellx);
    unmapoclmem(chunk.tiles[tile].field.d_celly,
                chunk.tiles[tile].field.celly);
    if (profiler_on)
    openclQueue.finish();
}
#endif