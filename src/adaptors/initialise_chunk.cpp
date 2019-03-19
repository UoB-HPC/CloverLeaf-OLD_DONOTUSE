
#if defined(USE_OPENMP) || defined(USE_OMPSS)

#include "../kernels/initialise_chunk_kernel_c.cc"
#include "../definitions_c.h"

void initialise_chunk(int tile)
{
    double xmin, ymin, dx, dy;

    dx = (grid.xmax - grid.xmin) / (float)grid.x_cells;
    dy = (grid.ymax - grid.ymin) / (float)grid.y_cells;

    xmin = grid.xmin + dx * (float)(chunk.tiles[tile].t_left - 1);
    ymin = grid.ymin + dx * (float)(chunk.tiles[tile].t_bottom - 1);

    initialise_chunk_kernel_c_(
        chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax,
        chunk.tiles[tile].t_ymin,
        chunk.tiles[tile].t_ymax,
        xmin, ymin, dx, dy,
        chunk.tiles[tile].field.vertexx,
        chunk.tiles[tile].field.vertexdx,
        chunk.tiles[tile].field.vertexy,
        chunk.tiles[tile].field.vertexdy,
        chunk.tiles[tile].field.cellx,
        chunk.tiles[tile].field.celldx,
        chunk.tiles[tile].field.celly,
        chunk.tiles[tile].field.celldy,
        chunk.tiles[tile].field.volume,
        chunk.tiles[tile].field.xarea,
        chunk.tiles[tile].field.yarea);
}

#endif

#if defined(USE_KOKKOS)

#include "../definitions_c.h"

// #undef field_2d_t
// #undef field_1d_t

// #define field_2d_t   const Kokkos::View<double**, Kokkos::HostSpace::memory_space>*
// #define field_1d_t   const Kokkos::View<double*, Kokkos::HostSpace::memory_space>*

#include "../kernels/initialise_chunk_kernel_c.cc"

void initialise_chunk(int tile)
{
    double xmin, ymin, dx, dy;

    dx = (grid.xmax - grid.xmin) / (float)grid.x_cells;
    dy = (grid.ymax - grid.ymin) / (float)grid.y_cells;

    xmin = grid.xmin + dx * (float)(chunk.tiles[tile].t_left - 1);
    ymin = grid.ymin + dx * (float)(chunk.tiles[tile].t_bottom - 1);

    initialise_chunk_kernel_c_(
        chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax,
        chunk.tiles[tile].t_ymin,
        chunk.tiles[tile].t_ymax,
        xmin, ymin, dx, dy,
        chunk.tiles[tile].field.vertexx,
        chunk.tiles[tile].field.vertexdx,
        chunk.tiles[tile].field.vertexy,
        chunk.tiles[tile].field.vertexdy,
        chunk.tiles[tile].field.cellx,
        chunk.tiles[tile].field.celldx,
        chunk.tiles[tile].field.celly,
        chunk.tiles[tile].field.celldy,
        chunk.tiles[tile].field.volume,
        chunk.tiles[tile].field.xarea,
        chunk.tiles[tile].field.yarea);

    Kokkos::deep_copy(chunk.tiles[tile].field.d_vertexx, chunk.tiles[tile].field.vertexx);
    Kokkos::deep_copy(chunk.tiles[tile].field.d_vertexdx, chunk.tiles[tile].field.vertexdx);
    Kokkos::deep_copy(chunk.tiles[tile].field.d_vertexy, chunk.tiles[tile].field.vertexy);
    Kokkos::deep_copy(chunk.tiles[tile].field.d_vertexdy, chunk.tiles[tile].field.vertexdy);
    Kokkos::deep_copy(chunk.tiles[tile].field.d_cellx, chunk.tiles[tile].field.cellx);
    Kokkos::deep_copy(chunk.tiles[tile].field.d_celldx, chunk.tiles[tile].field.celldx);
    Kokkos::deep_copy(chunk.tiles[tile].field.d_celly, chunk.tiles[tile].field.celly);
    Kokkos::deep_copy(chunk.tiles[tile].field.d_celldy, chunk.tiles[tile].field.celldy);
    Kokkos::deep_copy(chunk.tiles[tile].field.d_volume, chunk.tiles[tile].field.volume);
    Kokkos::deep_copy(chunk.tiles[tile].field.d_xarea, chunk.tiles[tile].field.xarea);
    Kokkos::deep_copy(chunk.tiles[tile].field.d_yarea, chunk.tiles[tile].field.yarea);
}

#endif

#if defined(USE_CUDA)

#include "../kernels/initialise_chunk_kernel_c.cc"
#include "../definitions_c.h"

void initialise_chunk(int tile)
{
    double xmin, ymin, dx, dy;

    dx = (grid.xmax - grid.xmin) / (float)grid.x_cells;
    dy = (grid.ymax - grid.ymin) / (float)grid.y_cells;

    xmin = grid.xmin + dx * (float)(chunk.tiles[tile].t_left - 1);
    ymin = grid.ymin + dx * (float)(chunk.tiles[tile].t_bottom - 1);

    initialise_chunk_kernel_c_(
        chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax,
        chunk.tiles[tile].t_ymin,
        chunk.tiles[tile].t_ymax,
        xmin, ymin, dx, dy,
        chunk.tiles[tile].field.vertexx,
        chunk.tiles[tile].field.vertexdx,
        chunk.tiles[tile].field.vertexy,
        chunk.tiles[tile].field.vertexdy,
        chunk.tiles[tile].field.cellx,
        chunk.tiles[tile].field.celldx,
        chunk.tiles[tile].field.celly,
        chunk.tiles[tile].field.celldy,
        chunk.tiles[tile].field.volume,
        chunk.tiles[tile].field.xarea,
        chunk.tiles[tile].field.yarea);

    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_vertexx,
                         chunk.tiles[tile].field.vertexx,
                         chunk.tiles[tile].field.vertexx_size * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_vertexdx,
                         chunk.tiles[tile].field.vertexdx,
                         chunk.tiles[tile].field.vertexdx_size * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_vertexy,
                         chunk.tiles[tile].field.vertexy,
                         chunk.tiles[tile].field.vertexy_size * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_vertexdy,
                         chunk.tiles[tile].field.vertexdy,
                         chunk.tiles[tile].field.vertexdy_size * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_cellx,
                         chunk.tiles[tile].field.cellx,
                         chunk.tiles[tile].field.cellx_size * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_celldx,
                         chunk.tiles[tile].field.celldx,
                         chunk.tiles[tile].field.celldx_size * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_celly,
                         chunk.tiles[tile].field.celly,
                         chunk.tiles[tile].field.celly_size * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_celldy,
                         chunk.tiles[tile].field.celldy,
                         chunk.tiles[tile].field.celldy_size * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_volume,
                         chunk.tiles[tile].field.volume,
                         chunk.tiles[tile].field.volume_size * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_xarea,
                         chunk.tiles[tile].field.xarea,
                         chunk.tiles[tile].field.xarea_size * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(chunk.tiles[tile].field.d_yarea,
                         chunk.tiles[tile].field.yarea,
                         chunk.tiles[tile].field.yarea_size * sizeof(double),
                         cudaMemcpyHostToDevice));

    if (profiler_on)
        cudaDeviceSynchronize();
}

#endif

#if defined(USE_OPENCL)

#include "../kernels/initialise_chunk_kernel_c.cc"
#include "../definitions_c.h"

void initialise_chunk(int tile)
{
    double xmin, ymin, dx, dy;

    mapoclmem(chunk.tiles[tile].field.d_vertexx,
              chunk.tiles[tile].field.vertexx,
              chunk.tiles[tile].field.vertexx_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_vertexdx,
              chunk.tiles[tile].field.vertexdx,
              chunk.tiles[tile].field.vertexdx_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_vertexy,
              chunk.tiles[tile].field.vertexy,
              chunk.tiles[tile].field.vertexy_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_vertexdy,
              chunk.tiles[tile].field.vertexdy,
              chunk.tiles[tile].field.vertexdy_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_cellx,
              chunk.tiles[tile].field.cellx,
              chunk.tiles[tile].field.cellx_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_celldx,
              chunk.tiles[tile].field.celldx,
              chunk.tiles[tile].field.celldx_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_celly,
              chunk.tiles[tile].field.celly,
              chunk.tiles[tile].field.celly_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_celldy,
              chunk.tiles[tile].field.celldy,
              chunk.tiles[tile].field.celldy_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_volume,
              chunk.tiles[tile].field.volume,
              chunk.tiles[tile].field.volume_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_xarea,
              chunk.tiles[tile].field.xarea,
              chunk.tiles[tile].field.xarea_size,
              CL_MAP_WRITE);
    mapoclmem(chunk.tiles[tile].field.d_yarea,
              chunk.tiles[tile].field.yarea,
              chunk.tiles[tile].field.yarea_size,
              CL_MAP_WRITE);

    field_2d_t vertexx  = chunk.tiles[tile].field.vertexx ;
    field_2d_t vertexdx = chunk.tiles[tile].field.vertexdx;
    field_2d_t vertexy  = chunk.tiles[tile].field.vertexy ;
    field_2d_t vertexdy = chunk.tiles[tile].field.vertexdy;
    field_2d_t cellx    = chunk.tiles[tile].field.cellx   ;
    field_2d_t celldx   = chunk.tiles[tile].field.celldx  ;
    field_2d_t celly    = chunk.tiles[tile].field.celly   ;
    field_2d_t celldy   = chunk.tiles[tile].field.celldy  ;
    field_2d_t volume   = chunk.tiles[tile].field.volume  ;
    field_2d_t xarea    = chunk.tiles[tile].field.xarea   ;
    field_2d_t yarea    = chunk.tiles[tile].field.yarea   ;

    dx = (grid.xmax - grid.xmin) / (float)grid.x_cells;
    dy = (grid.ymax - grid.ymin) / (float)grid.y_cells;

    xmin = grid.xmin + dx * (float)(chunk.tiles[tile].t_left - 1);
    ymin = grid.ymin + dx * (float)(chunk.tiles[tile].t_bottom - 1);

    initialise_chunk_kernel_c_(
        chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax,
        chunk.tiles[tile].t_ymin,
        chunk.tiles[tile].t_ymax,
        xmin, ymin, dx, dy,
        vertexx,
        vertexdx,
        vertexy,
        vertexdy,
        cellx,
        celldx,
        celly,
        celldy,
        volume,
        xarea,
        yarea);

    unmapoclmem(chunk.tiles[tile].field.d_yarea,
                chunk.tiles[tile].field.yarea);
    unmapoclmem(chunk.tiles[tile].field.d_xarea,
                chunk.tiles[tile].field.xarea);
    unmapoclmem(chunk.tiles[tile].field.d_volume,
                chunk.tiles[tile].field.volume);
    unmapoclmem(chunk.tiles[tile].field.d_celldy,
                chunk.tiles[tile].field.celldy);
    unmapoclmem(chunk.tiles[tile].field.d_celly,
                chunk.tiles[tile].field.celly);
    unmapoclmem(chunk.tiles[tile].field.d_celldx,
                chunk.tiles[tile].field.celldx);
    unmapoclmem(chunk.tiles[tile].field.d_cellx,
                chunk.tiles[tile].field.cellx);
    unmapoclmem(chunk.tiles[tile].field.d_vertexdy,
                chunk.tiles[tile].field.vertexdy);
    unmapoclmem(chunk.tiles[tile].field.d_vertexy,
                chunk.tiles[tile].field.vertexy);
    unmapoclmem(chunk.tiles[tile].field.d_vertexdx,
                chunk.tiles[tile].field.vertexdx);
    unmapoclmem(chunk.tiles[tile].field.d_vertexx,
                chunk.tiles[tile].field.vertexx);
    if (profiler_on)
        openclQueue.finish();
}

#endif
