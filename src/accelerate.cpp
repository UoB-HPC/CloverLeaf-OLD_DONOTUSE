#include "accelerate.h"
#include "definitions_c.h"
#include "kernels/accelerate_kernel_c.c"
#include "timer_c.h"

void accelerate_kokkos();
void accelerate_openmp();

void accelerate()
{
    double kernel_time = 0.0;
    if (profiler_on) {kernel_time = timer();}

#if defined(USE_KOKKOS)
    accelerate_kokkos();
#else
    accelerate_openmp();
#endif

    if (profiler_on) {profiler.acceleration += timer() - kernel_time;}
}



#if defined(USE_KOKKOS)

#include <Kokkos_Core.hpp>
using namespace Kokkos;

struct accelerate_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    View<double**> xarea,
         yarea,
         volume,
         density0 ,
         pressure ,
         viscosity,
         xvel0,
         yvel0,
         xvel1,
         yvel1;
    double dt;

    accelerate_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        double _dt):
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax), y_min(tile.t_ymin), y_max(tile.t_ymax),
        xarea(*(tile.field.xarea)),
        yarea(*(tile.field.yarea)),
        volume(*(tile.field.volume)),
        density0(*(tile.field.density0)),
        pressure(*(tile.field.pressure)),
        viscosity(*(tile.field.viscosity)),
        xvel0(*(tile.field.xvel0)),
        yvel0(*(tile.field.yvel0)),
        xvel1(*(tile.field.xvel1)),
        yvel1(*(tile.field.yvel1)),
        dt(_dt)
    {}

    void compute()
    {
        parallel_for(TeamPolicy<>(y_to - y_from + 1, Kokkos::AUTO), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TeamPolicy<>::member_type const& member) const
    {
        const int y = member.league_rank();
        parallel_for(TeamThreadRange(member, 0, x_to - x_from + 1), [&](const int& x) {
            int k = y + y_from;
            int j = x + x_from;

            accelerate_kernel_c_(
                j,  k,
                x_min,  x_max,
                y_min,  y_max,
                &xarea,
                &yarea,
                &volume,
                &density0 ,
                &pressure ,
                &viscosity,
                &xvel0,
                &yvel0,
                &xvel1,
                &yvel1,
                dt);
        });
    }
};

#endif

void accelerate_kokkos()
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        accelerate_functor g(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            dt);
        g.compute();
    }
}

void accelerate_openmp()
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        #pragma omp parallel
        {
            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax + 1,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax + 1,
            {
                accelerate_kernel_c_(j, k,
                chunk.tiles[tile].t_xmin, chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin, chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.xarea,
                chunk.tiles[tile].field.yarea,
                chunk.tiles[tile].field.volume,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.pressure,
                chunk.tiles[tile].field.viscosity,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.xvel1,
                chunk.tiles[tile].field.yvel1,
                dt);
            });
        }
    }
}