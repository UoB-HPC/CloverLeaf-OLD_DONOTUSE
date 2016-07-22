#include "ideal_gas.h"
#include "definitions_c.h"
#include "kernels/ideal_gas_kernel_c.c"

void ideal_gas_kokkos(int tile, bool predict);
void ideal_gas_openmp(int tile, bool predict);

void ideal_gas(int tile, bool predict)
{
#if defined(USE_KOKKOS)
    ideal_gas_kokkos(tile, predict);
#else
    ideal_gas_openmp(tile, predict);
#endif
}


#if defined(USE_KOKKOS)

#include <Kokkos_Core.hpp>
using namespace Kokkos;

struct ideal_gas_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    Kokkos::View<double**> energy,
           density;
    Kokkos::View<double**> pressure,
           soundspeed;

    ideal_gas_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        field_2d_t _density, field_2d_t _energy):
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax), y_min(tile.t_ymin), y_max(tile.t_ymax),
        energy(*_energy), density(*_density),
        pressure(*(tile.field.pressure)), soundspeed(*(tile.field.soundspeed))
    {}

    void compute()
    {
        parallel_for(TeamPolicy<>(y_to - y_from + 1, Kokkos::AUTO), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TeamPolicy<>::member_type const& member) const
    {
        const int y = member.league_rank();
        int k = y + y_from;
        parallel_for(TeamThreadRange(member, 0, x_to - x_from + 1), [&](const int& x) {
            int j = x + x_from;

            ideal_gas_kernel_c_(
                j, k,
                x_min, x_max,
                y_min, y_max,
                &density, &energy,
                &pressure, &soundspeed
            );
        });
    }
};


void ideal_gas_kokkos(int tile, bool predict)
{
    if (predict) {
        ideal_gas_functor g(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.density1,
            chunk.tiles[tile].field.energy1);
        g.compute();
    } else {
        ideal_gas_functor g(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.density0,
            chunk.tiles[tile].field.energy0);
        g.compute();
    }
}

#endif
void ideal_gas_openmp(int tile, bool predict)
{
    #pragma omp parallel
    {
        DOUBLEFOR(
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax, {
            if (predict)
            {
                ideal_gas_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.density1,
                    chunk.tiles[tile].field.energy1,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.soundspeed
                );
            } else {
                ideal_gas_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.energy0,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.soundspeed
                );
            }
        });
    }
}

