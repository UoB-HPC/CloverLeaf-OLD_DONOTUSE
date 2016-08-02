
#include <Kokkos_Core.hpp>
using namespace Kokkos;
#include "../../kernels/calc_dt_kernel_c.c"

struct calc_dt_functor {

    typedef double value_type;

    typedef View<double**>::size_type size_type;


    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    Kokkos::View<double**> xarea,
           yarea,
           volume,
           density0,
           energy0 ,
           pressure,
           viscosity,
           soundspeed,
           xvel0,
           yvel0,
           dtmin;
    Kokkos::View<double*> celldx,
           celldy;

    calc_dt_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),

        xarea(*(tile.field.xarea)),
        yarea(*(tile.field.yarea)),
        volume(*(tile.field.volume)),
        density0(*(tile.field.density0)),
        energy0(*(tile.field.energy0)),
        pressure(*(tile.field.pressure)),
        viscosity(*(tile.field.viscosity)),
        soundspeed(*(tile.field.soundspeed)),
        xvel0(*(tile.field.xvel0)),
        yvel0(*(tile.field.yvel0)),
        dtmin(*(tile.field.work_array1)),
        celldx(*(tile.field.celldx)),
        celldy(*(tile.field.celldy))
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
        parallel_reduce(TeamThreadRange(member, 0, x_to - x_from + 1), [&](const int& x, double & update) {
            int j = x + x_from;

            double val = calc_dt_kernel_c_(
                             j, k,
                             x_min, x_max, y_min, y_max,
                             &xarea,
                             &yarea,
                             &celldx,
                             &celldy,
                             &volume,
                             &density0,
                             &energy0 ,
                             &pressure,
                             &viscosity,
                             &soundspeed,
                             &xvel0,
                             &yvel0,
                             &dtmin);

        });
    }
};
