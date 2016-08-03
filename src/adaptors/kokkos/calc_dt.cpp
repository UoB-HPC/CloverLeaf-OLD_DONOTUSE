
#include <Kokkos_Core.hpp>
using namespace Kokkos;
#include "../../kernels/calc_dt_kernel_c.c"

struct calc_dt_minx_functor {

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
    int k;

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

    calc_dt_minx_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        int _k):

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
        celldy(*(tile.field.celldy)),
        k(_k)
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& x, value_type& update) const
    {
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
        printf("\t## %e\n", val);
        if (val < update)
            update = val;
        printf("\t#2 %e\n", update);
    }

    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst,
              const volatile value_type& src) const
    {
        if (src < dst) {
            dst = src;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void init(value_type& dst) const
    {
        dst = g_big;
    }
};

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

    struct tile_type tile;

    calc_dt_functor(
        struct tile_type _tile,
        int _x_from, int _x_to, int _y_from, int _y_to):

        tile(_tile),
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(_tile.t_xmin), x_max(_tile.t_xmax),
        y_min(_tile.t_ymin), y_max(_tile.t_ymax),

        xarea(*(_tile.field.xarea)),
        yarea(*(_tile.field.yarea)),
        volume(*(_tile.field.volume)),
        density0(*(_tile.field.density0)),
        energy0(*(_tile.field.energy0)),
        pressure(*(_tile.field.pressure)),
        viscosity(*(_tile.field.viscosity)),
        soundspeed(*(_tile.field.soundspeed)),
        xvel0(*(_tile.field.xvel0)),
        yvel0(*(_tile.field.yvel0)),
        dtmin(*(_tile.field.work_array1)),
        celldx(*(_tile.field.celldx)),
        celldy(*(_tile.field.celldy))
    {}

    void compute(double& min)
    {
        parallel_reduce(TeamPolicy<>(y_to - y_from + 1, Kokkos::AUTO), *this, min);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TeamPolicy<>::member_type const& member, value_type& update) const
    {
        const int y = member.league_rank();
        int k = y + y_from;

        calc_dt_minx_functor f(tile,
                               tile.t_xmin,
                               tile.t_xmax,
                               tile.t_ymin,
                               tile.t_ymax,
                               k);
        printf("\t# %e\n", update);
        value_type result = update;
        parallel_reduce(TeamThreadRange(member, 0, x_to - x_from + 1), f, result);
        printf("\t# %e\n\n", result);
        if (result < update)
            update = result;
    }

    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst,
              const volatile value_type& src) const
    {
        if (src < dst) {
            dst = src;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void init(value_type& dst) const
    {
        dst = g_big;
    }
};