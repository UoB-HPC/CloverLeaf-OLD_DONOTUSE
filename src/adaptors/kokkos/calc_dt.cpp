
#include <Kokkos_Core.hpp>
using namespace Kokkos;
#include "../../kernels/calc_dt_kernel_c.c"

// struct calc_dt_minx_functor {

//     typedef double value_type;

//     typedef field_2d_lt::size_type size_type;

//     int x_from,
//         x_to,
//         y_from,
//         y_to;
//     int x_min,
//         x_max,
//         y_min,
//         y_max;
//     int k;

//     field_2d_lt xarea,
//            yarea,
//            volume,
//            density0,
//            energy0 ,
//            pressure,
//            viscosity,
//            soundspeed,
//            xvel0,
//            yvel0,
//            dtmin;
//     field_1d_lt celldx,
//            celldy;

//     calc_dt_minx_functor(
//         struct tile_type tile,
//         int _x_from, int _x_to, int _y_from, int _y_to,
//         int _k):

//         x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
//         x_min(tile.t_xmin), x_max(tile.t_xmax),
//         y_min(tile.t_ymin), y_max(tile.t_ymax),

//         xarea((tile.field.d_xarea)),
//         yarea((tile.field.d_yarea)),
//         volume((tile.field.d_volume)),
//         density0((tile.field.d_density0)),
//         energy0((tile.field.d_energy0)),
//         pressure((tile.field.d_pressure)),
//         viscosity((tile.field.d_viscosity)),
//         soundspeed((tile.field.d_soundspeed)),
//         xvel0((tile.field.d_xvel0)),
//         yvel0((tile.field.d_yvel0)),
//         dtmin((tile.field.d_work_array1)),
//         celldx((tile.field.d_celldx)),
//         celldy((tile.field.d_celldy)),
//         k(_k)
//     {}

//     KOKKOS_INLINE_FUNCTION
//     void operator()(const int& x, value_type& update) const
//     {
//         int j = x + x_from;

//         double val = calc_dt_kernel_c_(
//                          j, k,
//                          x_min, x_max, y_min, y_max,
//                     xarea,
//                     yarea,
//                     celldx,
//                     celldy,
//                     volume,
//                     density0,
//                     energy0 ,
//                     pressure,
//                     viscosity,
//                     soundspeed,
//                     xvel0,
//                     yvel0,
//                     dtmin);
//         // printf("\t## %e\n", val);
//         if (val < update)
//             update = val;
//         // printf("\t#2 %e\n", update);
//     }

//     KOKKOS_INLINE_FUNCTION
//     void join(volatile value_type& dst,
//               const volatile value_type& src) const
//     {
//         if (src < dst) {
//             dst = src;
//         }
//     }

//     KOKKOS_INLINE_FUNCTION
//     void init(value_type& dst) const
//     {
//         dst = g_big;
//     }
// };

struct calc_dt_functor {

    typedef double value_type;

    typedef field_2d_lt::size_type size_type;

    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt xarea,
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
    field_1d_lt celldx,
                celldy;
    double g_big;

    struct tile_type tile;

    calc_dt_functor(
        struct tile_type _tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        double _g_big):

        tile(_tile),
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(_tile.t_xmin), x_max(_tile.t_xmax),
        y_min(_tile.t_ymin), y_max(_tile.t_ymax),
        g_big(_g_big),

        xarea((_tile.field.d_xarea)),
        yarea((_tile.field.d_yarea)),
        volume((_tile.field.d_volume)),
        density0((_tile.field.d_density0)),
        energy0((_tile.field.d_energy0)),
        pressure((_tile.field.d_pressure)),
        viscosity((_tile.field.d_viscosity)),
        soundspeed((_tile.field.d_soundspeed)),
        xvel0((_tile.field.d_xvel0)),
        yvel0((_tile.field.d_yvel0)),
        dtmin((_tile.field.d_work_array1)),
        celldx((_tile.field.d_celldx)),
        celldy((_tile.field.d_celldy))
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

        double result = update;
        for (int j = tile.t_xmin; j <= tile.t_xmax; j++) {
            double val = calc_dt_kernel_c_(
                             j, k,
                             x_min, x_max, y_min, y_max,
                             xarea,
                             yarea,
                             celldx,
                             celldy,
                             volume,
                             density0,
                             energy0 ,
                             pressure,
                             viscosity,
                             soundspeed,
                             xvel0,
                             yvel0,
                             dtmin);
            if (val < result)
                result = val;
        }
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
