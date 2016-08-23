
#include <Kokkos_Core.hpp>
#include "../../kernels/ideal_gas_kernel_c.c"

using namespace Kokkos;

// template <
//     typename const_field_2d_t,
//     typename field_2d_t,
//     typename const_field_1d_t,
//     typename field_1d_t
//     >
struct ideal_gas_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt energy,
                density;
    field_2d_lt pressure,
                soundspeed;

    ideal_gas_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        field_2d_lt _density, field_2d_lt _energy):
        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax), y_min(tile.t_ymin), y_max(tile.t_ymax),
        energy(_energy), density(_density),
        pressure((tile.field.d_pressure)), soundspeed((tile.field.d_soundspeed))
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
                density, energy,
                pressure, soundspeed
            );
        });
    }
};

