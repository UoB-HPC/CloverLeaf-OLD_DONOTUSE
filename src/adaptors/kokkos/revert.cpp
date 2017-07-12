#include "../../kernels/revert_kernel_c.cc"
#include <Kokkos_Core.hpp>
using namespace Kokkos;

struct revert_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    field_2d_lt density0, density1, energy0, energy1;

    revert_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to
    ):
        x_from(_x_from), x_to(_x_to),
        y_from(_y_from), y_to(_y_to),

        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),

        density0((tile.field.d_density0)),
        density1((tile.field.d_density1)),
        energy0((tile.field.d_energy0)),
        energy1((tile.field.d_energy1))
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

            revert_kernel_c_(
                j, k,
                x_min, x_max, y_min, y_max,
                density0,
                density1,
                energy0,
                energy1);
        });
    }
};
