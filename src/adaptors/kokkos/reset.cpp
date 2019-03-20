
#include "../../kernels/reset_field_kernel_c.cc"


#include <Kokkos_Core.hpp>
using namespace Kokkos;

struct reset_field_functor {
    int x_from, x_to, y_from, y_to;
    int x_min, x_max, y_min, y_max;
    field_2d_lt density0, density1,
                energy0, energy1,
                xvel0, yvel0,
                xvel1, yvel1;

    reset_field_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to
    ):
        x_from(_x_from), x_to(_x_to),
        y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),

        density0((tile.field.d_density0)), density1((tile.field.d_density1)),
        energy0((tile.field.d_energy0)), energy1((tile.field.d_energy1)),
        xvel0((tile.field.d_xvel0)), yvel0((tile.field.d_yvel0)),
        xvel1((tile.field.d_xvel1)), yvel1((tile.field.d_yvel1))
    {}

    void compute()
    {
        parallel_for("reset_field", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+2, x_to+2}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {
            reset_field_kernel_c_(
                j, k,
                x_min, x_max, y_min, y_max,
                density0,
                density1,
                energy0,
                energy1,
                xvel0,
                xvel1,
                yvel0,
                yvel1);
    }
};
