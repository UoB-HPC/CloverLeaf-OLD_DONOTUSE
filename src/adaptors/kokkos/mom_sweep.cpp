
#include <Kokkos_Core.hpp>
#include "../../kernels/advec_mom_kernel_c.c"

using namespace Kokkos;

struct mom_sweep_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt pre_vol,
                post_vol,
                volume,
                vol_flux_x,
                vol_flux_y;
    int mom_sweep;

    mom_sweep_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        int _mom_sweep):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        pre_vol((tile.field.d_work_array5)),
        post_vol((tile.field.d_work_array6)),
        volume((tile.field.d_volume)),
        vol_flux_x((tile.field.d_vol_flux_x)),
        vol_flux_y((tile.field.d_vol_flux_y)),
        mom_sweep(_mom_sweep)
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

            if (mom_sweep == 1) {
                ms1(j, k, x_min, x_max, y_min, y_max,
                    pre_vol,
                    post_vol,
                    volume,
                    vol_flux_x,
                    vol_flux_y);
            } else if (mom_sweep == 2) {
                ms2(j, k, x_min, x_max, y_min, y_max,
                    pre_vol,
                    post_vol,
                    volume,
                    vol_flux_x,
                    vol_flux_y);
            } else if (mom_sweep == 3) {
                ms3(j, k, x_min, x_max, y_min, y_max,
                    pre_vol,
                    post_vol,
                    volume,
                    vol_flux_x,
                    vol_flux_y);
            } else if (mom_sweep == 4) {
                ms4(j, k, x_min, x_max, y_min, y_max,
                    pre_vol,
                    post_vol,
                    volume,
                    vol_flux_x,
                    vol_flux_y);
            }
        });
    }
};
