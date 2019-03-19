
#include <Kokkos_Core.hpp>
// #include "../../kernels/advec_mom_kernel_c.cc"

using namespace Kokkos;

struct mom_direction_x1_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt node_flux,
                mass_flux_x;

    mom_direction_x1_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to):

        x_from(_x_from), x_to(_x_to),
        y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),

        node_flux((tile.field.d_work_array1)),
        mass_flux_x((tile.field.d_mass_flux_x))
    {}

    void compute()
    {
        parallel_for("mom_direction_x1", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+1, x_to+1}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {

            dx1(
                j, k,
                x_min, x_max, y_min, y_max,
                node_flux,
                mass_flux_x);
    }
};


struct mom_direction_y1_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt node_flux,
                mass_flux_y;

    mom_direction_y1_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        mass_flux_y((tile.field.d_mass_flux_y)),
        node_flux((tile.field.d_work_array1))
    {}

    void compute()
    {
        parallel_for("mom_direction_y1", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+1, x_to+1}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {

            dy1(
                j, k,
                x_min, x_max, y_min, y_max,
                node_flux,
                mass_flux_y);
    }
};


struct mom_direction_x2_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt node_flux,
                node_mass_pre,
                post_vol,
                density1,
                node_mass_post;

    mom_direction_x2_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        density1((tile.field.d_density1)),
        node_flux((tile.field.d_work_array1)),
        node_mass_post((tile.field.d_work_array2)),
        node_mass_pre((tile.field.d_work_array3)),
        post_vol((tile.field.d_work_array6))
    {}

    void compute()
    {
        parallel_for("mom_direction_x2", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+1, x_to+1}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {

            dx2(
                j, k,
                x_min, x_max, y_min, y_max,
                node_mass_post,
                node_mass_pre,
                density1,
                post_vol,
                node_flux);
    }
};

struct mom_direction_y2_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt node_flux,
                node_mass_pre,
                post_vol,
                density1,
                node_mass_post;

    mom_direction_y2_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        density1((tile.field.d_density1)),
        node_flux((tile.field.d_work_array1)),
        node_mass_post((tile.field.d_work_array2)),
        node_mass_pre((tile.field.d_work_array3)),
        post_vol((tile.field.d_work_array6))
    {}

    void compute()
    {
        parallel_for("mom_direction_y2", MDRangePolicy<Rank<2>>({y_from, x_from}, {y_to+1, x_to+1}), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int j) const
    {

            dy2(
                j, k,
                x_min, x_max, y_min, y_max,
                node_mass_post,
                node_mass_pre,
                density1,
                post_vol,
                node_flux);
    }
};

struct mom_direction_x3_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt node_flux,
                node_mass_pre,
                mom_flux,
                node_mass_post,
                vel1;

    field_1d_lt celldx;

    mom_direction_x3_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        field_2d_lt _vel1):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        celldx((tile.field.d_celldx)),
        node_flux((tile.field.d_work_array1)),
        node_mass_post((tile.field.d_work_array2)),
        node_mass_pre((tile.field.d_work_array3)),
        mom_flux((tile.field.d_work_array4)),
        vel1(_vel1)
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

            dx3(
                j, k,
                x_min, x_max, y_min, y_max,
                mom_flux,
                node_flux,
                node_mass_pre,
                celldx,
                vel1);
        });
    }
};

struct mom_direction_y3_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt node_flux,
                node_mass_pre,
                mom_flux,
                node_mass_post,
                vel1;

    field_1d_lt celldx;

    mom_direction_y3_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        field_2d_lt _vel1):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        celldx((tile.field.d_celldx)),
        node_flux((tile.field.d_work_array1)),
        node_mass_post((tile.field.d_work_array2)),
        node_mass_pre((tile.field.d_work_array3)),
        mom_flux((tile.field.d_work_array4)),
        vel1(_vel1)
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

            dy3(
                j, k,
                x_min, x_max, y_min, y_max,
                mom_flux,
                node_flux,
                node_mass_pre,
                celldx,
                vel1);
        });
    }
};


struct mom_direction_x4_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt node_mass_pre,
                mom_flux,
                node_mass_post,
                vel1;

    field_1d_lt celldx;

    mom_direction_x4_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        field_2d_lt _vel1):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        celldx((tile.field.d_celldx)),
        node_mass_post((tile.field.d_work_array2)),
        node_mass_pre((tile.field.d_work_array3)),
        mom_flux((tile.field.d_work_array4)),
        vel1(_vel1)
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

            dx4(
                j, k,
                x_min, x_max, y_min, y_max,
                vel1,
                node_mass_pre,
                mom_flux,
                node_mass_post);
        });
    }
};


struct mom_direction_y4_functor {
    int x_from,
        x_to,
        y_from,
        y_to;
    int x_min,
        x_max,
        y_min,
        y_max;
    field_2d_lt node_mass_pre,
                mom_flux,
                node_mass_post,
                vel1;

    field_1d_lt celldx;

    mom_direction_y4_functor(
        struct tile_type tile,
        int _x_from, int _x_to, int _y_from, int _y_to,
        field_2d_lt _vel1):

        x_from(_x_from), x_to(_x_to), y_from(_y_from), y_to(_y_to),
        x_min(tile.t_xmin), x_max(tile.t_xmax),
        y_min(tile.t_ymin), y_max(tile.t_ymax),
        celldx((tile.field.d_celldx)),
        node_mass_post((tile.field.d_work_array2)),
        node_mass_pre((tile.field.d_work_array3)),
        mom_flux((tile.field.d_work_array4)),
        vel1(_vel1)
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

            dy4(
                j, k,
                x_min, x_max, y_min, y_max,
                vel1,
                node_mass_pre,
                mom_flux,
                node_mass_post);
        });
    }
};
