
#include <Kokkos_Core.hpp>

#include "../../kernels/update_halo_kernel_c.cc"

using namespace Kokkos;



struct update_halo_functor_1 {

    int x_from, x_to;
    int depth;

    struct tile_type tile;

    field_2d_lt density0, density1;
    field_2d_lt energy0, energy1;
    field_2d_lt pressure, viscosity, soundspeed;
    field_2d_lt xvel0, yvel0, xvel1, yvel1;
    field_2d_lt vol_flux_x, mass_flux_x, vol_flux_y, mass_flux_y;

    Kokkos::View<int*> chunk_neighbours;
    Kokkos::View<int*> tile_neighbours;
    Kokkos::View<int*> fields;

    update_halo_functor_1(
      struct tile_type _tile,
      int _x_from, int _x_to, int _depth,
      Kokkos::View<int*> _chunk_neighbours,
      Kokkos::View<int*> _tile_neighbours,
      Kokkos::View<int*> _fields
    ) :

    tile(_tile),
    x_from(_x_from), x_to(_x_to),
    depth(_depth),
    chunk_neighbours(_chunk_neighbours),
    tile_neighbours(_tile_neighbours),
    fields(_fields),

    density0((tile.field.d_density0)),
    density1((tile.field.d_density1)),
    energy0((tile.field.d_energy0)),
    energy1((tile.field.d_energy1)),
    pressure((tile.field.d_pressure)),
    viscosity((tile.field.d_viscosity)),
    soundspeed((tile.field.d_soundspeed)),
    xvel0((tile.field.d_xvel0)),
    yvel0((tile.field.d_yvel0)),
    xvel1((tile.field.d_xvel1)),
    yvel1((tile.field.d_yvel1)),
    vol_flux_x((tile.field.d_vol_flux_x)),
    mass_flux_x((tile.field.d_mass_flux_x)),
    vol_flux_y((tile.field.d_vol_flux_y)),
    mass_flux_y((tile.field.d_mass_flux_y))

    {}

    void compute()
    {
      parallel_for("update_halo_1", RangePolicy<>(x_from, x_to+1), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int j) const
    {
        for (int k = 1; k <= depth; ++k) {
            update_halo_kernel_1(
                j, k,
                tile.t_xmin,
                tile.t_xmax,
                tile.t_ymin,
                tile.t_ymax,
                chunk_neighbours,
                tile_neighbours,
                density0,
                density1,
                energy0,
                energy1,
                pressure,
                viscosity,
                soundspeed,
                xvel0,
                yvel0,
                xvel1,
                yvel1,
                vol_flux_x,
                mass_flux_x,
                vol_flux_y,
                mass_flux_y,
                fields,
                depth);

         }
    }

};

struct update_halo_functor_2 {

    int y_from, y_to;
    int depth;

    struct tile_type tile;

    field_2d_lt density0, density1;
    field_2d_lt energy0, energy1;
    field_2d_lt pressure, viscosity, soundspeed;
    field_2d_lt xvel0, yvel0, xvel1, yvel1;
    field_2d_lt vol_flux_x, mass_flux_x, vol_flux_y, mass_flux_y;

    Kokkos::View<int*> chunk_neighbours;
    Kokkos::View<int*> tile_neighbours;
    Kokkos::View<int*> fields;

    update_halo_functor_2(
      struct tile_type _tile,
      int _y_from, int _y_to, int _depth,
      Kokkos::View<int*> _chunk_neighbours,
      Kokkos::View<int*> _tile_neighbours,
      Kokkos::View<int*> _fields
    ) :

    tile(_tile),
    y_from(_y_from), y_to(_y_to),
    depth(_depth),
    chunk_neighbours(_chunk_neighbours),
    tile_neighbours(_tile_neighbours),
    fields(_fields),

    density0((tile.field.d_density0)),
    density1((tile.field.d_density1)),
    energy0((tile.field.d_energy0)),
    energy1((tile.field.d_energy1)),
    pressure((tile.field.d_pressure)),
    viscosity((tile.field.d_viscosity)),
    soundspeed((tile.field.d_soundspeed)),
    xvel0((tile.field.d_xvel0)),
    yvel0((tile.field.d_yvel0)),
    xvel1((tile.field.d_xvel1)),
    yvel1((tile.field.d_yvel1)),
    vol_flux_x((tile.field.d_vol_flux_x)),
    mass_flux_x((tile.field.d_mass_flux_x)),
    vol_flux_y((tile.field.d_vol_flux_y)),
    mass_flux_y((tile.field.d_mass_flux_y))

    {}


    void compute()
    {
      parallel_for("update_halo_2", RangePolicy<>(y_from, y_to+1), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k) const
    {
        for (int j = 1; j <= depth; ++j) {
            update_halo_kernel_2(
                j, k,
                tile.t_xmin,
                tile.t_xmax,
                tile.t_ymin,
                tile.t_ymax,
                chunk_neighbours,
                tile_neighbours,
                density0,
                density1,
                energy0,
                energy1,
                pressure,
                viscosity,
                soundspeed,
                xvel0,
                yvel0,
                xvel1,
                yvel1,
                vol_flux_x,
                mass_flux_x,
                vol_flux_y,
                mass_flux_y,
                fields,
                depth);
         }
    }

};
