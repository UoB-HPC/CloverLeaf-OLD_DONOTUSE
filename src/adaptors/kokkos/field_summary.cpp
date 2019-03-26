
#include <Kokkos_Core.hpp>
#include "../../kernels/field_summary_kernel_c.cc"

using namespace Kokkos;

struct field_summary_functor {

  // Structure to do a multi-variable reduction
  typedef struct {
    double vol;
    double mass;
    double ie;
    double ke;
    double press;
  } value_type;

  // Functor data members
  int x_min, x_max, y_min, y_max;

  field_2d_lt volume;
  field_2d_lt density0;
  field_2d_lt energy0;
  field_2d_lt pressure;
  field_2d_lt xvel0;
  field_2d_lt yvel0;

  // Constructor to unpack tile variables
  field_summary_functor(
    int _x_min, int _x_max, int _y_min, int _y_max,
    field_2d_lt _volume, field_2d_lt _density0,
    field_2d_lt _energy0, field_2d_lt _pressure,
    field_2d_lt _xvel0, field_2d_lt _yvel0
    ) :

   x_min(_x_min), x_max(_x_max), y_min(_y_min), y_max(_y_max),
   volume(_volume), density0(_density0),
   energy0(_energy0), pressure(_pressure),
   xvel0(_xvel0), yvel0(_yvel0)
   {}

  // Compute : call the kernel in parallel
  void compute(value_type& result)
  {
   // Need to use a 1D reduction because the 2D reduction resulted in shared memory segfaults
    parallel_reduce("field_summary", (y_max+1-y_min)*(x_max+1-x_min), *this, result);
  }


  // Call the kernel
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type& update) const
  {
    value_type result;
    result.vol = 0.0;
    result.mass = 0.0;
    result.ie = 0.0;
    result.ke = 0.0;
    result.press = 0.0;

    const int j = y_min + (i / (x_max+1-x_min));
    const int k = x_min + (i % (x_max+1-x_min));

    field_summary_kernel_(
        j, k,
        x_min, x_max,
        y_min, y_max,
        volume,
        density0,
        energy0,
        pressure,
        xvel0,
        yvel0,
        &result.vol, &result.mass, &result.ie, &result.ke, &result.press);

    join(update, result);
  }

  // Tell Kokkos how to reduce the structure of doubles
  KOKKOS_INLINE_FUNCTION
  void join(value_type& update, const value_type& input) const
  {
    update.vol   += input.vol;
    update.mass  += input.mass;
    update.ie    += input.ie;
    update.ke    += input.ke;
    update.press += input.press;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& update, const volatile value_type& input) const
  {
    update.vol   += input.vol;
    update.mass  += input.mass;
    update.ie    += input.ie;
    update.ke    += input.ke;
    update.press += input.press;
  }

  // Initial values
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& update)
  {
    update.vol   = 0.0;
    update.mass  = 0.0;
    update.ie    = 0.0;
    update.ke    = 0.0;
    update.press = 0.0;
  }

};

