

/*
// outer y's, inner x's
// #define DOUBLEFOR(y_from, y_to, x_from, x_to, body) \
//     Kokkos::parallel_for((y_to) - (y_from) + 1, KOKKOS_LAMBDA (const int& i) { \
//             int k = i + (y_from); \
//         _Pragma("ivdep") \
//         for(int j = (x_from); j <= (x_to); j++) { \
//             body ;\
//         } \
//     });

// double kokkos - needs policy stuff
// #define DOUBLEFOR(y_from, y_to, x_from, x_to, body) \
//     Kokkos::parallel_for((y_to) - (y_from) + 1, KOKKOS_LAMBDA (const int& i) { \
//         int k = i + (y_from); \
//         Kokkos::parallel_for((x_to) - (x_from) + 1, KOKKOS_LAMBDA (const int& f) { \
//             int j = f + (x_from); \
//             body ;\
//         }); \
//     });
*/


// outer y's, inner x's
#define DOUBLEFOR(y_from, y_to, x_from, x_to, body) \
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, (x_to) - (x_from) + 1), KOKKOS_LAMBDA (const int& i) { \
        int j = i + (x_from); \
        for(int k = (y_from); k <= (y_to); k++) { \
            body ;\
        } \
    });

#define T3ACCESS(d, y, x) (*d)((y) - (y_min-2), (x) - (x_min-2))
#define KOKKOS_ACCESS(d, y, x) T3ACCESS(d, y, x)

#define DENSITY0(d, x, y) KOKKOS_ACCESS(d, x, y)
#define DENSITY1(d, x, y) KOKKOS_ACCESS(d, x, y)

#define ENERGY0(d, x, y)       KOKKOS_ACCESS(d, x, y)
#define ENERGY1(d, x, y)       KOKKOS_ACCESS(d, x, y)
#define PRESSURE(d, x, y)      KOKKOS_ACCESS(d, x, y)
#define VISCOSITY(d, x, y)     KOKKOS_ACCESS(d, x, y)
#define SOUNDSPEED(d, x, y)    KOKKOS_ACCESS(d, x, y)

#define VEL(d, x, y)           KOKKOS_ACCESS(d, x, y)
#define XVEL0(d, x, y)         VEL(d, x, y)
#define XVEL1(d, x, y)         VEL(d, x, y)
#define YVEL0(d, x, y)         VEL(d, x, y)
#define YVEL1(d, x, y)         VEL(d, x, y)

#define VOL_FLUX_X(d, x, y)    KOKKOS_ACCESS(d, x, y)
#define MASS_FLUX_X(d, x, y)   KOKKOS_ACCESS(d, x, y)
#define VOL_FLUX_Y(d, x, y)    KOKKOS_ACCESS(d, x, y)
#define MASS_FLUX_Y(d, x, y)   KOKKOS_ACCESS(d, x, y)

#define VOLUME(d, x, y)        KOKKOS_ACCESS(d, x, y)
#define XAREA(d, x, y)         KOKKOS_ACCESS(d, x, y)
#define YAREA(d, x, y)         KOKKOS_ACCESS(d, x, y)

#define WORK_ARRAY(d, x, y)    KOKKOS_ACCESS(d, x, y)

#define FIELD_1D(d, i, j) (*d)((i) - (j))

#define const_field_2d_t   const Kokkos::View<double**>*
#define field_2d_t         const Kokkos::View<double**>*

#define const_field_1d_t   const Kokkos::View<double*>*
#define field_1d_t         const Kokkos::View<double*>*


struct field_type {
    field_2d_t density0;
    field_2d_t density1;
    field_2d_t energy0;
    field_2d_t energy1;
    field_2d_t pressure;
    field_2d_t viscosity;
    field_2d_t soundspeed;
    field_2d_t xvel0; field_2d_t xvel1;
    field_2d_t yvel0; field_2d_t yvel1;
    field_2d_t vol_flux_x; field_2d_t mass_flux_x;
    field_2d_t vol_flux_y; field_2d_t mass_flux_y;
    //node_flux; stepbymass; volume_change; pre_vo
    field_2d_t work_array1;
    //node_mass_post; post_vol
    field_2d_t work_array2;
    //node_mass_pre; pre_mass
    field_2d_t work_array3;
    //advec_vel; post_mass
    field_2d_t work_array4;
    //mom_flux; advec_vol
    field_2d_t work_array5;
    //pre_vol; post_ener
    field_2d_t work_array6;
    //post_vol; ener_flux
    field_2d_t work_array7;

    field_1d_t cellx;
    field_1d_t celly;
    field_1d_t vertexx;
    field_1d_t vertexy;
    field_1d_t celldx;
    field_1d_t celldy;
    field_1d_t vertexdx;
    field_1d_t vertexdy;

    field_2d_t volume;
    field_2d_t xarea;
    field_2d_t yarea;
};