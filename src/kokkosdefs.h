
struct field_type {
    Kokkos::View<double**>* density0;
    Kokkos::View<double**>* density1;
    Kokkos::View<double**>* energy0;
    Kokkos::View<double**>* energy1;
    Kokkos::View<double**>* pressure;
    Kokkos::View<double**>* viscosity;
    Kokkos::View<double**>* soundspeed;
    Kokkos::View<double**>* xvel0; Kokkos::View<double**>* xvel1;
    Kokkos::View<double**>* yvel0; Kokkos::View<double**>* yvel1;
    Kokkos::View<double**>* vol_flux_x; Kokkos::View<double**>* mass_flux_x;
    Kokkos::View<double**>* vol_flux_y; Kokkos::View<double**>* mass_flux_y;
    //node_flux; stepbymass; volume_change; pre_vo
    Kokkos::View<double**>* work_array1;
    //node_mass_post; post_vol
    Kokkos::View<double**>* work_array2;
    //node_mass_pre; pre_mass
    Kokkos::View<double**>* work_array3;
    //advec_vel; post_mass
    Kokkos::View<double**>* work_array4;
    //mom_flux; advec_vol
    Kokkos::View<double**>* work_array5;
    //pre_vol; post_ener
    Kokkos::View<double**>* work_array6;
    //post_vol; ener_flux
    Kokkos::View<double**>* work_array7;

    Kokkos::View<double*>* cellx;
    Kokkos::View<double*>* celly;
    Kokkos::View<double*>* vertexx;
    Kokkos::View<double*>* vertexy;
    Kokkos::View<double*>* celldx;
    Kokkos::View<double*>* celldy;
    Kokkos::View<double*>* vertexdx;
    Kokkos::View<double*>* vertexdy;

    double* volume;
    double* xarea;
    double* yarea;
};


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

// outer x's, inner y's
#define DOUBLEFOR(y_from, y_to, x_from, x_to, body) \
    Kokkos::parallel_for((x_to) - (x_from) + 1, KOKKOS_LAMBDA (const int& i) { \
            int j = i + (x_from); \
        for(int k = (y_from); k <= (y_to); k++) { \
            body ;\
        } \
    });

#define KOKKOS_ACCESS(d, x, y) (*d)(x - x_min, y - y_min)

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

#define WORK_ARRAY(d, x, y)    KOKKOS_ACCESS(d, x, y)

#define CONSTFIELDPARAM   const Kokkos::View<double**>*
#define FIELDPARAM        Kokkos::View<double**>*
// const_1d_field_type
// const_2d_field_type
// 1d_field_type
// 2d_field_type
#define T3ACCESS(d, x, y) (*d)(x - x_min, y - y_min)
