
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
    double* work_array1;
    //node_mass_post; post_vol
    double* work_array2;
    //node_mass_pre; pre_mass
    double* work_array3;
    //advec_vel; post_mass
    double* work_array4;
    //mom_flux; advec_vol
    double* work_array5;
    //pre_vol; post_ener
    double* work_array6;
    //post_vol; ener_flux
    double* work_array7;
    double* cellx;
    double* celly;
    double* vertexx;
    double* vertexy;
    double* celldx;
    double* celldy;
    double* vertexdx;
    double* vertexdy;
    double* volume;
    double* xarea;
    double* yarea;
};


#define DOUBLEFOR(k_from, k_to, j_from, j_to, body) \
    Kokkos::parallel_for((k_to) - (k_from) + 1, KOKKOS_LAMBDA (const int& i) { \
            int k = i + (k_from); \
        _Pragma("ivdep") \
        for(int j = (j_from); j <= (j_to); j++) { \
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

#define CONSTFIELDPARAM   const Kokkos::View<double**>*
#define FIELDPARAM        Kokkos::View<double**>*

#define T3ACCESS(d, x, y) (*d)(x - x_min, y - y_min)
