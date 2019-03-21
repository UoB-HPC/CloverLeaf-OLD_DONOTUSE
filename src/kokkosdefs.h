#define FTNREF1D(i_index,i_lb) \
    ((i_index)-(i_lb))
#define FTNREF2D(i_index, j_index, i_size, i_lb, j_lb) \
    ((i_size) * (j_index - (j_lb)) + \
        (i_index) - (i_lb))



#define DOUBLEFOR(y_from, y_to, x_from, x_to, body) \
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, (y_to) - (y_from) + 1), KOKKOS_LAMBDA (const int& i) { \
        int k = i + (y_from); \
        for(int j = (x_from); j <= (x_to); j++) { \
            body ;\
        } \
    });

#define T3ACCESS(d, x, y)       (d)((y) - (y_min-2), (x) - (x_min-2))
#define T2ACCESS(d, x, y)       (d)((y) - (y_min-2), (x) - (x_min-2))
#define T1ACCESS(d, x, y)       (d)((y) - (y_min-2), (x) - (x_min-2))
#define KOKKOS_ACCESS(d, y, x)  T3ACCESS(d, y, x)

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

#define FIELD_1D(d, i, j) (d)((i) - (j))

#define const_field_2d_lt    const Kokkos::View<double**>
#define field_2d_lt          const Kokkos::View<double**>

#define const_field_1d_lt    const Kokkos::View<double*>
#define field_1d_lt          const Kokkos::View<double*>

#define flag_t               Kokkos::View<int*>

typedef typename Kokkos::View<double**>::HostMirror host_view_2d_t;
typedef typename Kokkos::View<double*>::HostMirror  host_view_1d_t;

#if defined(__NVCC__)

#define kernelqual   template< \
    typename const_field_2d_t = Kokkos::View<double**>, \
    typename field_2d_t = Kokkos::View<double**>, \
    typename const_field_1d_t = Kokkos::View<double*>, \
    typename field_1d_t = Kokkos::View<double*> \
    > __device__ __host__

#else  //defined(__NVCC__)

#define kernelqual   template< \
    typename const_field_2d_t = Kokkos::View<double**>, \
    typename field_2d_t = Kokkos::View<double**>, \
    typename const_field_1d_t = Kokkos::View<double*>, \
    typename field_1d_t = Kokkos::View<double*> \
    typename flag_t = Kokkos::View<int*>
    >

#endif  //defined(__NVCC__)

struct field_type {
    Kokkos::View<double**> d_density0;
    Kokkos::View<double**> d_density1;
    Kokkos::View<double**> d_energy0;
    Kokkos::View<double**> d_energy1;
    Kokkos::View<double**> d_pressure;
    Kokkos::View<double**> d_viscosity;
    Kokkos::View<double**> d_soundspeed;
    Kokkos::View<double**> d_xvel0;
    Kokkos::View<double**> d_xvel1;
    Kokkos::View<double**> d_yvel0;
    Kokkos::View<double**> d_yvel1;
    Kokkos::View<double**> d_vol_flux_x;
    Kokkos::View<double**> d_mass_flux_x;
    Kokkos::View<double**> d_vol_flux_y;
    Kokkos::View<double**> d_mass_flux_y;
    //node_fluxd_; stepbymass; volume_change; pre_vo
    Kokkos::View<double**> d_work_array1;
    //node_massd__post; post_vol
    Kokkos::View<double**> d_work_array2;
    //node_massd__pre; pre_mass
    Kokkos::View<double**> d_work_array3;
    //advec_veld_; post_mass
    Kokkos::View<double**> d_work_array4;
    //mom_flux;d_ advec_vol
    Kokkos::View<double**> d_work_array5;
    //pre_vol; d_post_ener
    Kokkos::View<double**> d_work_array6;
    //post_vol;d_ ener_flux
    Kokkos::View<double**> d_work_array7;

    Kokkos::View<double*> d_cellx;
    Kokkos::View<double*> d_celly;
    Kokkos::View<double*> d_vertexx;
    Kokkos::View<double*> d_vertexy;
    Kokkos::View<double*> d_celldx;
    Kokkos::View<double*> d_celldy;
    Kokkos::View<double*> d_vertexdx;
    Kokkos::View<double*> d_vertexdy;

    Kokkos::View<double**> d_volume;
    Kokkos::View<double**> d_xarea;
    Kokkos::View<double**> d_yarea;


    host_view_2d_t density0;
    host_view_2d_t density1;
    host_view_2d_t energy0;
    host_view_2d_t energy1;
    host_view_2d_t pressure;
    host_view_2d_t viscosity;
    host_view_2d_t soundspeed;
    host_view_2d_t xvel0;
    host_view_2d_t xvel1;
    host_view_2d_t yvel0;
    host_view_2d_t yvel1;
    host_view_2d_t vol_flux_x;
    host_view_2d_t mass_flux_x;
    host_view_2d_t vol_flux_y;
    host_view_2d_t mass_flux_y;
    //node_flux; stepbymass; volume_change; pre_vo
    host_view_2d_t work_array1;
    //node_mass_post; post_vol
    host_view_2d_t work_array2;
    //node_mass_pre; pre_mass
    host_view_2d_t work_array3;
    //advec_vel; post_mass
    host_view_2d_t work_array4;
    //mom_flux; advec_vol
    host_view_2d_t work_array5;
    //pre_vol; post_ener
    host_view_2d_t work_array6;
    //post_vol; ener_flux
    host_view_2d_t work_array7;

    host_view_1d_t cellx;
    host_view_1d_t celly;
    host_view_1d_t vertexx;
    host_view_1d_t vertexy;
    host_view_1d_t celldx;
    host_view_1d_t celldy;
    host_view_1d_t vertexdx;
    host_view_1d_t vertexdy;

    host_view_2d_t volume;
    host_view_2d_t xarea;
    host_view_2d_t yarea;
};


// Can't enable tiles because tile exchange is not implemented for Kokkos
//#define ENABLE_TILES
