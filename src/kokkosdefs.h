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

#define T3ACCESS(d, x, y) (d)((y) - (y_min-2), (x) - (x_min-2))
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

#define FIELD_1D(d, i, j) (d)((i) - (j))

// #define const_field_2d_t     cf2t
// #define field_2d_t           f2t

// #define const_field_1d_t     cf1t
// #define field_1d_t           f1t

// #define const_field_2d_t     const Kokkos::View<double**, T>*
// #define field_2d_t           const Kokkos::View<double**, T>*

// #define const_field_1d_t     const Kokkos::View<double*, T>*
// #define field_1d_t           const Kokkos::View<double*, T>*

#define const_field_2d_lt    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>
#define field_2d_lt          const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>

#define const_field_1d_lt    const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>
#define field_1d_lt          const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>

#define flag_t               int*

#if defined(__NVCC__)
#define kernelqual   template< \
    typename const_field_2d_t = Kokkos::View<double**>, \
    typename field_2d_t = Kokkos::View<double**>, \
    typename const_field_1d_t = Kokkos::View<double*>, \
    typename field_1d_t = Kokkos::View<double*> \
    > __device__ __host__
#else
#define kernelqual
#endif

struct field_type {
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_density0;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_density1;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_energy0;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_energy1;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_pressure;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_viscosity;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_soundspeed;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_xvel0;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_xvel1;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_yvel0;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_yvel1;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_vol_flux_x;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_mass_flux_x;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_vol_flux_y;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_mass_flux_y;
    //node_fluxd_; stepbymass; volume_change; pre_vo
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_work_array1;
    //node_massd__post; post_vol
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_work_array2;
    //node_massd__pre; pre_mass
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_work_array3;
    //advec_veld_; post_mass
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_work_array4;
    //mom_flux;d_ advec_vol
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_work_array5;
    //pre_vol; d_post_ener
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_work_array6;
    //post_vol;d_ ener_flux
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_work_array7;

    const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> d_cellx;
    const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> d_celly;
    const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> d_vertexx;
    const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> d_vertexy;
    const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> d_celldx;
    const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> d_celldy;
    const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> d_vertexdx;
    const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> d_vertexdy;

    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_volume;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_xarea;
    const Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space> d_yarea;


    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror density0;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror density1;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror energy0;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror energy1;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror pressure;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror viscosity;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror soundspeed;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror xvel0;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror xvel1;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror yvel0;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror yvel1;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror vol_flux_x;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror mass_flux_x;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror vol_flux_y;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror mass_flux_y;
    //node_flux; stepbymass; volume_change; pre_vo
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror work_array1;
    //node_mass_post; post_vol
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror work_array2;
    //node_mass_pre; pre_mass
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror work_array3;
    //advec_vel; post_mass
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror work_array4;
    //mom_flux; advec_vol
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror work_array5;
    //pre_vol; post_ener
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror work_array6;
    //post_vol; ener_flux
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror work_array7;

    typename Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror cellx;
    typename Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror celly;
    typename Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror vertexx;
    typename Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror vertexy;
    typename Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror celldx;
    typename Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror celldy;
    typename Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror vertexdx;
    typename Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror vertexdy;

    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror volume;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror xarea;
    typename Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>::HostMirror yarea;
};