#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include "definitions_c.h"

void initOpenCL()
{
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cerr << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    for (int i = 0; i < all_platforms.size(); i++) {
        std::cout << "Platform available: " << all_platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "\nUsing platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cerr << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    for (int i = 0; i < all_devices.size(); i++) {
        std::cout << "Device available: " << all_devices[i].getInfo<CL_DEVICE_NAME>() << "\n";

    }
    // TODO parameterise
    cl::Device default_device = all_devices[0];
    std::cout << "\nUsing device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
    printf("\tECC: %d\n\tGlobal cache size: %d\n\tMax compute units: %d\n\tMax work group size: %d\n",
           default_device.getInfo<CL_DEVICE_ERROR_CORRECTION_SUPPORT>(),
           default_device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>(),
           default_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(),
           default_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
          );

    openclContext = cl::Context(default_device);

    cl::Program::Sources sources;
    std::vector<std::string> files;

    std::string prefix;
    const char * ocl_src_prefix = std::getenv("OCL_SRC_PREFIX");
    if (ocl_src_prefix != NULL) {
        prefix = std::string(ocl_src_prefix);
    } else {
        prefix = ".";
    }

    std::cout << "Loading kernels from " << prefix << std::endl;

    files.push_back(prefix + "/src/openclaccessdefs.h");
    files.push_back(prefix + "/src/kernels/ftocmacros.h");

    files.push_back(prefix + "/src/kernels/accelerate_kernel.cc");
    files.push_back(prefix + "/src/adaptors/opencl/accelerate.c");

    files.push_back(prefix + "/src/kernels/PdV_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/pdv.c");

    files.push_back(prefix + "/src/kernels/ideal_gas_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/ideal_gas.c");

    files.push_back(prefix + "/src/kernels/calc_dt_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/calc_dt.c");

    files.push_back(prefix + "/src/kernels/advec_mom_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/advec_mom.c");

    files.push_back(prefix + "/src/kernels/advec_cell_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/advec_cell.c");

    files.push_back(prefix + "/src/kernels/flux_calc_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/flux_calc.c");

    files.push_back(prefix + "/src/kernels/viscosity_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/viscosity.c");

    files.push_back(prefix + "/src/kernels/revert_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/revert.c");

    files.push_back(prefix + "/src/kernels/reset_field_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/reset_field.c");

    files.push_back(prefix + "/src/kernels/update_halo_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/update_halo.c");

    files.push_back(prefix + "/src/kernels/field_summary_kernel_c.cc");
    files.push_back(prefix + "/src/adaptors/opencl/field_summary.c");

    std::stringstream buffer;
    for (int i = 0; i < files.size(); i++) {
        std::ifstream t(files[i]);
        if (!t.is_open()) {
            std::cerr << "opencl file read error, file " << i << std::endl;
            exit(1);
        }
        buffer << t.rdbuf() << std::endl;
    }
    std::string kernel_code = buffer.str();
    // std::cout << kernel_code << std::endl;

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    char buildOptions[200];
    sprintf(buildOptions,
            "-D x_min_def=%d "
            "-D x_max_def=%d "
            "-D y_min_def=%d "
            "-D y_max_def=%d ",
            chunk.tiles[0].t_xmin,
            chunk.tiles[0].t_xmax,
            chunk.tiles[0].t_ymin,
            chunk.tiles[0].t_ymax);

    openclProgram = cl::Program(openclContext, sources);
    if (openclProgram.build({default_device}, buildOptions) != CL_SUCCESS) {
        std::cerr << " Error building: " << openclProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }
    openclQueue = cl::CommandQueue(openclContext, default_device);
    // exit(0);
}