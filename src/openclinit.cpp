#include "cl.hpp"
#include <fstream>
#include <sstream>
#include <string>

void initOpenCL()
{
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    // TODO parameterise
    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Context context({default_device});

    cl::Program::Sources sources;

    std::string files[3] = {
        "./src/opencldefs.h",
        "./src/kernels/accelerate_kernel.c",
        "./src/adaptors/opencl/accelerate.c"
    };
    std::stringstream buffer;
    for (int i = 0; i < 3; i++) {
        std::ifstream t(files[i]);
        buffer << t.rdbuf();
    }
    std::string kernel_code = buffer.str();
    std::cout << kernel_code << std::endl;

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    if (program.build({default_device}, "-I src/kernels/accelerate_kernel.c") != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }
    exit(0);
}