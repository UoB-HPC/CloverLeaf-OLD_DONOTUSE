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

    openclContext = cl::Context({default_device});

    cl::Program::Sources sources;
    int NUMFILES = 3;
    std::string files[NUMFILES] = {
        "./src/openclaccessdefs.h",

        "./src/kernels/accelerate_kernel.c",
        "./src/adaptors/opencl/accelerate.c"

    };
    std::stringstream buffer;
    for (int i = 0; i < NUMFILES; i++) {
        std::ifstream t(files[i]);
        buffer << t.rdbuf();
    }
    std::string kernel_code = buffer.str();
    // std::cout << kernel_code << std::endl;

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    openclProgram = cl::Program(openclContext, sources);
    if (openclProgram.build({default_device}) != CL_SUCCESS) {
        std::cout << " Error building: " << openclProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }
    openclQueue = cl::CommandQueue(openclContext, default_device);
    // exit(0);
}