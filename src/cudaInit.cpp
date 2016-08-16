
void initCuda()
{
    unsigned device_id = 0;
    cudaSetDevice(device_id);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "Using Device " << prop.name << std::endl;
}