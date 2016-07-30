// #include "./src/kernels/accelerate_kernel_c.c"


void kernel acclerate(
    global const int* A,
    global const int* B,
    global int* C)
{
    int r = test(4);
    C[get_global_id(0)] = r;
}

