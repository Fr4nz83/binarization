#pragma once


//================ Macro Definition ===============
#define BITWIDTH 32
#define LOG_BITWIDTH 5
#define CEIL(X) (((X) + BITWIDTH - 1) >> LOG_BITWIDTH) // Equivalente a ceil(X/32)
#define FEIL(X) ((((X) + BITWIDTH - 1) >> LOG_BITWIDTH) << LOG_BITWIDTH) // Equivalente a ceil(X/32)*32 (ovvero, il multiplo di 32 >= X)

#define BITWIDTH64 64
#define LOG_BITWIDTH64 6
#define CEIL64(X) (((X)+BITWIDTH64-1)>>LOG_BITWIDTH64)
#define FEIL64(X) ((((X)+BITWIDTH64-1)>>LOG_BITWIDTH64)<<LOG_BITWIDTH64)


/**
 * CUDA Error report function. This is for debugging purposes.
 *
 * @param code Error code returned by CUDA driver or runtime.
 * @param file File to be written in for the error message.
 * @param line Error position with line number.
 * @param abort Whether abort from the running.
 */
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU_ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
