#include <tuple>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "merge_sort.h"

#define min(a, b) (a < b ? a : b)
// Based on https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu

// numElements is carried out of parseCommandLineArguments this way because
// merge_sort.h fixes its return type to std::tuple<dim3, dim3> (no count).
static int g_numElements = 32;

__host__ std::tuple<dim3, dim3> parseCommandLineArguments(int argc, char** argv) 
{
    int numElements = 32;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'x':
                    toSet = &threadsPerBlock.x;
                    break;
                case 'y':
                    toSet = &threadsPerBlock.y;
                    break;
                case 'z':
                    toSet = &threadsPerBlock.z;
                    break;
                case 'X':
                    toSet = &blocksPerGrid.x;
                    break;
                case 'Y':
                    toSet = &blocksPerGrid.y;
                    break;
                case 'Z':
                    toSet = &blocksPerGrid.z;
                    break;
                case 'n':
                    i++;
                    numElements = std::stoi(argv[i]);
                    break;
            }
            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
    }

    g_numElements = numElements;
    return {threadsPerBlock, blocksPerGrid};
}

// NOTE: seeding is no longer done here (time(0) has 1-second resolution, so
// calling srand(time(0)) inside a fast loop produces the SAME seed, and
// therefore the SAME "random" array, on every iteration). Seeding is now
// done once per-array by the caller (see main), using a value that changes
// on every iteration, so each of the batch's arrays is genuinely distinct.
__host__ long *generateRandomLongArray(int numElements)
{
    long *randomLongs = (long *)malloc(numElements * sizeof(long));
    if (randomLongs == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory for random array!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; i++)
    {
        randomLongs[i] = rand() % 1000;
    }

    return randomLongs;
}

__host__ void printHostMemory(long *host_mem, int num_elments)
{
    for(int i = 0; i < num_elments; i++)
    {
        printf("%ld ",host_mem[i]);
    }
    printf("\n");
}

__host__ int main(int argc, char** argv) 
{
    auto[threadsPerBlock, blocksPerGrid] = parseCommandLineArguments(argc, argv);
    int numElements = g_numElements;

    int numArrays = 200; // batch of 200 independent arrays to demonstrate GPU execution at scale

    for (int run = 0; run < numArrays; run++)
    {
        // Seed varies per-array (mixing wall-clock time with the loop index)
        // so each of the 200 arrays is genuinely different, even though many
        // iterations happen within the same wall-clock second.
        srand((unsigned int)(time(0)) * 2654435761u + run);

        long *data = generateRandomLongArray(numElements);

        printf("Array %d - Unsorted data: ", run);
        printHostMemory(data, numElements);

        mergesort(data, numElements, threadsPerBlock, blocksPerGrid);

        printf("Array %d - Sorted data: ", run);
        printHostMemory(data, numElements);

        free(data);
    }

    return 0;
}

__host__ std::tuple <long* ,long* ,dim3* ,dim3*> allocateMemory(int numElements)
{
    long *D_data, *D_swp;
    dim3 *D_threads, *D_blocks;

    cudaMalloc((void**) &D_data, numElements * sizeof(long));
    cudaMalloc((void**) &D_swp, numElements * sizeof(long));

    cudaMalloc((void**) &D_threads, sizeof(dim3));
    cudaMalloc((void**) &D_blocks, sizeof(dim3));

    return {D_data, D_swp, D_threads, D_blocks};
}

__host__ void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    auto[D_data, D_swp, D_threads, D_blocks] = allocateMemory((int)size);

    cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    printf("Kernel execution time: %f ms\n", elapsedMs);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost);

    cudaFree(D_data);
    cudaFree(D_swp);
    cudaFree(D_threads);
    cudaFree(D_blocks);
}

__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);

    long start = width * idx * slices;
    long middle, end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);

        gpu_bottomUpMerge(source, dest, start, middle, end);

        start += width;
    }
}

__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;

    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}
