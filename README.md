# GPU-Accelerated Batch Merge Sort using CUDA

## Overview
This project implements the merge sort algorithm on the GPU using CUDA, then demonstrates execution at scale by running the GPU-based sort across 200 independently-generated random arrays in a single program execution.

Each array is sorted using a bottom-up, iterative merge sort kernel (`gpu_mergesort`), where the input is divided into slices and threads across a configurable grid cooperatively merge increasingly large sorted runs until the full array is sorted. Kernel execution time is measured per array using CUDA events.

## Code Organization

- `merge_sort.cu` / `merge_sort.h` — source code implementing host-side memory management, kernel launch orchestration, and the GPU merge sort kernel itself.
- `Makefile` — builds the project into `merge_sort.exe`.
- `results_batch_run_log.txt` — sample output from running the program: 200 arrays, each showing unsorted input, sorted output, and GPU kernel execution time.
- `helper/helper_cuda.h` — placeholder for the NVIDIA cuda-samples helper header referenced by the build (not used by any code path in this project).

## How to Build
make build
## How to Run
make run
