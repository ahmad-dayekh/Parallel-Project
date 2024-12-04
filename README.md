This repository contains various implementations of the LeNet-5 Convolutional Neural Network (CNN) for exploring optimization techniques using different paradigms like sequential execution, MPI, OpenMP, and CUDA.

### Files:
- **images.py**: Python script for preprocessing and handling image data.
- **train_lenet5.py**: Python script to train the LeNet-5 model.

### C/CUDA Implementations:
- **lenet_seq.c**: Sequential implementation of LeNet-5 in C.
- **lenet_mpi.c**: MPI-based parallel implementation of LeNet-5 in C for distributed systems.
- **lenet_openmp.c**: OpenMP-based parallel implementation of LeNet-5 in C for shared-memory systems.
- **lenet_cuda.cu**: CUDA-based implementation of LeNet-5 for GPU acceleration.