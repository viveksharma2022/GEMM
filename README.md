# GEMM
general matrix multiplication

# To compile
nvcc tensor_mul.cu -o tensor_mul -gencode arch=compute_[ARCH],code=sm_[ARCH]<br>
example:<br>
nvcc tensor_mul.cu -o tensor_mul -gencode arch=compute_89,code=sm_89