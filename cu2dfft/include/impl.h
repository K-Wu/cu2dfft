#pragma once

#include "cu2dfft.h"
#define MAX_1D_PLANS 1
#define NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING 3
#define NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES 8 // Also the maximal bitwidth_shmem_addr_hi

typedef int mycu2dfftHandle; // plan handle

struct cu1dfftPlan
{
    int size;
};

int current_plan_id = 0;
struct cu1dfftPlan my_1dplans[MAX_1D_PLANS];
typedef int mycufftHandle;

cufftResult CUFFTAPI mycufftPlan1d(cufftHandle *plan, 
                                 int nx, 
                                 cufftType type, 
                                 int batch){
    if (current_plan_id >= MAX_1D_PLANS){
        return CUFFT_INVALID_PLAN;
    }
    my_1dplans[current_plan_id].size = nx;
    current_plan_id++;
    assert(batch == 1);//not supporting batch mode
    assert(type == CUFFT_C2C);//not supporting other types
    return CUFFT_SUCCESS;
}

__device__ __forceinline__ unsigned int bit_reverse(unsigned int input_scalar, int log_2_length){
    //TODO: memoize to reduce special function unit overhead
    return __brev(input_scalar) >>(32 - log_2_length);
}

__device__ __forceinline__ cuComplex fft_root_lookup(int i, int N, int fft_direction){
    // TODO: memoize
    // TODO: use cuda native way to support complex numbers
    return make_cuComplex(cosf(2*M_PIf32*i/(1<<N)*fft_direction), sinf(2*M_PIf32*i/(1<<N)*fft_direction));
}

__device__ __forceinline__ cuComplex memorized_fft_root_lookup(int i, int N, int fft_direction, int log_2_length, cuComplex* root_look_up_table){
    if (fft_direction==1){
        return root_look_up_table[i+1<<N];
    }
    else if (fft_direction == -1){
        return make_cuComplex(cuCrealf(root_look_up_table[i+1<<N]), -cuCimagf(root_look_up_table[i+1<<N]));
    }
}

__device__ __forceinline__ int get_global_index(int block_idx_hi, int block_idx_lo, int shmem_addr_hi, int shmem_addr_lo, int bitwidth_shmem_addr_hi, int bitwidth_block_idx_lo){
    return (block_idx_hi << (
            bitwidth_shmem_addr_hi + bitwidth_block_idx_lo + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) | (
                   shmem_addr_hi << (
                   NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + bitwidth_block_idx_lo)) | (
                   block_idx_lo << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) | shmem_addr_lo;
}

__device__ __forceinline__ void locality_preserved_butterfly1d_stage(cufftComplex (*shmem_data)[1<<NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING+1], int log_2_length, int fft_direction, int processing_bit_significance, int processing_bit_significance_beg, int processing_bit_significance_end,
                                         int block_idx_hi, int block_idx_lo, int bitwidth_shmem_addr_hi, int bitwidth_block_idx_lo, cuComplex* root_look_up_table){
    for (int element_loop_idx = 0; element_loop_idx< (1<<(bitwidth_shmem_addr_hi + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - 1))/blockDim.x; element_loop_idx++){
        int element_idx = element_loop_idx * blockDim.x + threadIdx.x;
        int bits_more_significant_than_processing, bits_less_significant_than_processing, element_1_idx, element_0_idx;
        if (processing_bit_significance >= NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING){ // TODO: switch to template
            // the processing bit is in the outer dimension of shmem_inout_data
            bits_more_significant_than_processing = element_idx >> (
                    processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
            bits_less_significant_than_processing = element_idx & ((1 << (
                    processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) - 1);
            element_1_idx = (bits_more_significant_than_processing << (
                        processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + 1)) + (
                                    1 << (
                                        processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + bits_less_significant_than_processing;
            element_0_idx = (bits_more_significant_than_processing << (
                        processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + 1)) + (
                                    0 << (
                                        processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + bits_less_significant_than_processing;
        }
        else{
            // the processing bit is in the inner dimension of shmem_inout_data
            bits_more_significant_than_processing = element_idx >> (
                processing_bit_significance);
            bits_less_significant_than_processing = element_idx & ((1 << (
                processing_bit_significance)) - 1);
            element_1_idx = (bits_more_significant_than_processing << (processing_bit_significance + 1)) + (
                    1 << (processing_bit_significance)) + bits_less_significant_than_processing;
            element_0_idx = (bits_more_significant_than_processing << (processing_bit_significance + 1)) + (
                    0 << (processing_bit_significance)) + bits_less_significant_than_processing;
        }
        int element_0_idx_hi = element_0_idx >> NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING;
        int element_0_idx_lo = element_0_idx & ((1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) - 1);
        int element_1_idx_hi = element_1_idx >> NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING;
        int element_1_idx_lo = element_1_idx & ((1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) - 1);
        int global_element0_idx = get_global_index(block_idx_hi, block_idx_lo, element_0_idx_hi, element_0_idx_lo,
                                               bitwidth_shmem_addr_hi, bitwidth_block_idx_lo);
        int global_element1_idx = get_global_index(block_idx_hi, block_idx_lo, element_1_idx_hi, element_1_idx_lo,
                                               bitwidth_shmem_addr_hi, bitwidth_block_idx_lo);
        int idx_quotient = global_element0_idx / (1<< (processing_bit_significance + 1));
        cufftComplex processed_element_1 = cuCmulf(shmem_data[element_1_idx_hi][element_1_idx_lo], memorized_fft_root_lookup(
            bit_reverse(idx_quotient, log_2_length - processing_bit_significance - 1),
            1<< (log_2_length - processing_bit_significance), fft_direction,log_2_length,root_look_up_table));
        cufftComplex processed_element_0 = shmem_data[element_0_idx_hi][element_0_idx_lo];
        shmem_data[element_1_idx_hi][element_1_idx_lo] = cuCsubf(processed_element_0, processed_element_1);
        shmem_data[element_0_idx_hi][element_0_idx_lo] = cuCaddf(processed_element_0, processed_element_1);
    }

}

__global__ void locality_preserved_batch_butterfly1d_per_sm(cufftComplex* d_out, cufftComplex* d_in, int log_2_length, int fft_direction, int batch_idx, cuComplex* root_look_up_table){
    // TODO: separate real and imaginary part to two variables to reduce bank conflict
    __shared__ cufftComplex shmem_data[1<<NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES][1<<NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING+1]; // +1 to reduce bank conflicts when accessing elements with the same inner index
    // first, load the data to the shared memory
    int bitwidth_block_idx_hi = batch_idx * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES;
    int bitwidth_block_idx_lo = max(0, log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - (1+batch_idx)*NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES);
    int num_batches = max(1,(log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING+NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES-1)/NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES);
    int block_idx_hi = blockIdx.x >> bitwidth_block_idx_lo;
    int block_idx_lo = blockIdx.x & ((1<<bitwidth_block_idx_lo)-1);
    int bitwidth_shmem_addr_hi = NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES;
    if (batch_idx == num_batches -1){
        if (NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES > log_2_length - (num_batches - 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING){
            if (num_batches == 1){
                bitwidth_shmem_addr_hi = log_2_length - (num_batches - 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING;
            }
        }
    }

    for (int load_idx = 0; load_idx< (1 << (bitwidth_shmem_addr_hi+NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING))/blockDim.x;load_idx++){
        int shmem_addr_lo = threadIdx.x % (1<<NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
        int shmem_addr_hi = threadIdx.x / (1<<NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
        shmem_addr_hi += (load_idx *blockDim.x) >> NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING;
        int global_idx = get_global_index(block_idx_hi, block_idx_lo, shmem_addr_hi, shmem_addr_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo);
        shmem_data[shmem_addr_hi][shmem_addr_lo] = d_in[global_idx];
    }

    // process each radix-2 butterfly stage
    for (int processing_bit_significance = log_2_length - batch_idx * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES-1; processing_bit_significance >= max(NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                                          log_2_length - (
                                                                  batch_idx + 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES); processing_bit_significance--){
            //TODO: incomplete!
            locality_preserved_butterfly1d_stage(shmem_data, log_2_length, fft_direction,
                                             processing_bit_significance,
                                             log_2_length - batch_idx * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES,
                                             max(NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                                 log_2_length - (
                                                             batch_idx + 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES),
                                             block_idx_hi, block_idx_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo, root_look_up_table);
    }

    // process bits in the colaescing least siginicificant bits if last batch
    if (batch_idx == num_batches -1){
        for (int processing_bit_significance = NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - 1; processing_bit_significance>=0; processing_bit_significance--){
            //TODO: incomplete!
            locality_preserved_butterfly1d_stage(shmem_data, log_2_length, fft_direction,
                                             processing_bit_significance,
                                             NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                             0,
                                             block_idx_hi, block_idx_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo, root_look_up_table);
        }

    }


    // last, copy the data from the shared memory to the output
    for (int load_idx = 0; load_idx< (1 << (bitwidth_shmem_addr_hi+NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING))/blockDim.x;load_idx++){
        int shmem_addr_lo = threadIdx.x % (1<<NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
        int shmem_addr_hi = threadIdx.x / (1<<NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
        shmem_addr_hi += (load_idx *blockDim.x) >> NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING;
        int global_idx = get_global_index(block_idx_hi, block_idx_lo, shmem_addr_hi, shmem_addr_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo);
        d_out[global_idx] = shmem_data[shmem_addr_hi][shmem_addr_lo];
    }

}

// __global__ __device__ void bit_reversal_permutation(cufftComplex * in_data, cufftComplex * out_data, int log_2_length){
//     int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
//     for (int idx = thread_id; idx<length; idx += blockDim.x * gridDim.x){
//         out_data[idx] = in_data[bit_reverse(idx, log_2_length)];
//     }
// }

__global__ void bit_reversal_permutation(cufftComplex *odata, int log_2_length){
    // TODO: finish this GPU version
    return;
}

__global__ void ifft_rescale(cufftComplex *odata, int log_2_length){
    //TODO: finish this GPU version
    return;
}

// TODO: store size in a global struct when creating the plan
// TODO: rejecting non-power-of-2 sizes during plan creation

// corresponding to locality_preserved_fft implementation in plain_fft_golden.py
void mycu1dfftExecC2C(cufftHandle plan,
                      cufftComplex *idata,
                      cufftComplex *odata,
                      int direction)
{
    int log_2_length = log2(my_1dplans[plan].size);
    int bitwidth_block_idx = max(0, log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES);
    cuComplex* root_lookup_table;
    //TODO: check cuda error
    // TODO: put malloc to planning
    cudaMalloc(&root_lookup_table, sizeof(cuComplex) * (1 << (log_2_length+1)));
    for (int batch_idx = 0; batch_idx < max(1, (
                                              log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - 1) / NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES); batch_idx++){
        if (batch_idx>=1){ //TODO: determine thread num
            locality_preserved_batch_butterfly1d_per_sm<<<1<<bitwidth_block_idx,1<<NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES>>>(odata, odata, log_2_length, direction, batch_idx, root_lookup_table);
        }
        else{
            locality_preserved_batch_butterfly1d_per_sm<<<1<<bitwidth_block_idx,1<<NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES>>>(odata, idata, log_2_length, direction, batch_idx, root_lookup_table);
        }
    }
    bit_reversal_permutation<<<1,1>>>(odata, log_2_length);
    if (direction == CUFFT_INVERSE){
        ifft_rescale<<<1,1>>>(odata, log_2_length);
    }
}