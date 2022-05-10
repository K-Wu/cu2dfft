#pragma once

#include "cu2dfft.h"
#define MAX_1D_PLANS 1
#define NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING 3
#define BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING 4
#define NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES 6 // Also the maximal bitwidth_shmem_addr_hi
#define BLOCKSIZE (1 << (2 + NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES))

typedef int mycu2dfftHandle; // plan handle

struct cu1dfftPlan
{
    int size;
};

int current_plan_id = 0;
struct cu1dfftPlan my_1dplans[MAX_1D_PLANS];
typedef int mycufftHandle;

bool is_power_of_two(int scalar)
{
    return (scalar & (scalar - 1)) == 0;
}

cufftResult CUFFTAPI mycufftPlan1d(cufftHandle *plan,
                                   int nx,
                                   cufftType type,
                                   int batch)
{
    if (current_plan_id >= MAX_1D_PLANS)
    {
        return CUFFT_INVALID_PLAN;
    }
    if (!is_power_of_two(nx))
    {
        return CUFFT_INVALID_SIZE;
    }
    my_1dplans[current_plan_id].size = nx;
    *plan = current_plan_id;
    current_plan_id++;
    assert(batch == 1);        // not supporting batch mode
    assert(type == CUFFT_C2C); // not supporting other types

    return CUFFT_SUCCESS;
}

__device__ __forceinline__ unsigned int bit_reverse(unsigned int input_scalar, int log_2_length)
{
    // TODO: memoize to reduce special function unit overhead
    return __brev(input_scalar) >> (32 - log_2_length);
}

__device__ __forceinline__ cuComplex fft_root_lookup(int i, int N, int fft_direction)
{
    cuComplex result = make_cuComplex(cosf(2 * M_PIf32 * i / N * fft_direction), sinf(2 * M_PIf32 * i / (N)*fft_direction));
    return result;
}

__device__ __forceinline__ cuComplex memorized_fft_root_lookup(int i, int N, int fft_direction, int log_2_length, cuComplex *root_look_up_table)
{
    // if (fft_direction==1){
    //     return root_look_up_table[i+(1<<(N-2))];
    // }
    // else if (fft_direction == -1){
    //     return make_cuComplex(cuCrealf(root_look_up_table[i+(1<<(N-2))]), -cuCimagf(root_look_up_table[i+(1<<(N-2))]));
    // }
    return make_cuComplex(cosf(2 * M_PIf32 * i / N * fft_direction), sinf(2 * M_PIf32 * i / (N)*fft_direction));
}

__device__ __forceinline__ int get_global_index(int block_idx_hi, int block_idx_lo, int shmem_addr_hi, int shmem_addr_lo, int bitwidth_shmem_addr_hi, int bitwidth_block_idx_lo)
{
    // The global element index is partitioned as (block_idx_hi, shmem_addr_hi, block_idx_lo, shmem_addr_coalescing_lo)
    return (block_idx_hi << (bitwidth_shmem_addr_hi + bitwidth_block_idx_lo + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) | (shmem_addr_hi << (NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + bitwidth_block_idx_lo)) | (block_idx_lo << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) | shmem_addr_lo;
}

__device__ __forceinline__ void locality_preserved_butterfly1d_stage(cuComplex (*shmem_data)[1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING], int log_2_length, int fft_direction, int processing_bit_significance, int processing_bit_significance_beg, int processing_bit_significance_end,
                                                                     int block_idx_hi, int block_idx_lo, int bitwidth_shmem_addr_hi, int bitwidth_block_idx_lo, cuComplex *root_look_up_table)
{
    for (int element_loop_idx = 0; element_loop_idx < (1 << (bitwidth_shmem_addr_hi + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - 1)) / BLOCKSIZE; element_loop_idx++)
    {
        int element_idx = element_loop_idx * BLOCKSIZE + threadIdx.x;
        int bits_more_significant_than_processing, bits_less_significant_than_processing, element_1_idx, element_0_idx;
        if (processing_bit_significance >= NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)
        { // TODO: switch to template
            // the processing bit is in the outer dimension of shmem_inout_data
            bits_more_significant_than_processing = element_idx >> (processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
            bits_less_significant_than_processing = element_idx & ((1 << (processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) - 1);
            element_1_idx = (bits_more_significant_than_processing << (processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + 1)) + (1 << (processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + bits_less_significant_than_processing;
            element_0_idx = (bits_more_significant_than_processing << (processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + 1)) + (0 << (processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + bits_less_significant_than_processing;
        }
        else
        {
            // the processing bit is in the inner dimension of shmem_inout_data
            bits_more_significant_than_processing = element_idx >> (processing_bit_significance);
            bits_less_significant_than_processing = element_idx & ((1 << (processing_bit_significance)) - 1);
            element_1_idx = (bits_more_significant_than_processing << (processing_bit_significance + 1)) + (1 << (processing_bit_significance)) + bits_less_significant_than_processing;
            element_0_idx = (bits_more_significant_than_processing << (processing_bit_significance + 1)) + (0 << (processing_bit_significance)) + bits_less_significant_than_processing;
        }
        int element_0_idx_hi = element_0_idx >> NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING;
        int element_0_idx_lo = element_0_idx & ((1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) - 1);
        int element_1_idx_hi = element_1_idx >> NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING;
        int element_1_idx_lo = element_1_idx & ((1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) - 1);
        int global_element0_idx = get_global_index(block_idx_hi, block_idx_lo, element_0_idx_hi, element_0_idx_lo,
                                                   bitwidth_shmem_addr_hi, bitwidth_block_idx_lo);
        int global_element1_idx = get_global_index(block_idx_hi, block_idx_lo, element_1_idx_hi, element_1_idx_lo,
                                                   bitwidth_shmem_addr_hi, bitwidth_block_idx_lo);
        int idx_quotient = global_element0_idx >> (processing_bit_significance + 1);
        // cufftComplex processed_element_1 = cuCmulf(make_cuFloatComplex(shmem_data_real[element_1_idx_hi][element_1_idx_lo],shmem_data_real[element_1_idx_hi][element_1_idx_lo]), memorized_fft_root_lookup(
        //     bit_reverse(idx_quotient, log_2_length - processing_bit_significance - 1),
        //     1<< (log_2_length - processing_bit_significance), fft_direction,log_2_length,root_look_up_table));
        // cufftComplex processed_element_0 = make_cuFloatComplex(shmem_data_real[element_0_idx_hi][element_0_idx_lo],shmem_data_imag[element_0_idx_hi][element_0_idx_lo]);
        cufftComplex processed_element_1 = cuCmulf(shmem_data[element_1_idx_hi][element_1_idx_lo], memorized_fft_root_lookup(
                                                                                                       bit_reverse(idx_quotient, log_2_length - processing_bit_significance - 1),
                                                                                                       1 << (log_2_length - processing_bit_significance), fft_direction, log_2_length, root_look_up_table));
        cufftComplex processed_element_0 = shmem_data[element_0_idx_hi][element_0_idx_lo];
        cufftComplex new_element1 = cuCsubf(processed_element_0, processed_element_1);
        // shmem_data_real[element_1_idx_hi][element_1_idx_lo] = cuCrealf(new_element1);
        // shmem_data_imag[element_1_idx_hi][element_1_idx_lo] = cuCimagf(new_element1);
        cufftComplex new_element0 = cuCaddf(processed_element_0, processed_element_1);
        // shmem_data_real[element_0_idx_hi][element_0_idx_lo] = cuCrealf(new_element0);
        // shmem_data_imag[element_0_idx_hi][element_0_idx_lo] = cuCimagf(new_element0);
        shmem_data[element_1_idx_hi][element_1_idx_lo] = new_element1;
        shmem_data[element_0_idx_hi][element_0_idx_lo] = new_element0;
        // printf("(%d, %d) (%d, %d) W(%d,%d) (%f, %f)\n", global_element0_idx, global_element1_idx, element_0_idx, element_1_idx, bit_reverse(idx_quotient, log_2_length - processing_bit_significance - 1), 1 << (log_2_length - processing_bit_significance), memorized_fft_root_lookup(
        //     bit_reverse(idx_quotient, log_2_length - processing_bit_significance - 1),
        //     1 << (log_2_length - processing_bit_significance), fft_direction, log_2_length, root_look_up_table).x,
        //     memorized_fft_root_lookup(
        //         bit_reverse(idx_quotient, log_2_length - processing_bit_significance - 1),
        //         1 << (log_2_length - processing_bit_significance), fft_direction, log_2_length, root_look_up_table).y);
        __syncthreads(); // TODO: figure out a warpsync strategy
        //__syncwarp();
    }
}

template <bool last_batch>
__global__ void locality_preserved_batch_butterfly1d_per_sm(cufftComplex *d_out, cufftComplex *d_in, int log_2_length, int fft_direction, int batch_idx, int num_batches, cuComplex *root_look_up_table)
{
    // TODO: separate real and imaginary part to two variables to reduce bank conflict
    //__shared__ float shmem_data_real[1<<NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES][1<<NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING];
    //__shared__ float shmem_data_imag[1<<NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES][1<<NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING];
    __shared__ cufftComplex shmem_data[1 << NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES][1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING];
    // first, load the data to the shared memory
    // int bitwidth_block_idx_hi = batch_idx * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES;
    int bitwidth_block_idx_lo;
    // if (batch_idx < num_batches - 1){
    if constexpr (last_batch)
    {
        bitwidth_block_idx_lo = max(0, log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - (1 + batch_idx) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES);
    }
    else
    {
        bitwidth_block_idx_lo = log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - (1 + batch_idx) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES;
    }
    // int bitwidth_block_idx_lo = max(0, log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - (1+batch_idx)*NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES);
    int block_idx_hi = blockIdx.x >> bitwidth_block_idx_lo;
    int block_idx_lo = blockIdx.x & ((1 << bitwidth_block_idx_lo) - 1);
    int bitwidth_shmem_addr_hi = NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES;
    // if (batch_idx == num_batches -1){
    if constexpr (last_batch)
    {
        if (NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES > log_2_length - (num_batches - 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)
        {
            if (num_batches == 1)
            {
                bitwidth_shmem_addr_hi = log_2_length - (num_batches - 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING;
            }
        }
    }

    for (int load_idx = 0; load_idx < ((1 << (bitwidth_shmem_addr_hi + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + BLOCKSIZE - 1) / BLOCKSIZE; load_idx++)
    {
        int shmem_addr_lo = threadIdx.x % (1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
        int shmem_addr_hi = threadIdx.x / (1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
        shmem_addr_hi += ((load_idx * BLOCKSIZE) >> NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
        int global_idx = get_global_index(block_idx_hi, block_idx_lo, shmem_addr_hi, shmem_addr_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo);
        // cufftComplex loaded_data =  d_in[global_idx];
        cufftComplex loaded_data = d_in[global_idx];
        // shmem_data_real[shmem_addr_hi][shmem_addr_lo] = cuCrealf(loaded_data);
        // shmem_data_imag[shmem_addr_hi][shmem_addr_lo] = cuCimagf(loaded_data);
        if (shmem_addr_hi < (1 << (bitwidth_shmem_addr_hi)))
            shmem_data[shmem_addr_hi][shmem_addr_lo] = loaded_data;
    }
    __syncthreads(); // TODO: figure out a warpsync strategy
    //__syncwarp();
    // process each radix-2 butterfly stage
    for (int processing_bit_significance = log_2_length - batch_idx * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - 1; processing_bit_significance >= bitwidth_block_idx_lo + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING; processing_bit_significance--)
    {

        locality_preserved_butterfly1d_stage(shmem_data, log_2_length, fft_direction,
                                             processing_bit_significance,
                                             log_2_length - batch_idx * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES,
                                             max(NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                                 log_2_length - (batch_idx + 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES),
                                             block_idx_hi, block_idx_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo, root_look_up_table);
    }

    // process bits in the colaescing least siginicificant bits if last batch
    // if (batch_idx == num_batches -1){
    if constexpr (last_batch)
    {
        for (int processing_bit_significance = NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - 1; processing_bit_significance >= 0; processing_bit_significance--)
        {
            locality_preserved_butterfly1d_stage(shmem_data, log_2_length, fft_direction,
                                                 processing_bit_significance,
                                                 NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                                 0,
                                                 block_idx_hi, block_idx_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo, root_look_up_table);
        }
    }
    __syncthreads(); // TODO: figure out a warpsync strategy
    // last, copy the data from the shared memory to the output
    for (int store_idx = 0; store_idx < ((1 << (bitwidth_shmem_addr_hi + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + BLOCKSIZE - 1) / BLOCKSIZE; store_idx++)
    {
        int shmem_addr_lo = threadIdx.x % (1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
        int shmem_addr_hi = threadIdx.x / (1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
        shmem_addr_hi += ((store_idx * BLOCKSIZE) >> NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
        int global_idx = get_global_index(block_idx_hi, block_idx_lo, shmem_addr_hi, shmem_addr_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo);
        if (shmem_addr_hi < (1 << (bitwidth_shmem_addr_hi)))
            d_out[global_idx] = shmem_data[shmem_addr_hi][shmem_addr_lo];
        // d_out[global_idx] = make_cuFloatComplex(shmem_data_real[shmem_addr_hi][shmem_addr_lo], shmem_data_imag[shmem_addr_hi][shmem_addr_lo]);
    }
}

// __global__ __device__ void bit_reversal_permutation(cufftComplex * in_data, cufftComplex * out_data, int log_2_length){
//     int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
//     for (int idx = thread_id; idx<length; idx += blockDim.x * gridDim.x){
//         out_data[idx] = in_data[bit_reverse(idx, log_2_length)];
//     }
// }

// #define BITWIDTH_BIT_REVERSAL_PERMUTATION_LOOP (4)
template <bool scaling_flag>
__global__ void bit_reversal_permutation_and_scaling(cufftComplex *odata, cufftComplex *idata, int log_2_length)
{
    // The simplest implementation. blockDim.x == (1<< BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING). gridDim.x == 1<<(log_2_length - 2*BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - BITWIDTH_BIT_REVERSAL_PERMUTATION_LOOP)
    // each block conducts 1<<BITWIDTH_BIT_REVERSAL_PERMUTATION_LOOP iterations
    assert(log_2_length > 2 * BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);

    // not implemented for data size smaller than 2^(BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING+1)
    __shared__ cufftComplex shmem_data[2][1 << BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING][1 + (1 << BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)]; //+1 to reduce bank conflict
    //__shared__ int bit_reverse_lookup[1 << BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING];
    // the global element_index is decomposed into (shmem_addr_hi==threadIdx_hi, blockIdx, loopIdx, shmem_addr_lo == threadIdx_lo)

    int shmem_addr_lo = threadIdx.x % (1 << BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
    int shmem_addr_hi = threadIdx.x / (1 << BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
    // if (shmem_addr_hi == 0)
    // {
    //     bit_reverse_lookup[shmem_addr_lo] = bit_reverse(shmem_addr_lo, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
    // }
    // __syncthreads();

    // const int bit_reverse_lookup[1 << BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

    int reversed_blockIdx = bit_reverse((blockIdx.x), log_2_length - 2 * BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING);
    if (reversed_blockIdx < blockIdx.x)
        return;

    __syncthreads();

    if constexpr (scaling_flag)
    {
        shmem_data[0][bit_reverse(shmem_addr_lo, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)][bit_reverse(shmem_addr_hi, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)] = cuCmulf(make_cuComplex(1.0f / (1 << log_2_length), 0), idata[(shmem_addr_hi << (log_2_length - BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + (reversed_blockIdx << (BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + shmem_addr_lo]);
    }
    else
    {
        shmem_data[0][bit_reverse(shmem_addr_lo, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)][bit_reverse(shmem_addr_hi, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)] = idata[(shmem_addr_hi << (log_2_length - BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + (reversed_blockIdx << (BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + shmem_addr_lo];
    }

    if (reversed_blockIdx > blockIdx.x)
    {

        if constexpr (scaling_flag)
        {
            shmem_data[1][bit_reverse(shmem_addr_lo, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)][bit_reverse(shmem_addr_hi, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)] = cuCmulf(make_cuComplex(1.0f / (1 << log_2_length), 0), idata[(shmem_addr_hi << (log_2_length - BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + (blockIdx.x << (BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + shmem_addr_lo]);
        }
        else
        {
            shmem_data[1][bit_reverse(shmem_addr_lo, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)][bit_reverse(shmem_addr_hi, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)] = idata[(shmem_addr_hi << (log_2_length - BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + (blockIdx.x << (BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + shmem_addr_lo];
        }
    }
    __syncthreads();
    // printf("bit-reverse (%d, %d)\n", (shmem_addr_hi << (log_2_length - BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + (reversed_blockIdx << (BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + shmem_addr_lo,
    //(bit_reverse(shmem_addr_hi, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) << (log_2_length - BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + (blockIdx.x << BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) + bit_reverse(shmem_addr_lo, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING));
    // printf("bit-reverse (%d, %d)\n", (shmem_addr_hi << (log_2_length - BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + (blockIdx.x << (BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + shmem_addr_lo,
    //(bit_reverse(shmem_addr_hi, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) << (log_2_length - BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + (reversed_blockIdx << BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) + bit_reverse(shmem_addr_lo, BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING));

    odata[(shmem_addr_hi << (log_2_length - BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + (blockIdx.x << BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) + shmem_addr_lo] = shmem_data[0][shmem_addr_hi][shmem_addr_lo];
    if (reversed_blockIdx > blockIdx.x)
    {
        odata[(shmem_addr_hi << (log_2_length - BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + (reversed_blockIdx << BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) + shmem_addr_lo] = shmem_data[1][shmem_addr_hi][shmem_addr_lo];
    }

    return;
}

// corresponding to locality_preserved_fft implementation in plain_fft_golden.py
void mycu1dfftExecC2C(cufftHandle plan,
                      cufftComplex *idata,
                      cufftComplex *odata,
                      int direction)
{
    int log_2_length = log2(my_1dplans[plan].size);
    int bitwidth_block_idx = max(0, log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES);
    cuComplex *root_lookup_table;
    // TODO: put malloc and its calculation to planning
    // cudaMalloc(&root_lookup_table, sizeof(cuComplex) * (1 << (log_2_length+1)));
    int num_batches = max(1, (
                                 log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - 1) /
                                 NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES);

    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++)
    {
        if (batch_idx >= 1)
        {
            if (batch_idx == num_batches - 1)
            {
                locality_preserved_batch_butterfly1d_per_sm<true><<<1 << bitwidth_block_idx, BLOCKSIZE>>>(odata, odata, log_2_length, direction, batch_idx, num_batches, nullptr);
            }
            else
            {
                locality_preserved_batch_butterfly1d_per_sm<false><<<1 << bitwidth_block_idx, BLOCKSIZE>>>(odata, odata, log_2_length, direction, batch_idx, num_batches, nullptr);
            }
        }
        else
        {
            if (batch_idx == num_batches - 1)
            {
                locality_preserved_batch_butterfly1d_per_sm<true><<<1 << bitwidth_block_idx, BLOCKSIZE>>>(odata, idata, log_2_length, direction, batch_idx, num_batches, nullptr);
            }
            else
            {
                locality_preserved_batch_butterfly1d_per_sm<false><<<1 << bitwidth_block_idx, BLOCKSIZE>>>(odata, idata, log_2_length, direction, batch_idx, num_batches, nullptr);
            }
        }
        cuda_err_chk(cudaDeviceSynchronize());
    }

    if (direction == CUFFT_INVERSE)
    {
        bit_reversal_permutation_and_scaling<true><<<1 << (log_2_length - 2 * BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING), 1 << (2 * BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)>>>(odata, odata, log_2_length);
    }
    else
    {
        bit_reversal_permutation_and_scaling<false><<<1 << (log_2_length - 2 * BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING), 1 << (2 * BIT_REVERSAL_PERMUTATION_NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)>>>(odata, odata, log_2_length);
    }
}