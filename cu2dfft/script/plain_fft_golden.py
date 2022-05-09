import numpy as np
import math

CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

DECIMATION_IN_INPUT = 1
DECIMATION_IN_OUTPUT = -1

VERBOSE = True


def debug_print(*args):
    if VERBOSE:
        print(*args)


# code from https://stackoverflow.com/a/54992207/5555077
def norm(vector):
    return np.sqrt(sum(x * np.conjugate(x) for x in vector))


def cosine_similarity(vec_a, vec_b):
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    dot = sum(a * np.conjugate(b) for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)


def is_power_of_two(n):
    return n & (n - 1) == 0


def bit_reverse(input_scalar, log_2_length):
    # from https://stackoverflow.com/a/58575689/5555077
    return int(format(input_scalar, '0%db' % log_2_length)[::-1], 2)


def bit_reversal_permutation(data, log_2_length):
    assert (len(data.shape) == 1)
    result_data = np.zeros(len(data), dtype=np.complex64)
    for i in range(len(data)):
        result_data[i] = data[bit_reverse(i, log_2_length)]
        # debug_print("bit-reverse(", i, bit_reverse(i, log_2_length), ")")
    return result_data


def ifft_rescale(data, log_2_length):
    result_data = data / (2 ** log_2_length)
    return result_data


def fft_root_lookup(i, N, fft_direction):
    result = np.exp(2j * np.pi * fft_direction * i / N)
    return result


def order_preserving_input_index_difference(stage_idx, log_2_length):
    # In the first stage (stage_idx 0) the result is N/2. In the second stage it is N/4. etc.
    return 2 ** (log_2_length - 1 - stage_idx)


def order_preserving_input_pair_first_index_to_output_pair_first_index(stage_idx, log_2_length, input_pair_first_index):
    # In the first stage it is {0,1}X...X mapped to {0,1}X...X. In the second stage it is X{0,1}X...X mapped to {0,1}X...X. etc.
    beginning_bits_with_significance_descended = (input_pair_first_index >> (log_2_length - stage_idx)) << (
            log_2_length - stage_idx - 1)
    ending_bits_with_significance_preserved = input_pair_first_index & (((1 << (log_2_length - stage_idx))) - 1)
    return beginning_bits_with_significance_descended + ending_bits_with_significance_preserved


def order_preserving_butterfly1d_stage(fft_direction, in_data, stage_idx, log_2_length):
    # the input is in bit-reversed order if decimation_flag is DECIMATION_IN_INPUT, or the output is in bit-reversed order if decimation_flag is DECIMATION_IN_OUTPUT
    assert (len(in_data.shape) == 1)
    out_data = np.zeros(len(in_data), dtype=np.complex64)
    # the input and the processing is the same as decimation in output, except that 1) the output indexing scheme is changed
    # and 2) the scaling factor is now in ordre
    for idx in range(2 ** (log_2_length - 1)):
        idx_reminder = idx % 2 ** (log_2_length - stage_idx - 1)
        idx_quotient = idx // 2 ** (log_2_length - stage_idx - 1)
        element_idx_0 = idx_quotient * (2 ** (log_2_length - stage_idx)) + idx_reminder
        element_idx_1 = idx_quotient * (
                2 ** (log_2_length - stage_idx)) + idx_reminder + order_preserving_input_index_difference(stage_idx,
                                                                                                          log_2_length)
        debug_print("   ", idx, ":(", idx_quotient, idx_reminder, ")", "(", element_idx_0, element_idx_1, 2 ** (
                log_2_length - stage_idx - 1), ")", "(",
                    order_preserving_input_pair_first_index_to_output_pair_first_index(stage_idx, log_2_length,
                                                                                       element_idx_0),
                    order_preserving_input_pair_first_index_to_output_pair_first_index(stage_idx, log_2_length,
                                                                                       element_idx_0) + 2 ** (
                            log_2_length - 1), ")", "(W",
                    idx_quotient, 2 ** (stage_idx + 1),
                    fft_root_lookup(idx_quotient, 2 ** (stage_idx + 1), fft_direction), ")")
        element_0 = in_data[element_idx_0]
        element_1 = in_data[element_idx_1] * fft_root_lookup(idx_quotient,
                                                             2 ** (stage_idx + 1), fft_direction)
        out_data[order_preserving_input_pair_first_index_to_output_pair_first_index(stage_idx, log_2_length,
                                                                                    element_idx_0)] = element_0 + element_1
        out_data[order_preserving_input_pair_first_index_to_output_pair_first_index(stage_idx, log_2_length,
                                                                                    element_idx_0) + 2 ** (
                         log_2_length - 1)] = element_0 - element_1
    return out_data


def naive_butterfly1d_stage(fft_direction, decimation_flag, in_data, stage_idx, log_2_length):
    # the input is in bit-reversed order if decimation_flag is DECIMATION_IN_INPUT, or the output is in bit-reversed order if decimation_flag is DECIMATION_IN_OUTPUT
    assert (len(in_data.shape) == 1)
    out_data = np.zeros(len(in_data), dtype=np.complex64)
    if decimation_flag == DECIMATION_IN_INPUT:
        # butterfly smallest to largest. in each butterfly, for each edge whose index % 2^(stage_idx+1)>=2^(stage_idx) apply scale factor W^{index % 2^(stage_idx+1)-2^stage_idx}_{2^(stage_idx+1)} root before butterfly, and each such edge apply scale -1 during the butterfly.
        # From https://ocw.mit.edu/courses/res-6-008-digital-signal-processing-spring-2011/9e8a5b1ff26b0da76069c4e5b205a0d2_MITRES_6_008S11_lec19.pdf
        # For each i%2^(stage_idx+1)<=2^(stage_idx), output[i] and output[i+2^(stage_idx)] share the same input before any scaling.
        for idx in range(2 ** (log_2_length - 1)):
            idx_reminder = idx % 2 ** (stage_idx)
            idx_quotient = idx // 2 ** (stage_idx)

            element_idx_0 = idx_quotient * (2 ** (stage_idx + 1)) + idx_reminder
            element_idx_1 = idx_quotient * (2 ** (stage_idx + 1)) + idx_reminder + 2 ** (stage_idx)
            debug_print("   ", idx, "(", idx_quotient, idx_reminder, ")", "(", element_idx_0, element_idx_1,
                        2 ** (stage_idx),
                        ")", "(W",
                        idx_reminder, 2 ** (stage_idx + 1),
                        fft_root_lookup(idx_reminder, 2 ** (stage_idx + 1), fft_direction), ")")
            element_0 = in_data[element_idx_0]
            element_1 = in_data[element_idx_1] * fft_root_lookup(idx_reminder, 2 ** (stage_idx + 1), fft_direction)
            out_data[element_idx_0] = element_0 + element_1
            out_data[element_idx_1] = element_0 - element_1
    elif decimation_flag == DECIMATION_IN_OUTPUT:
        # butterfly largest to smallest. in each butterfly, for each edge whose index % 2^(N-stage_idx)>=2^(N-stage_idx-1) apply scale factor W^{index / 2^(N-stage_idx)}_{2^(stage_idx+1)} root before butterfly, and each such edge apply scale -1 during the butterfly.
        # output[i] and output[i+2^(N-stage_idx-1)] share the same input before any scaling.
        # For each i%2^(N-stage_idx)<=2^(N-stage_idx-1), output[i] and output[i+2^(N-stage_idx-1)] share the same input before any scaling.
        # N is stage_num, i.e., log_2_length
        for idx in range(2 ** (log_2_length - 1)):
            idx_reminder = idx % 2 ** (log_2_length - stage_idx - 1)
            idx_quotient = idx // 2 ** (log_2_length - stage_idx - 1)
            element_idx_0 = idx_quotient * (2 ** (log_2_length - stage_idx)) + idx_reminder
            element_idx_1 = idx_quotient * (2 ** (log_2_length - stage_idx)) + idx_reminder + 2 ** (
                    log_2_length - stage_idx - 1)
            debug_print("   ", idx, ":(", idx_quotient, idx_reminder, ")", "(", element_idx_0, element_idx_1, 2 ** (
                    log_2_length - stage_idx - 1), ")", "(W",
                        bit_reverse(idx_quotient, stage_idx), 2 ** (stage_idx + 1),
                        fft_root_lookup(bit_reverse(idx_quotient, stage_idx), 2 ** (stage_idx + 1), fft_direction), ")")
            element_0 = in_data[element_idx_0]
            element_1 = in_data[element_idx_1] * fft_root_lookup(bit_reverse(idx_quotient, stage_idx),
                                                                 2 ** (stage_idx + 1), fft_direction)
            out_data[element_idx_0] = element_0 + element_1
            out_data[element_idx_1] = element_0 - element_1
    return out_data


def order_preserved_fft(input_data, fft_direction):
    output_data = np.zeros(len(input_data), dtype=np.complex64)
    if len(input_data.shape) == 1:
        # 1D FFT
        if not is_power_of_two(len(input_data)):
            raise ValueError("Input data must be a power of two")
        log_2_length = int(np.log2(len(input_data)))
        output_data = input_data
        for stage_idx in range(log_2_length):
            debug_print("stage idx", stage_idx)
            output_data = order_preserving_butterfly1d_stage(fft_direction, output_data, stage_idx, log_2_length)
            debug_print(output_data)
        if fft_direction == CUFFT_INVERSE:
            output_data = ifft_rescale(output_data, log_2_length)
    elif len(input_data.shape) == 2:
        raise NotImplementedError
    return output_data


def get_global_index(block_idx_hi, block_idx_lo, shmem_addr_hi, shmem_addr_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo,
                     NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING):
    return (block_idx_hi << (
            bitwidth_shmem_addr_hi + bitwidth_block_idx_lo + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) | (
                   shmem_addr_hi << (
                   NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + bitwidth_block_idx_lo)) | (
                   block_idx_lo << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) | shmem_addr_lo


def locality_preserved_butterfly1d_stage(shmem_inout_data, log_2_length, fft_direction, decimation_flag,
                                         NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING, processing_bit_significance,
                                         processing_bit_significance_beg, processing_bit_significance_end,
                                         block_idx_hi, block_idx_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo):
    # processing_bit_significance_end is also the significance offset of shared memory outer index to global element index
    assert (DECIMATION_IN_OUTPUT == decimation_flag)

    for element_forloop_idx in range(
            2 ** (bitwidth_shmem_addr_hi + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - 1)):
        # element_forloop_idx is decomposed into (bits_more_significant_than_processing), and (bits_less_significant_than_processing) where processing bit should be in the middle
        # where processing_bit is definitely more significant than least significant bits for coalescing in this if clause
        # then (bits_more_significant_than_processing,processing_bit,bits_less_significant_than_processing) is also decomposed into (element_idx_hi, element_idx_lo)
        if (processing_bit_significance >= NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING):
            # the processing bit is in the outer dimension of shmem_inout_data
            # assert (
            #            bitwidth_shmem_addr_hi == processing_bit_significance_beg - processing_bit_significance_end)
            bits_more_significant_than_processing = element_forloop_idx >> (
                    processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)
            bits_less_significant_than_processing = element_forloop_idx & ((1 << (
                    processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) - 1)
            element_1_idx = (bits_more_significant_than_processing << (
                        processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + 1)) + (
                                    1 << (
                                        processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + bits_less_significant_than_processing
            element_0_idx = (bits_more_significant_than_processing << (
                        processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + 1)) + (
                                    0 << (
                                        processing_bit_significance - processing_bit_significance_end + NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)) + bits_less_significant_than_processing
        else:
            # the processing bit is in the inner dimension of shmem_inout_data
            bits_more_significant_than_processing = element_forloop_idx >> (
                processing_bit_significance)
            bits_less_significant_than_processing = element_forloop_idx & ((1 << (
                processing_bit_significance)) - 1)
            element_1_idx = (bits_more_significant_than_processing << (processing_bit_significance + 1)) + (
                    1 << (processing_bit_significance)) + bits_less_significant_than_processing
            element_0_idx = (bits_more_significant_than_processing << (processing_bit_significance + 1)) + (
                    0 << (processing_bit_significance)) + bits_less_significant_than_processing

        element_0_idx_hi = element_0_idx >> NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING
        element_0_idx_lo = element_0_idx & ((1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) - 1)
        element_1_idx_hi = element_1_idx >> NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING
        element_1_idx_lo = element_1_idx & ((1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING) - 1)
        if processing_bit_significance >= NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING:
            assert (element_0_idx_lo == element_1_idx_lo)
        global_element0_idx = get_global_index(block_idx_hi, block_idx_lo, element_0_idx_hi, element_0_idx_lo,
                                               bitwidth_shmem_addr_hi, bitwidth_block_idx_lo,
                                               NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)
        global_element1_idx = get_global_index(block_idx_hi, block_idx_lo, element_1_idx_hi, element_1_idx_lo,
                                               bitwidth_shmem_addr_hi, bitwidth_block_idx_lo,
                                               NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)
        idx_quotient = global_element0_idx // (2 ** (processing_bit_significance + 1))
        debug_print("   (", block_idx_hi, block_idx_lo, element_forloop_idx, ")", "(", global_element0_idx,
                    global_element1_idx, global_element1_idx - global_element0_idx, ") (BR",
                    idx_quotient, log_2_length - processing_bit_significance - 1, ") (W",
                    bit_reverse(idx_quotient, log_2_length - processing_bit_significance - 1),
                    2 ** (log_2_length - processing_bit_significance),
                    fft_root_lookup(bit_reverse(idx_quotient, log_2_length - processing_bit_significance - 1),
                                    2 ** (log_2_length - processing_bit_significance), fft_direction), ")")
        processed_element_1 = shmem_inout_data[element_1_idx_hi][element_1_idx_lo] * fft_root_lookup(
            bit_reverse(idx_quotient, log_2_length - processing_bit_significance - 1),
            2 ** (log_2_length - processing_bit_significance), fft_direction)
        processed_element_0 = shmem_inout_data[element_0_idx_hi][element_0_idx_lo]

        shmem_inout_data[element_1_idx_hi][element_1_idx_lo] = processed_element_0 - processed_element_1
        shmem_inout_data[element_0_idx_hi][element_0_idx_lo] = processed_element_0 + processed_element_1
    return


def locality_preserved_batch_butterfly1d_per_sm(output_data, input_data, log_2_length, fft_direction,
                                                NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                                NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES, batch_idx, block_idx):
    shmem_data = np.zeros(
        (2 ** NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES, 2 ** NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + 1),
        dtype=np.complex64)  # +1 to reduce bank conflicts when accessing elements with the same inner index
    # first, load the data to the shared memory
    # The global element index is partitioned as (block_idx_hi, shmem_addr_hi, block_idx_lo, shmem_addr_coalescing_lo)
    # where the bitwidth of block_idx_hi equals batch_idx*NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES
    # and the bitwidth of block_idx_lo equals max(0,log_2_length-NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING-(1+batch_idx)*NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES)
    # and the bitwidth of shmem_addr_hi equals min(NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING, log_2_length-NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES*batch_idx-NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)
    bitwidth_block_idx_hi = batch_idx * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES
    bitwidth_block_idx_lo = max(0, log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - (
            1 + batch_idx) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES)
    # bitwidth_shmem_addr_hi = min(NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
    #                             log_2_length - NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES * batch_idx - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)
    num_batches = max(1, (
            log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - 1) // NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES)
    block_idx_hi = block_idx >> bitwidth_block_idx_lo
    block_idx_lo = block_idx & ((1 << bitwidth_block_idx_lo) - 1)

    bitwidth_shmem_addr_hi = NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES
    if batch_idx == num_batches - 1:
        if NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES > log_2_length - (
                num_batches - 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING:
            # the last batch is not full, so we still 1) load as much as we can, i.e., global element id range in
            # (block_idx_hi, shmem_addr_hi, shmem_addr_coalescing_lo)
            # 2） but only process those not processed yet, i.e., lower bits in shmem_addr_hi
            if num_batches == 1:
                bitwidth_shmem_addr_hi = log_2_length - (
                            num_batches - 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING
            else:
                # bitwidth_shmem_addr_hi is still NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES
                # but the workload is reduced
                pass

    for shmem_addr_hi in range(1 << bitwidth_shmem_addr_hi):
        for shmem_addr_lo in range(1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING):  # bitwidth_shmem_addr_lo
            global_idx = get_global_index(block_idx_hi, block_idx_lo, shmem_addr_hi, shmem_addr_lo, bitwidth_shmem_addr_hi,
                                          bitwidth_block_idx_lo, NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)
            shmem_data[shmem_addr_hi][shmem_addr_lo] = input_data[global_idx]

            debug_print("GLOBAL READ SM", block_idx, "  batch", batch_idx, "global element idx ", global_idx, "shmem addr",
                        shmem_addr_hi, shmem_addr_lo)

    # process each radix-2 butterfly stage
    for processing_bit_significance in reversed(range(max(NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                                          log_2_length - (
                                                                  batch_idx + 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES),
                                                      log_2_length - batch_idx * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES)):
        locality_preserved_butterfly1d_stage(shmem_data, log_2_length, fft_direction, DECIMATION_IN_OUTPUT,
                                             NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                             processing_bit_significance,
                                             log_2_length - batch_idx * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES,
                                             max(NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                                 log_2_length - (
                                                             batch_idx + 1) * NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES),
                                             block_idx_hi, block_idx_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo)

    # process bits in the colaescing least siginicificant bits if last batch
    # num_batches = max(1,(log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING+NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES-1) // NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES)
    if batch_idx == num_batches - 1:
        for processing_bit_significance in reversed(range(NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)):
            locality_preserved_butterfly1d_stage(shmem_data, log_2_length, fft_direction, DECIMATION_IN_OUTPUT,
                                                 NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                                 processing_bit_significance,
                                                 NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING, 0,
                                                 block_idx_hi, block_idx_lo, bitwidth_shmem_addr_hi, bitwidth_block_idx_lo)

    # last, copy the data from the shared memory to the output
    for shmem_addr_hi in range(1 << bitwidth_shmem_addr_hi):
        for shmem_addr_lo in range(1 << NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING):
            global_idx = get_global_index(block_idx_hi, block_idx_lo, shmem_addr_hi, shmem_addr_lo, bitwidth_shmem_addr_hi,
                                          bitwidth_block_idx_lo, NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING)
            output_data[global_idx] = shmem_data[shmem_addr_hi][shmem_addr_lo]
            debug_print("GLOBAL WRITE SM", block_idx, "  batch", batch_idx, "global element idx ", global_idx,
                        "shmem addr", shmem_addr_hi, shmem_addr_lo)


def locality_preserved_fft(input_data, fft_direction):
    # based on decimation in output
    NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING = 3
    NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES = 3

    if len(input_data.shape) == 1:
        if not is_power_of_two(len(input_data)):
            raise ValueError("Input data must be a power of two")
        log_2_length = int(np.log2(len(input_data)))
        output_data = np.zeros(len(input_data), dtype=np.complex64)

        bitwidth_block_idx = max(0,
                              log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING - NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES)

        for batch_idx in range(max(1, (
                                              log_2_length - NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING + NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES - 1) // NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES)):

            for block_idx in range(2 ** bitwidth_block_idx):
                if batch_idx >= 1:
                    locality_preserved_batch_butterfly1d_per_sm(output_data, output_data, log_2_length, fft_direction,
                                                                NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                                                NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES, batch_idx,
                                                                block_idx)

                else:
                    locality_preserved_batch_butterfly1d_per_sm(output_data, input_data, log_2_length, fft_direction,
                                                                NUM_BITS_LEAST_SIGNIFICANT_COALESCING_PRESERVING,
                                                                NUM_BITS_IN_A_BATCH_OF_BUTTERFLY_STAGES, batch_idx,
                                                                block_idx)

        output_data = bit_reversal_permutation(output_data, log_2_length)
        if fft_direction == CUFFT_INVERSE:
            output_data = ifft_rescale(output_data, log_2_length)
        return output_data


    elif len(input_data.shape) == 2:
        raise NotImplementedError


def plain_fft(input_data, decimation_flag, fft_direction):
    if len(input_data.shape) == 1:
        # 1D FFT
        if not is_power_of_two(len(input_data)):
            raise ValueError("Input data must be a power of two")
        log_2_length = int(np.log2(len(input_data)))
        output_data = input_data
        if decimation_flag == DECIMATION_IN_INPUT:
            output_data = bit_reversal_permutation(output_data, log_2_length)
        for stage_idx in range(log_2_length):
            debug_print("stage idx", stage_idx)
            output_data = naive_butterfly1d_stage(fft_direction, decimation_flag, output_data, stage_idx, log_2_length)
            debug_print(output_data)
        if decimation_flag == DECIMATION_IN_OUTPUT:
            output_data = bit_reversal_permutation(output_data, log_2_length)
    elif len(input_data.shape) == 2:
        # 2D FFT naive implementation for now
        # TODO: use 2d decomposition
        if not is_power_of_two(input_data.shape[0]) or not is_power_of_two(input_data.shape[1]):
            raise ValueError("Input data must be a power of two")
        log_2_length = int(np.log2(input_data.shape[0]))
        output_data = np.zeros(input_data.shape, dtype=np.complex64)
        for idx in range(input_data.shape[0]):
            output_data[idx, :] = plain_fft(input_data[idx], decimation_flag, fft_direction)
        for col_idx in range(input_data.shape[1]):
            output_data[:, col_idx] = plain_fft(output_data[:, col_idx], decimation_flag, fft_direction)
    if fft_direction == CUFFT_INVERSE:
        output_data = ifft_rescale(output_data, log_2_length)
    return output_data


if __name__ == "__main__":
    test_input1 = np.random.rand((128)).squeeze()  # generate vector with length 10254
    test_golden_fft_result1 = np.fft.fft(test_input1)
    test_golden_ifft_result1 = np.fft.ifft(test_golden_fft_result1)
    test_actual_fft_result1 = plain_fft(test_input1, DECIMATION_IN_INPUT, CUFFT_FORWARD)
    test_actual_ifft_result1 = plain_fft(test_actual_fft_result1, DECIMATION_IN_INPUT, CUFFT_INVERSE)
    test_actual_fft_result1_1 = plain_fft(test_input1, DECIMATION_IN_OUTPUT, CUFFT_FORWARD)
    test_actual_ifft_result1_1 = plain_fft(test_actual_fft_result1_1, DECIMATION_IN_OUTPUT, CUFFT_INVERSE)
    debug_print("starting order_preserved fft")
    test_actual_fft_result1_2 = order_preserved_fft(test_input1, CUFFT_FORWARD)
    debug_print("starting order_preserved ifft")
    test_actual_ifft_result1_2 = order_preserved_fft(test_actual_fft_result1_2, CUFFT_INVERSE)
    print(np.allclose(test_actual_fft_result1, test_actual_fft_result1_1),
          cosine_similarity(test_actual_fft_result1, test_actual_fft_result1_1))
    print(np.allclose(test_actual_ifft_result1, test_actual_ifft_result1_1),
          cosine_similarity(test_actual_ifft_result1, test_actual_ifft_result1_1))
    print(np.allclose(test_golden_ifft_result1, test_actual_ifft_result1),
          cosine_similarity(test_golden_ifft_result1, test_actual_ifft_result1))
    print(np.allclose(test_golden_fft_result1, test_actual_fft_result1),
          cosine_similarity(test_golden_fft_result1, test_actual_fft_result1))
    print(np.allclose(test_golden_ifft_result1, test_actual_ifft_result1_1),
          cosine_similarity(test_golden_ifft_result1, test_actual_ifft_result1_1))
    print(np.allclose(test_golden_fft_result1, test_actual_fft_result1_1),
          cosine_similarity(test_golden_fft_result1, test_actual_fft_result1_1))
    print(np.allclose(test_golden_ifft_result1, test_actual_ifft_result1_2),
          cosine_similarity(test_golden_ifft_result1, test_actual_ifft_result1_2))
    print(np.allclose(test_golden_fft_result1, test_actual_fft_result1_2),
          cosine_similarity(test_golden_fft_result1, test_actual_fft_result1_2))

    debug_print("starting locality_preserved fft")
    test_actual_fft_result1_3 = locality_preserved_fft(test_input1, CUFFT_FORWARD)
    debug_print("starting locality_preserved ifft")
    test_actual_ifft_result1_3 = locality_preserved_fft(test_actual_fft_result1_3, CUFFT_INVERSE)
    print(np.allclose(test_actual_fft_result1, test_actual_fft_result1_3),
          cosine_similarity(test_actual_fft_result1, test_actual_fft_result1_3))
    print(np.allclose(test_actual_ifft_result1, test_actual_ifft_result1_3),
          cosine_similarity(test_actual_ifft_result1, test_actual_ifft_result1_3))
    if 0:
        test_input2 = np.random.rand(128, 128)  # generate matrix with size (1024, 1024)
        test_golden_result2 = np.fft.fft2(test_input2)
        test_golden_ifft_result2 = np.fft.ifft2(test_golden_result2)
        test_actual_result2 = plain_fft(test_input2, DECIMATION_IN_INPUT, CUFFT_FORWARD)
        test_actual_ifft_result2 = plain_fft(test_actual_result2, DECIMATION_IN_INPUT, CUFFT_INVERSE)
        print(np.allclose(test_golden_ifft_result2, test_actual_ifft_result2),
              cosine_similarity(test_golden_ifft_result2.flatten(), test_actual_ifft_result2.flatten()))
        print(np.allclose(test_golden_result2, test_actual_result2),
              cosine_similarity(test_golden_result2.flatten(), test_actual_result2.flatten()))
        test_actual_result2_1 = plain_fft(test_input2, DECIMATION_IN_OUTPUT, CUFFT_FORWARD)
        test_actual_ifft_result2_1 = plain_fft(test_actual_result2_1, DECIMATION_IN_OUTPUT, CUFFT_INVERSE)
        print(np.allclose(test_golden_ifft_result2, test_actual_ifft_result2_1),
              cosine_similarity(test_golden_ifft_result2.flatten(), test_actual_ifft_result2_1.flatten()))
        print(np.allclose(test_golden_result2, test_actual_result2_1),
              cosine_similarity(test_golden_result2.flatten(), test_actual_result2_1.flatten()))
