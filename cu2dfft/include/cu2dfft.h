#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <chrono>
#include <curand.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#define WARP_SIZE (32)
#define RTX_3090_MAX_BLOCKSIZE 1024
#define RTX_3090_SM_NUM 82
#define RTX_2070MQ_SM_NUM 36
