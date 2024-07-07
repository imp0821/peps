#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"

#include <math.h>
#include <time.h>
#include <cblas.h>
//#include "mkl.h"

#include "aqua_configs.h"
#include <omp.h>

extern ArrayType* float4construct_md_array(float *res_data,
                         int size,
                         int ndims,
                         int *dims,
                         int *lbs,
                         Oid elmtype);

extern ArrayType*
float4construct_md_array_ncpy(int size,
                         int ndims,
                         int *dims,
                         int *lbs,
                         Oid elmtype);

void avgpool_core(float4* input_data, float4* avg_data, 
                int out_nelems_channel, int width, int height, 
                int num_channel, int kernel_size, 
                int padding, int stride);

PG_FUNCTION_INFO_V1(maxpool);
Datum 
maxpool(PG_FUNCTION_ARGS)
{      
    int width, height, kernel_size, stride, padding;
    int out_width, out_height, out_nelems, out_nelems_channel;
    int res_idx, res_idx_local;
    ArrayType *feature_map;
    int feature_map_dims[2];
    int lbound[2] = {1, 1};
    int matrix_head;
    int matrix_head_c;
    float pool_max;
    
    ArrayType *input_f = PG_GETARG_ARRAYTYPE_P(0);
    Oid elemtype = input_f->elemtype;
    int *dims = ARR_DIMS(input_f);
    int num_channel = dims[0];
    int nelems_channel = dims[1];
    float4 *data = (float4 *) ARR_DATA_PTR(input_f);

    //im2col
    height = width = sqrt(nelems_channel);
    kernel_size = PG_GETARG_INT32(1);   // default: 2
    padding = PG_GETARG_INT32(2);       // default: 0
    stride = PG_GETARG_INT32(3);        // default: 2
    out_width = (width + 2 * padding - kernel_size)/ stride + 1;
    out_height = (height + 2 * padding - kernel_size)/ stride + 1;
    out_nelems = out_width * out_height * num_channel;
    out_nelems_channel = out_width * out_height; 
    res_idx = 0;

     
    feature_map_dims[0] = num_channel;
    feature_map_dims[1] = out_width * out_height;
    feature_map = float4construct_md_array_ncpy(out_nelems, 2, feature_map_dims, lbound, elemtype);
    float4 *res_data = (float4 *)ARR_DATA_PTR(feature_map);
    // float4 *res_data = (float4 *)palloc(sizeof(float4)*out_nelems);


    omp_set_num_threads(OMP_THREADS);

    if (padding == 0) {
        #pragma omp parallel for private(matrix_head, matrix_head_c, pool_max, res_idx_local)  
        for (int c = 0; c < num_channel; c++) {
            matrix_head_c = nelems_channel * c;
            res_idx_local = c * out_nelems_channel;
            for (int rn = 0; rn < height - kernel_size + 1; rn+= stride) {
                for (int cn = 0; cn < width - kernel_size + 1; cn += stride) {
                    matrix_head = matrix_head_c + rn * width + cn;
                    pool_max = -INFINITY;
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            pool_max = fmaxf(pool_max, data[matrix_head + ky * width + kx]);
                        }
                    }
                    res_data[res_idx_local++] = pool_max;
                }
            }   
        }
    } else {
        #pragma omp parallel for private(matrix_head, matrix_head_c, pool_max, res_idx_local)  
        for (int c = 0; c < num_channel; c++) {
            matrix_head_c = nelems_channel * c;
            res_idx_local = c * out_nelems_channel;
            for (int rn = -padding; rn < height + padding - kernel_size + 1; rn+= stride) {
                for (int cn = -padding; cn < width +padding - kernel_size + 1; cn += stride) {
                    matrix_head = matrix_head_c + rn * width + cn;
                    pool_max = -INFINITY;
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            if (ky + rn >= 0 && ky + rn < height && kx + cn >= 0 && kx + cn < width) {
                                pool_max = fmaxf(pool_max, data[matrix_head + ky * width + kx]);
                            }  
                        }
                    }
                    res_data[res_idx_local++] = pool_max;
                }
            }   
        }
    }
    
    
     
    PG_RETURN_ARRAYTYPE_P(feature_map);
}

PG_FUNCTION_INFO_V1(avgpool);
Datum 
avgpool(PG_FUNCTION_ARGS)
{      
    int width, height, kernel_size, stride, padding, kernel_size_2;
    int out_width, out_height, out_nelems_channel;
    ArrayType *feature_map;
    int feature_map_dims[2];
    int lbound[2] = {1, 1};
    
    ArrayType *input_f = PG_GETARG_ARRAYTYPE_P(0);
    Oid elemtype = input_f->elemtype;
    int *dims = ARR_DIMS(input_f);
    int num_channel = dims[0];
    int nelems_channel = dims[1];
    float4 *data = (float4 *) ARR_DATA_PTR(input_f);

    //im2col
    height = width = sqrt(nelems_channel);
    kernel_size = PG_GETARG_INT32(1);   // default: 2
    padding = PG_GETARG_INT32(2);       // default: 0
    stride = PG_GETARG_INT32(3);        // default: 0
    out_width = (width + 2 * padding - kernel_size)/ stride + 1;
    out_height = (height + 2 * padding - kernel_size)/ stride + 1;
    out_nelems_channel = out_width * out_height; 
    
    //out_nelems = out_width * out_height * num_channel;
    feature_map_dims[0] = num_channel;
    feature_map_dims[1] = out_width * out_height;
    // feature_map = float4construct_md_array_ncpy(out_nelems, 2, feature_map_dims, lbound, elemtype);
    // float4 *res_data = (float4 *)ARR_DATA_PTR(feature_map);
    float4 *res_data = (float4 *)palloc(sizeof(float4)* out_nelems_channel * num_channel);
    avgpool_core(data, res_data, out_nelems_channel, width, height, num_channel, kernel_size, padding, stride);
     
     
    feature_map = float4construct_md_array(res_data, out_nelems_channel * num_channel, 2, feature_map_dims, lbound, elemtype);
    PG_RETURN_ARRAYTYPE_P(feature_map);
}

/*
avgpool_conv(kmatrix real[][],  fmatrix real[][], bias real[], 
    kernel INT = 2, padding INT = 0, stride INT = 2)
*/
PG_FUNCTION_INFO_V1(avgpool_conv);
Datum 
avgpool_conv(PG_FUNCTION_ARGS)
{      
    openblas_set_num_threads(GEMM_THREADS);
    //mkl_set_num_threads(GEMM_THREADS);
    int width, height, kernel_size, stride, padding;
    int out_width, out_height, f_out_nelems_channel;
    
    ArrayType *kmatrix = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *fmatrix = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *bias_vec = PG_GETARG_ARRAYTYPE_P(2);
    int *fmatrix_dims = ARR_DIMS(fmatrix);
    int f_num_channel = fmatrix_dims[0];
    int f_nelems_channel = fmatrix_dims[1];
    float4 *f_data = (float4 *) ARR_DATA_PTR(fmatrix);

    float4 *kmatrix_data = (float4 *) ARR_DATA_PTR(kmatrix);
    int *kmatrix_dims = ARR_DIMS(kmatrix);  //112, 64
    int out_channel = kmatrix_dims[0]; 
    int klen = kmatrix_dims[1];

    //im2col
    height = width = sqrt(f_nelems_channel);
    kernel_size = PG_GETARG_INT32(3);   // default: 2
    padding = PG_GETARG_INT32(4);       // default: 0
    stride = PG_GETARG_INT32(5);        // default: 0

    out_width = (width + 2 * padding - kernel_size)/ stride + 1;
    out_height = (height + 2 * padding - kernel_size)/ stride + 1;
    f_out_nelems_channel = out_width * out_height; 
    float4 *f_avg_data = (float4 *)palloc(sizeof(float4)* f_out_nelems_channel * f_num_channel);
    avgpool_core(f_data, f_avg_data, f_out_nelems_channel, width, height, f_num_channel, kernel_size, padding, stride);
    
    
    // kfm:  k (out_channel, klen) * avg_f (f_num_channel, f_out_nelems_channel)
    if (klen != f_num_channel) {
        elog(ERROR, "[avgpool_conv] The number of matrix columns must equal the number of vector elements");
    }
    ArrayType *res;
    int res_dims[2] = {out_channel, f_out_nelems_channel};
    int lbound[2] = {1, 1};
    float4 *res_data = (float4 *) palloc0(sizeof(float4) * out_channel * f_out_nelems_channel);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_channel, f_out_nelems_channel,
                klen, 1.0, kmatrix_data, klen, f_avg_data, f_out_nelems_channel,
                0.0, res_data, f_out_nelems_channel);

    if (ARR_DIMS(bias_vec)[0] == out_channel) {
        float4 *bias = (float4 *) ARR_DATA_PTR(bias_vec);
        int offset = 0;
        for (int c = 0; c < out_channel; c++) {
            for (int i = offset; i < offset + f_out_nelems_channel; i++) {
                res_data[i] += bias[c];
                //relu
                res_data[i] = fmax(0, res_data[i]);
            }
            offset += f_out_nelems_channel;
        }
    } else {
        // elog(INFO, "[avg_conv] no bias");
    }
     
    res = float4construct_md_array(res_data, f_out_nelems_channel * out_channel, 2, res_dims, lbound, FLOAT4OID);
    PG_RETURN_ARRAYTYPE_P(res);
}


void avgpool_core(float4* input_data, float4* avg_data, int out_nelems_channel, int width, int height, int num_channel, int kernel_size, int padding, int stride) 
{
    int matrix_head, matrix_head_c;
    int nelems_channel = width * height;
    int kernel_size_2 = kernel_size * kernel_size;
    int res_idx_local;
    float4 avg_sum = 0;

    omp_set_num_threads(OMP_THREADS);
    #pragma omp parallel for private(matrix_head, matrix_head_c, avg_sum, res_idx_local)  
    for (int c = 0; c < num_channel; c++) {
        matrix_head_c = nelems_channel * c;
        res_idx_local = c * out_nelems_channel;
        for (int rn = -padding; rn < height + padding - kernel_size + 1; rn+= stride) {
            for (int cn = -padding; cn < width +padding - kernel_size + 1; cn += stride) {
                matrix_head = matrix_head_c + rn * width + cn;
                avg_sum = 0;
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        if (ky + rn >= 0 && ky + rn < height && kx + cn >= 0 && kx + cn < width) {
                            avg_sum += input_data[matrix_head + ky * width + kx];
                        }  
                    }
                }
                avg_data[res_idx_local++] = avg_sum / kernel_size_2;
            }
        }   
    }
}