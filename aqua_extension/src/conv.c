#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "catalog/pg_type.h"
#include "optimizer/planner.h"
#include "nodes/supportnodes.h"

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

extern ArrayType* float4construct_md_array_ncpy(int size,
                         int ndims,
                         int *dims,
                         int *lbs,
                         Oid elmtype);

void im2col_core(float4* input_f, int *input_f_shape, float4* f_matrix, int kernel_size, int padding, int stride);
void im2col_core_W(float4 *input_f, int *input_f_shape, float4* f_matrix, int kernel_W, int padding_W, int stride);
void im2col_core_H(float4 *input_f, int *input_f_shape, float4* f_matrix, int kernel_H, int padding_H, int stride);

PG_FUNCTION_INFO_V1(kfm_im2col_ns);
Datum 
kfm_im2col_ns(PG_FUNCTION_ARGS)
{     
    openblas_set_num_threads(GEMM_THREADS);
    // mkl_set_num_threads(GEMM_THREADS);
    int width, height, kernel_H, kernel_W, stride, padding_H, padding_W;
    int out_width, out_height, out_nelems;
    int f_dims[2];
    
    ArrayType *array_input_f = PG_GETARG_ARRAYTYPE_P(1);
    Oid elemtype = array_input_f->elemtype;
    int *dims = ARR_DIMS(array_input_f);
    int num_channel = dims[0];
    int nelems_channel = dims[1];
    float4 *input_f = (float4 *) ARR_DATA_PTR(array_input_f);

    //im2col
    kernel_H = PG_GETARG_INT32(3);
    kernel_W = PG_GETARG_INT32(4);
    padding_H = PG_GETARG_INT32(5);
    padding_W = PG_GETARG_INT32(6);
    stride = PG_GETARG_INT32(7);
    height = width = sqrt(nelems_channel);
    out_width = (width + 2 * padding_W - kernel_W)/ stride + 1;
    out_height = (height + 2 * padding_H - kernel_H)/ stride + 1;
    out_nelems = out_width * out_height * num_channel * kernel_W * kernel_H;
    
    float4 *f_matrix = (float4 *)palloc0(sizeof(float4) * out_nelems);
    if (kernel_H == 1)
        im2col_core_W(input_f, dims, f_matrix, kernel_W, padding_W, stride);
    else
        im2col_core_H(input_f, dims, f_matrix, kernel_H, padding_H, stride);
    

    f_dims[0] = out_width * out_height;
    f_dims[1] = num_channel * kernel_H * kernel_W;

    // kfm
    ArrayType *res;
    int lbound[2] = {1, 1};
    int k_channel, klen, res_rows, res_cols, res_size;

    ArrayType *array_k = PG_GETARG_ARRAYTYPE_P(0);
    float4 *k_matrix = (float4 *) ARR_DATA_PTR(array_k);
    int *kmatrix_dims = ARR_DIMS(array_k);
    k_channel = kmatrix_dims[0];
    klen = kmatrix_dims[1];

    res_rows = k_channel;
    res_cols = f_dims[0];
    res_size = res_rows * res_cols;
    int res_dims[2] = {res_rows, res_cols};
    
    if (klen != f_dims[1]) {
        elog(ERROR, "The number of matrix columns must equal the number of vector elements");
    }

    res = float4construct_md_array_ncpy(res_size, 2, res_dims, lbound, FLOAT4OID);
    float4 *res_data = (float4 *)ARR_DATA_PTR(res);
    // float4 *res_data = (float4 *) palloc(sizeof(float4) * res_size);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                res_rows, res_cols, klen,
                1.0, k_matrix, klen, f_matrix, klen,
                0.0, res_data, res_cols); 
    
    ArrayType *bias_vec = PG_GETARG_ARRAYTYPE_P(2);
    if (ARR_DIMS(bias_vec)[0] == k_channel) {
        float4 *bias = (float4 *) ARR_DATA_PTR(bias_vec);
        int offset = 0;
        for (int c = 0; c < k_channel; c++) {
            for (int i = offset; i < offset + res_cols; i++) {
                res_data[i] += bias[c];
                res_data[i] = fmax(0, res_data[i]);
            }
            offset += res_cols;
        }
    } else {
        //elog(INFO, "[kfm_im2col_ns] no bias");
    }
    
    //res = float4construct_md_array(res_data, res_size, 2, res_dims, lbound, FLOAT4OID);

    PG_RETURN_ARRAYTYPE_P(res);
}

PG_FUNCTION_INFO_V1(kfm_im2col);
Datum 
kfm_im2col(PG_FUNCTION_ARGS)
{     
    openblas_set_num_threads(GEMM_THREADS);
    //mkl_set_num_threads(GEMM_THREADS);
    // clock_t start, end, tmp;
    // double cpu_time_used;
    // start = clock(); 
    int width, height, kernel_size, stride, padding;
    int out_width, out_height, out_nelems;
    int f_dims[2];
    
    ArrayType *array_input_f = PG_GETARG_ARRAYTYPE_P(1);
    Oid elemtype = array_input_f->elemtype;
    int *dims = ARR_DIMS(array_input_f);
    int num_channel = dims[0];
    int nelems_channel = dims[1];
    float4 *input_f = (float4 *) ARR_DATA_PTR(array_input_f);

    //im2col
    kernel_size = PG_GETARG_INT32(3);
    padding = PG_GETARG_INT32(4);
    stride = PG_GETARG_INT32(5);
    height = width = sqrt(nelems_channel);
    out_width = (width + 2 * padding - kernel_size)/ stride + 1;
    out_height = (height + 2 * padding - kernel_size)/ stride + 1;
    out_nelems = out_width * out_height * num_channel * kernel_size *kernel_size;
    
    float4 *f_matrix = (float4 *)palloc0(sizeof(float4) * out_nelems);
    im2col_core(input_f, dims, f_matrix, kernel_size, padding, stride);
    

    f_dims[0] = out_width * out_height;
    f_dims[1] = num_channel * kernel_size * kernel_size;

    // kfm
    ArrayType *res;
    int lbound[2] = {1, 1};
    int k_channel, klen, res_rows, res_cols, res_size;

    ArrayType *array_k = PG_GETARG_ARRAYTYPE_P(0);
    float4 *k_matrix = (float4 *) ARR_DATA_PTR(array_k);
    int *kmatrix_dims = ARR_DIMS(array_k);
    k_channel = kmatrix_dims[0];
    klen = kmatrix_dims[1];

    res_rows = k_channel;
    res_cols = f_dims[0];
    res_size = res_rows * res_cols;
    int res_dims[2] = {res_rows, res_cols};
    
    if (klen != f_dims[1]) {
        elog(ERROR, "The number of matrix columns must equal the number of vector elements");
    }
 
    // res = float4construct_md_array_ncpy(res_size, 2, res_dims, lbound, FLOAT4OID);
    // float4 *res_data = (float4 *)ARR_DATA_PTR(res);
    float4 *res_data = (float4 *) palloc(sizeof(float4) * res_size);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                res_rows, res_cols, klen,
                1.0, k_matrix, klen, f_matrix, klen,
                0.0, res_data, res_cols);

    ArrayType *bias_vec = PG_GETARG_ARRAYTYPE_P(2);
    if (ARR_DIMS(bias_vec)[0] == k_channel) {
        float4 *bias = (float4 *) ARR_DATA_PTR(bias_vec);
        int offset = 0;
        for (int c = 0; c < k_channel; c++) {
            for (int i = offset; i < offset + res_cols; i++) {
                res_data[i] += bias[c];
                res_data[i] = fmax(0, res_data[i]);
            }
            offset += res_cols;
        }
    } else {
        // elog(INFO, "[kfm_im2col] no bias");
    }
    
    res = float4construct_md_array(res_data, res_size, 2, res_dims, lbound, FLOAT4OID);
    PG_RETURN_ARRAYTYPE_P(res);
}

PG_FUNCTION_INFO_V1(group_kfm_im2col);
Datum 
group_kfm_im2col(PG_FUNCTION_ARGS)
{     
    int width, height, kernel_size, stride, padding;
    int out_width, out_height, out_nelems;
    int f_dims[2];
    
    ArrayType *array_input_f = PG_GETARG_ARRAYTYPE_P(1);
    Oid elemtype = array_input_f->elemtype;
    int *dims = ARR_DIMS(array_input_f);
    int total_num_channel = dims[0];  //128
    int nelems_channel = dims[1]; //256
    float4 *input_f = (float4 *) ARR_DATA_PTR(array_input_f);

    int split_num = PG_GETARG_INT32(6);
    if (total_num_channel % split_num != 0) {
        ereport(ERROR, (errmsg("The number of rows in the input array is not evenly divisible by split_num")));
    }
    int sub_num_channel = total_num_channel / split_num;
    dims[0] = sub_num_channel;

    //im2col
    kernel_size = PG_GETARG_INT32(3);
    padding = PG_GETARG_INT32(4);
    stride = PG_GETARG_INT32(5);
    height = width = sqrt(nelems_channel);
    out_width = (width + 2 * padding - kernel_size + 1)/ stride;
    out_height = (height + 2 * padding - kernel_size + 1)/ stride;
    out_nelems = out_width * out_height * sub_num_channel * kernel_size *kernel_size;

    f_dims[0] = out_width * out_height;
    f_dims[1] = sub_num_channel * kernel_size * kernel_size;

    //kernel
    ArrayType *res;
    int lbound[2] = {1, 1};
    int k_channel, klen, res_rows, res_cols, res_size;

    ArrayType *array_k = PG_GETARG_ARRAYTYPE_P(0);
    float4 *k_matrix = (float4 *) ARR_DATA_PTR(array_k);
    int *kmatrix_dims = ARR_DIMS(array_k);  // 32, 4, 36
    Assert(kmatrix_dims[0] == split_num);
    k_channel = kmatrix_dims[1];
    klen = kmatrix_dims[2];
    res_rows = k_channel;
    res_cols = f_dims[0];
    res_size = res_rows * res_cols;
    int res_dims[2] = {res_rows * split_num, res_cols};

    if (klen != f_dims[1]) {
        elog(ERROR, "The number of matrix columns must equal the number of vector elements");
    }
    float4 *res_data = (float4 *) palloc(sizeof(float4) * res_size * split_num);
    float4 *f_matrix[split_num];
    for (int gn =0; gn < split_num; gn++) {
        f_matrix[gn] = (float4 *)palloc0(sizeof(float4) * out_nelems);
    }

    for (int gn = 0; gn < split_num; gn++) {
        im2col_core(input_f + gn * nelems_channel * sub_num_channel, dims, f_matrix[gn], kernel_size, padding, stride);
    }

    // omp_set_num_threads(OMP_THREADS);
    #pragma omp parallel for
    for (int gn = 0; gn < split_num; gn++) {

        //float4 *f_matrix = (float4 *)palloc0(sizeof(float4) * out_nelems);
        // im2col_core(input_f + gn * nelems_channel * sub_num_channel, dims, f_matrix[gn], kernel_size, padding, stride);

        // kfm
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    res_rows, res_cols, klen,
                    1.0, k_matrix + gn * k_channel * klen, klen, f_matrix[gn], klen,
                    0.0, res_data + gn * res_size, res_cols);
    }

    ArrayType *bias_vec = PG_GETARG_ARRAYTYPE_P(2);
    if (ARR_DIMS(bias_vec)[0] == split_num * k_channel) {
        float4 *bias = (float4 *) ARR_DATA_PTR(bias_vec);
        int offset = 0;
        for (int c = 0; c < split_num * k_channel; c++) {
            for (int i = offset; i < offset + res_cols; i++) {
                res_data[i] += bias[c];
                res_data[i] = fmax(0, res_data[i]);
            }
            offset += res_cols;
        }
    } 

    res = float4construct_md_array(res_data, res_size * split_num, 2, res_dims, lbound, FLOAT4OID); 
   
    PG_RETURN_ARRAYTYPE_P(res);
}


void im2col_core(float4 *input_f, int *input_f_shape, float4* f_matrix, int kernel_size, int padding, int stride) {
    int matrix_head, matrix_head_c, res_idx, res_idx_local, width, height;
    int num_channel = input_f_shape[0];
    int nelems_channel = input_f_shape[1];
    width = height = sqrt(nelems_channel);

    int out_height = (height + 2 * padding - kernel_size) / stride + 1;  
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;  

    res_idx = 0;
    int ky_offset[kernel_size];
    for (int i=0; i<kernel_size; i++)
        ky_offset[i] = width*i;
    
    
    if (padding != 0) {
        omp_set_num_threads(8);
        #pragma omp parallel for collapse(2) private(matrix_head, matrix_head_c, res_idx_local)  
        for (int rn = -padding; rn < height + padding - kernel_size + 1; rn+= stride) {
            for (int cn = -padding; cn < width +padding - kernel_size + 1; cn += stride) {
                res_idx_local = ((rn + padding) / stride * out_width + (cn + padding) / stride) * num_channel * kernel_size * kernel_size; 
                matrix_head = rn * width + cn;
                for (int c = 0; c < num_channel; c++) {
                    matrix_head_c = matrix_head + nelems_channel * c;
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            if (ky + rn < 0 || ky + rn >= height || kx + cn < 0 || kx + cn >= width) {
                                res_idx_local ++;
                            } else {
                                f_matrix[res_idx_local++] = input_f[matrix_head_c + ky_offset[ky] + kx];
                            }
                        }
                    }
                }
            }
        }
    } else {
        omp_set_num_threads(8);
        #pragma omp parallel for collapse(2) private(matrix_head, matrix_head_c, res_idx_local)  
        for (int rn = 0; rn < height - kernel_size + 1; rn+= stride) {
            for (int cn = 0; cn < width - kernel_size + 1; cn += stride) {
                res_idx_local = ((rn + padding) / stride * out_width + (cn + padding) / stride) * num_channel * kernel_size * kernel_size; 
                matrix_head = rn * width + cn;
                for (int c = 0; c < num_channel; c++) {
                    matrix_head_c = matrix_head + nelems_channel * c;
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            f_matrix[res_idx_local++] = input_f[matrix_head_c + ky_offset[ky] + kx];
                        }
                    }
                }
            }
        }
    }

}

void im2col_core_W(float4 *input_f, int *input_f_shape, float4* f_matrix, int kernel_W, int padding_W, int stride) {
    int matrix_head, matrix_head_c, res_idx, res_idx_local, width, height;
    int num_channel = input_f_shape[0];
    int nelems_channel = input_f_shape[1];
    width = height = sqrt(nelems_channel);

    int out_width = (width + 2 * padding_W - kernel_W) / stride + 1; 

    res_idx = 0;

    omp_set_num_threads(OMP_THREADS);
    #pragma omp parallel for collapse(2) private(matrix_head, matrix_head_c, res_idx_local)  
    for (int rn = 0; rn < height; rn+= stride) {
        for (int cn = -padding_W; cn < width + padding_W - kernel_W + 1; cn += stride) {
            res_idx_local = (rn / stride * out_width + (cn + padding_W) / stride) * num_channel * kernel_W; 
            matrix_head = rn * width + cn;
            for (int c = 0; c < num_channel; c++) {
                matrix_head_c = matrix_head + nelems_channel * c;
                for (int kx = 0; kx < kernel_W; kx++) {
                    if (kx + cn < 0 || kx + cn >= width) {
                        //res_idx ++;
                        res_idx_local ++;
                    } else {
                        f_matrix[res_idx_local++] = input_f[matrix_head_c + kx];
                    }
                }
            }
        }
    }

}

void im2col_core_H(float4 *input_f, int *input_f_shape, float4* f_matrix, int kernel_H, int padding_H, int stride) {
    int matrix_head, matrix_head_c, res_idx, res_idx_local, width, height;
    int num_channel = input_f_shape[0];
    int nelems_channel = input_f_shape[1];
    width = height = sqrt(nelems_channel);

    int out_height = (height + 2 * padding_H - kernel_H) / stride + 1; 

    int ky_offset[kernel_H];
    for (int i=0; i<kernel_H; i++)
        ky_offset[i] = width*i;
    res_idx = 0;

    omp_set_num_threads(OMP_THREADS);
    #pragma omp parallel for collapse(2) private(matrix_head, matrix_head_c, res_idx_local)  
    for (int rn = -padding_H; rn < height + padding_H -kernel_H + 1; rn+= stride) {
        for (int cn = 0; cn < width; cn += stride) {
            res_idx_local = ((rn + padding_H) / stride * out_height + cn / stride) * num_channel * kernel_H; 
            matrix_head = rn * width + cn;
            for (int c = 0; c < num_channel; c++) {
                matrix_head_c = matrix_head + nelems_channel * c;
                for (int ky = 0; ky < kernel_H; ky++) {
                    if (ky + rn < 0 || ky + rn >= height) {
                        // res_idx ++;
                        res_idx_local ++;
                    } else {
                        // f_matrix[res_idx++] = input_f[matrix_head_c + ky_offset[ky]];
                        f_matrix[res_idx_local++] = input_f[matrix_head_c + ky_offset[ky]];
                    }
                }
            }
        }
    }

}

 