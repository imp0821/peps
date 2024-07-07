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
// #include "mkl.h"

#include "aqua_configs.h"
#include <omp.h>

ArrayType* float4construct_md_array(float *res_data,
                         int size,
                         int ndims,
                         int *dims,
                         int *lbs,
                         Oid elmtype);

ArrayType*
float4construct_md_array_ncpy(int size,
                         int ndims,
                         int *dims,
                         int *lbs,
                         Oid elmtype);


PG_FUNCTION_INFO_V1(im2col);
Datum 
im2col(PG_FUNCTION_ARGS)
{     
    // clock_t start, end, tmp;
    // double cpu_time_used;
    // start = clock(); 
    int width, height, kernel_size, stride, padding;
    int out_width, out_height, out_nelems;
    int res_idx;
    ArrayType *output_f;
    int output_f_dims[2];
    int lbound[2] = {1, 1};
    int matrix_head, matrix_head_c, k_offset;
    
    ArrayType *input_f = PG_GETARG_ARRAYTYPE_P(0);
    Oid elemtype = input_f->elemtype;
    int *dims = ARR_DIMS(input_f);
    int num_channel = dims[0];
    int nelems_channel = dims[1];
    float4 *data = (float4 *) ARR_DATA_PTR(input_f);

    //im2col
    kernel_size = PG_GETARG_INT32(1);
    padding = PG_GETARG_INT32(2);
    stride = PG_GETARG_INT32(3);
    height = width = sqrt(nelems_channel);
    out_width = (width + 2 * padding - kernel_size + 1)/ stride;
    out_height = (height + 2 * padding - kernel_size + 1)/ stride;
    out_nelems = out_width * out_height * num_channel * kernel_size *kernel_size;
    res_idx = 0;


    // tmp = clock();
    // cpu_time_used = ((double) (tmp - start)) / CLOCKS_PER_SEC;
    // elog(INFO, "[im2col] t1 took %f ms to execute.\n", cpu_time_used*1000);
    // elog(INFO, "[im2col] num_channel: %d, width: %d, outwidth: %d", num_channel, width, out_width);
    float4 *res_data = (float4 *)palloc(sizeof(float4) * out_nelems);
    for (int rn = -padding; rn < height + padding - kernel_size + 1; rn+= stride) {
        for (int cn = -padding; cn < width +padding - kernel_size + 1; cn += stride) {
            matrix_head = rn * width + cn;
            for (int c = 0; c < num_channel; c++) {
                matrix_head_c = matrix_head + nelems_channel * c;
                for (int ky = 0; ky < kernel_size; ky++) {
                    k_offset = ky * width;
                    for (int kx = 0; kx < kernel_size; kx++) {
                        if (ky + rn < 0 || ky + rn >= height || kx + cn < 0 || kx + cn >= width) {
                            res_data[res_idx++] = 0;
                        } else {
                            res_data[res_idx++] = data[matrix_head_c + k_offset + kx];
                        }
                    }
                }
            }
        }
    }
    // tmp = clock();
    // cpu_time_used = ((double) (tmp - start)) / CLOCKS_PER_SEC;
    // elog(INFO, "[im2col] t2 took %f ms to execute.\n", cpu_time_used*1000);

    output_f_dims[0] = out_width * out_height;
    output_f_dims[1] = num_channel * kernel_size * kernel_size;
    //output_f = construct_md_array(res_data, NULL, 2, output_f_dims, lbound, elemtype, sizeof(float4), true, TYPALIGN_INT);
    output_f = float4construct_md_array(res_data, out_nelems, 2, output_f_dims, lbound, elemtype);

    // end = clock();
    // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // elog(INFO, "[im2col] took %f ms to execute.\n", cpu_time_used*1000);
    PG_RETURN_ARRAYTYPE_P(output_f);
}


PG_FUNCTION_INFO_V1(relu);
Datum 
relu(PG_FUNCTION_ARGS)
{      
    int res_idx, size, i;
    // Datum *res_data;
    // float4 *res_data;
    ArrayType *feature_map;
    int feature_map_dims[2];
    int lbound[2] = {1, 1};
    
    ArrayType *input_f = PG_GETARG_ARRAYTYPE_P(0);
    Oid elemtype = input_f->elemtype;
    int *dims = ARR_DIMS(input_f);
    int num_channel = dims[0];
    int nelems_channel = dims[1];
    float4 *data = (float4 *) ARR_DATA_PTR(input_f);

    res_idx = 0;
    size = num_channel * nelems_channel;
    // res_data = (float4 *) palloc(sizeof(float4) * size);
    // res_data = (Datum *) palloc(sizeof(Datum) * num_channel * nelems_channel);

    for (i = 0; i< size; i++) {
        data[i] = fmaxf(0, data[i]); 
    }

    feature_map_dims[0] = num_channel;
    feature_map_dims[1] = nelems_channel;
    feature_map = float4construct_md_array(data, size, 2, feature_map_dims, lbound, elemtype);
    //feature_map = construct_md_array(res_data, NULL, 2, feature_map_dims, lbound, elemtype, sizeof(float4), true, TYPALIGN_INT);

    PG_RETURN_ARRAYTYPE_P(feature_map);
}

PG_FUNCTION_INFO_V1(kfm);
Datum 
kfm(PG_FUNCTION_ARGS)
{
    openblas_set_num_threads(GEMM_THREADS);
    // mkl_set_num_threads(GEMM_THREADS);
    ArrayType *kmatrix = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *fmatrix = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *bias_vec = PG_GETARG_ARRAYTYPE_P(2);

    float4 *kmatrix_data = (float4 *) ARR_DATA_PTR(kmatrix);
    int *kmatrix_dims = ARR_DIMS(kmatrix);  //27, 64
    int out_channel = kmatrix_dims[0]; 
    int klen = kmatrix_dims[1];

    float4 *fmatrix_data_float4 = (float4 *) ARR_DATA_PTR(fmatrix);
    int *fmatrix_dims = ARR_DIMS(fmatrix);  //1024, 27
    int flen = fmatrix_dims[0];
    int fklen = fmatrix_dims[1];

    int res_rows = out_channel;
    int res_cols = flen;
    int size = res_rows * res_cols;
    ArrayType *res;
    int lbound[2] = {1, 1};
    int res_dims[2] = {res_rows, res_cols};

    if (klen != fklen) {
        elog(ERROR, "The number of matrix columns must equal the number of vector elements");
    }

    // Datum *res_data = (Datum *) palloc(sizeof(Datum) * size);
    float4 *res_data_float4 = (float4 *) palloc(sizeof(float4) * size);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                res_rows, res_cols, klen,
                1.0, kmatrix_data, klen, fmatrix_data_float4, klen,
                0.0, res_data_float4, res_cols);
    
    if (ARR_DIMS(bias_vec)[0] == out_channel) {
        float4 *bias = (float4 *) ARR_DATA_PTR(bias_vec);
        int offset = 0;
        for (int c = 0; c < out_channel; c++) {
            for (int i = offset; i < offset + flen; i++) {
                res_data_float4[i] += bias[c];
                res_data_float4[i] = fmax(0, res_data_float4[i]);
            }
            offset += flen;
        }
    }

    res = float4construct_md_array(res_data_float4, size, 2, res_dims, lbound, FLOAT4OID);
    PG_RETURN_ARRAYTYPE_P(res);
}

PG_FUNCTION_INFO_V1(kfm_nt); // no need to trans
Datum 
kfm_nt(PG_FUNCTION_ARGS)
{
    openblas_set_num_threads(GEMM_THREADS);
    // mkl_set_num_threads(GEMM_THREADS);
    ArrayType *kmatrix = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *fmatrix = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *bias_vec = PG_GETARG_ARRAYTYPE_P(2);

    float4 *kmatrix_data = (float4 *) ARR_DATA_PTR(kmatrix);
    int *kmatrix_dims = ARR_DIMS(kmatrix);  //112, 64
    int out_channel = kmatrix_dims[0]; 
    int klen = kmatrix_dims[1];

    float4 *fmatrix_data = (float4 *) ARR_DATA_PTR(fmatrix);
    int *fmatrix_dims = ARR_DIMS(fmatrix);  //64, 1024
    
    int fklen = fmatrix_dims[0];
    int flen = fmatrix_dims[1];

    int res_rows = out_channel;
    int res_cols = flen;
    int size = res_rows * res_cols;
    // elog(INFO, "out_channel:%d, klen:%d, fklen:%d, flen:%d",out_channel, klen, fklen, flen);
    ArrayType *res;
    int lbound[2] = {1, 1};
    int res_dims[2] = {res_rows, res_cols};

    if (klen != fklen) {
        elog(ERROR, "The number of matrix columns must equal the number of vector elements");
    }

     
    float4 *res_data = (float4 *) palloc0(sizeof(float4) * out_channel * flen);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                res_rows, res_cols, klen,
                1.0, kmatrix_data, klen, fmatrix_data, flen,
                0.0, res_data, res_cols);
    
    // omp_set_num_threads(16);
    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < res_rows; i++) {
    //     for (int j = 0; j < res_cols; j++) {
    //         float sum = 0.0;
    //         for (int k = 0; k<klen; k++) {
    //             sum += kmatrix_data[i * klen + k] * fmatrix_data[k * flen + j];  
    //         }
    //         res_data[i * flen + j] = sum;  
    //     } 
    // }

    if (ARR_DIMS(bias_vec)[0] == out_channel) {
        float4 *bias = (float4 *) ARR_DATA_PTR(bias_vec);
        int offset = 0;
        bool acti = PG_GETARG_BOOL(3);
        if (acti) {
            for (int c = 0; c < out_channel; c++) {
                for (int i = offset; i < offset + flen; i++) {
                    res_data[i] += bias[c];
                    res_data[i] = fmax(0, res_data[i]);
                }
                offset += flen;
            }
        } else {
            for (int c = 0; c < out_channel; c++) {
                for (int i = offset; i < offset + flen; i++) {
                    res_data[i] += bias[c];
                }
                offset += flen;
            }
        }
    }  
    res = float4construct_md_array(res_data, size, 2, res_dims, lbound, FLOAT4OID);
    PG_RETURN_ARRAYTYPE_P(res);
}

PG_FUNCTION_INFO_V1(mvm);
Datum 
mvm(PG_FUNCTION_ARGS)
{   
    ArrayType *matrix = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *vector = PG_GETARG_ARRAYTYPE_P(1); // 1024*1
    ArrayType *bias_vec = PG_GETARG_ARRAYTYPE_P(2);

    float4 *matrix_data = (float4 *) ARR_DATA_PTR(matrix);
    float4 *vector_data = (float4 *) ARR_DATA_PTR(vector);
    int *matrix_dims = ARR_DIMS(matrix);
    // int vector_nelems = ArrayGetNItems(ARR_NDIM(vector), ARR_DIMS(vector));
    int *vector_dims = ARR_DIMS(vector);
    int vector_nelems = vector_dims[0]; 
    int matrix_rows = matrix_dims[0];
    int matrix_cols = matrix_dims[1];
    int res_nelems = matrix_rows;
    //float4 *res_data = (float4 *) palloc0(sizeof(float4) * res_nelems);
    Datum *res_data = (Datum *) palloc(sizeof(Datum) * res_nelems);
    ArrayType *res;

    if (matrix_cols != vector_nelems) {
        elog(ERROR, "The number of matrix columns must equal the number of vector elements");
    }

   
    float4 *bias = (float4 *) ARR_DATA_PTR(bias_vec);
    bool bias_flag = (ARR_DIMS(bias_vec)[0] == matrix_rows);

    // omp_set_num_threads(4);
    # pragma omp parallel for
    for (int i = 0; i < matrix_rows; i++) {
        float4 sum = 0;
        for (int j = 0; j < matrix_cols; j++) {
            sum += matrix_data[i * matrix_cols + j] * vector_data[j];
        }
        if (bias_flag){
            sum += bias[i];
        }
        // res_data[i] = sum;
        res_data[i] = Float4GetDatum(sum);

    }
    int			dims[1];
	int			lbs[1];
    dims[0] = res_nelems;
    lbs[0] = 1;
    // res = float4construct_md_array(res_data, res_nelems, 1, dims, lbs, FLOAT4OID);
    res = construct_array(res_data, res_nelems, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT);
    PG_RETURN_ARRAYTYPE_P(res);
}


PG_FUNCTION_INFO_V1(madd);
Datum 
madd(PG_FUNCTION_ARGS)
{
    clock_t start, end, tmp;
    double cpu_time_used;
    start = clock();
    ArrayType *res;
    ArrayType *matrix1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *matrix2 = PG_GETARG_ARRAYTYPE_P(1);
    Oid typef = matrix1->elemtype;

    float4 *matrix1_data = (float4 *) ARR_DATA_PTR(matrix1);
    float4 *matrix2_data = (float4 *) ARR_DATA_PTR(matrix2);
    
    int *matrix_dims = ARR_DIMS(matrix1);
    int res_rows = matrix_dims[0];
    int res_cols = matrix_dims[1];
    int size = res_rows * res_cols;
    // float4 *res_data = (float4 *) palloc(sizeof(float4) * size);
     
    
    int res_idx = 0;
    int lbound[2] = {1, 1};
    int res_dims[2] = {res_rows, res_cols};
    
    // omp_set_num_threads(4);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        matrix1_data[i] += matrix2_data[i];
        // res_data[i] = matrix1_data[i] + matrix2_data[i];
        //res_data[i] = Float4GetDatum(matrix1_data[i] + matrix2_data[i]);
    }

    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    res = float4construct_md_array(matrix1_data, size, 2, res_dims, lbound, FLOAT4OID);
    // elog(INFO, "[madd] took %f ms to execute.\n", cpu_time_used*1000);
    PG_RETURN_ARRAYTYPE_P(res);
}

ArrayType*
float4construct_md_array(float *res_data,
                         int size,
                         int ndims,
                         int *dims,
                         int *lbs,
                         Oid elmtype)
{
    ArrayType *res;
    MemoryContext oldcontext;

    oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);
    
    int nbytes = sizeof(float4) * size;
    res = (ArrayType *)palloc0(nbytes + ARR_OVERHEAD_NONULLS(ndims));
    SET_VARSIZE(res, nbytes + ARR_OVERHEAD_NONULLS(ndims));
    res->ndim = ndims;
    res->dataoffset = 0; //assuming no null value
    res->elemtype = FLOAT4OID;

    memcpy(ARR_DIMS(res), dims, ndims * sizeof(int));
	memcpy(ARR_LBOUND(res), lbs, ndims * sizeof(int));
	memcpy(ARR_DATA_PTR(res), res_data, nbytes);

    MemoryContextSwitchTo(oldcontext);

    return res;
}

ArrayType*
float4construct_md_array_ncpy(int size,
                         int ndims,
                         int *dims,
                         int *lbs,
                         Oid elmtype)
{
    ArrayType *res;
    MemoryContext oldcontext;

    oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);
    
    int nbytes = sizeof(float4) * size;
    res = (ArrayType *)palloc0(nbytes + ARR_OVERHEAD_NONULLS(ndims));
    SET_VARSIZE(res, nbytes + ARR_OVERHEAD_NONULLS(ndims));
    res->ndim = ndims;
    res->dataoffset = 0; //assuming no null value
    res->elemtype = FLOAT4OID;

    memcpy(ARR_DIMS(res), dims, ndims * sizeof(int));
	memcpy(ARR_LBOUND(res), lbs, ndims * sizeof(int));
	//memcpy(ARR_DATA_PTR(res), res_data, nbytes);

    MemoryContextSwitchTo(oldcontext);

    return res;
}