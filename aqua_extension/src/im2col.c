#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define ROWS 14
#define COLS 1024

// 声明你的 im2col 函数，你需要自己实现这个函数
// 这里只是一个假设的函数原型，你需要根据实际情况修改
float* im2col_test(float *data, int kernel_size, int padding, int stride);

int main(void) {
    int i, j;
    clock_t start, end;
    double cpu_time_used;
    int size = ROWS * COLS;
    // 创建数组
    float *data = (float*)malloc(sizeof(float) * size);

    // 随机生成数据
    srand(time(0));  
    for(i = 0; i < ROWS; i++)
        for(j = 0; j < COLS; j++)
            data[i*COLS + j] = (float)rand() / RAND_MAX;

    // 调用 im2col 函数
    start = clock();
    float *new_data = im2col_test(data, 3, 1, 1);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("im2col took %f ms to execute \n", cpu_time_used * 1000);
     
    free(data);

    return 0;
}

float* im2col_test(float *data, int kernel_size, int padding, int stride) {
    int width, height;
    int out_width, out_height, out_nelems;
    int num_channel = ROWS;
    int nelems_channel = COLS;
    int res_idx = 0;
    
    width = height = sqrt(COLS);
    out_width = (width + 2 * padding - kernel_size + 1)/ stride;
    out_height = (height + 2 * padding - kernel_size + 1)/ stride;
    out_nelems = out_width * out_height * num_channel * kernel_size *kernel_size;
    
    // float *res_data = (float *)malloc(sizeof(float) * out_nelems);
    float *res_data = (float *)calloc(out_nelems, sizeof(float));
    int matrix_head, matrix_head_c, k_offset;
    
    int channel_offsets[num_channel];
    for (int i=0; i<num_channel; i++)
        channel_offsets[i] = nelems_channel * i;

    for (int rn = -padding; rn < height + padding - kernel_size + 1; rn+= stride) {
        for (int cn = -padding; cn < width +padding - kernel_size + 1; cn += stride) {
            matrix_head = rn * width + cn;
            for (int c = 0; c < num_channel; c++) {
                matrix_head_c = matrix_head + channel_offsets[c];
                for (int ky = 0; ky < kernel_size; ky++) {
                    k_offset = ky * width;
                    for (int kx = 0; kx < kernel_size; kx++) {
                        if (ky + rn < 0 || ky + rn >= height || kx + cn < 0 || kx + cn >= width) {
                            res_idx += 1;
                        } else {
                            res_data[res_idx++] = data[matrix_head_c + k_offset + kx];
                        }
                    }
                }
            }
        }
    }

    int float_size = sizeof(float);


    // for (int rn = 0; rn < out_width; rn++) {
    //     for (int cn = 0; cn < out_height; cn++) {
    //         for (int c = 0; c < num_channel; c++){
    //             //first kernel line
    //             if (rn < padding) {
    //                 res_idx += kernel_size;
    //             } else if (cn < padding) {
    //                 res_idx += 1;
    //                 memcpy(res_data+res_idx, data, 2);
    //                 res_idx += 2;
    //             } else if (cn >=num_channel + padding) {

    //             } else {
    //                 memcpy(res_data+res_idx, data, 3);
    //                 res_idx += 3;
    //             }
    //         }
    //     }
    // }
    return res_data;
}