// PG_FUNCTION_INFO_V1(im2col_init);
// Datum 
// im2col_init(PG_FUNCTION_ARGS)
// {      
//     int width, height, kernel_size, stride, padding;
//     int out_width, out_height, out_nelems;
//     int res_idx;
//     float4 *res_data;
//     ArrayType *feature_map;
//     int feature_map_dims[2];
//     int lbound[2] = {1, 1};
//     int matrix_head;
//     int matrix_head_c;
    
//     ArrayType *image = PG_GETARG_ARRAYTYPE_P(0);
//     int *dims = ARR_DIMS(image);
//     int num_channel = dims[0];
//     int nelems_channel = dims[1];
//     float4 *data = (float4 *) ARR_DATA_PTR(image);

//     //im2col
//     height = width = sqrt(nelems_channel);
//     kernel_size = 3;
//     stride = 1;
//     padding =1;
//     out_width = (width + 2 * padding - kernel_size + 1)/ stride;
//     out_height = (height + 2 * padding - kernel_size + 1)/ stride;
//     out_nelems = out_width * out_height * num_channel * kernel_size *kernel_size;
//     res_idx = 0;
    
//     res_data = (float4 *)palloc(sizeof(float4) * out_nelems);

//     for (int rn = -padding; rn < height + padding - kernel_size + 1; rn+= stride) {
//         for (int cn = -padding; cn < width +padding - kernel_size + 1; cn += stride) {
//             matrix_head = rn * width + cn;
//             for (int c = 0; c < num_channel; c++) {
//                 matrix_head_c = matrix_head + nelems_channel * c;
//                 for (int ky = 0; ky < kernel_size; ky++) {
//                     for (int kx = 0; kx < kernel_size; kx++) {
//                         if (ky + rn < 0 || ky + rn >= height || kx + cn < 0 || kx + cn >= width) {
//                             res_data[res_idx++] = 0;
//                         } else {
//                             res_data[res_idx++] = data[matrix_head_c + ky * width + kx];
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     feature_map_dims[0] = out_width * out_height;
//     feature_map_dims[1] = num_channel * kernel_size * kernel_size;
//     feature_map = float4construct_md_array(res_data, out_nelems, 2, feature_map_dims, lbound, FLOAT4OID);
//     PG_RETURN_ARRAYTYPE_P(feature_map);
// }
