#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"

#include "funcapi.h"
#include <jpeglib.h>

#include <omp.h>

PG_FUNCTION_INFO_V1(test_flops);
Datum 
test_flops(PG_FUNCTION_ARGS)
{
    float4 result = 1.0;  
    int i;  

    omp_set_num_threads(8);
    #pragma omp parallel for
    for (i = 0; i < 10000000; i++) {  
        result = result * 1.01 + 0.001;  
    }  
  
    PG_RETURN_FLOAT4(result);  
}

PG_FUNCTION_INFO_V1(virtual_in);
Datum 
virtual_in(PG_FUNCTION_ARGS)
{
    char *file_path = PG_GETARG_CSTRING(0);
    // bytea *raw_image;
    int image_size, num_elements_channel, num_channels;
    
    Datum *elements;
    uint8 *image_data;
    ArrayType *image_array;
    int dims[2];
    int lbound[2] = {1, 1};

    int array_len = sizeof(float4);
    bool array_val = true;
    char array_align = TYPALIGN_INT;
    Oid array_type = FLOAT4OID;

    FILE *jpeg_file;
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPROW row_pointer;
    int row_stride, current_row;

    jpeg_file = fopen(file_path, "rb");
    if (jpeg_file == NULL) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("Failed to open JPEG file")));
    }

    // Initialize the JPEG decompression object
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, jpeg_file);
    jpeg_read_header(&cinfo, TRUE);

    // Start the decompression process
    jpeg_start_decompress(&cinfo);

    // Allocate memory for image data
    image_size = cinfo.output_width * cinfo.output_height * cinfo.output_components;
    image_data = (uint8 *) palloc(image_size * sizeof(uint8));

    // Read scanlines and store image data
    row_stride = cinfo.output_width * cinfo.output_components;
    current_row = 0;

    while (cinfo.output_scanline < cinfo.output_height) {
        row_pointer = &image_data[current_row * row_stride];
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
        current_row++;
    }

    // Finish decompression
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(jpeg_file);

    //start convert image data to channel-indexed arrays
    num_channels = cinfo.output_components;
    num_elements_channel = image_size / sizeof(uint8) / num_channels; // number of elements for each channel
    
    if (image_size % sizeof(uint8) != 0) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("Input data size is not a multiple of sizeof(uint8)")));
    }
    // elog(INFO, "[INFO]number of elements and channels are: %d, %d", num_elements_channel, num_channels);

    // start assignment
    float4 mean[3] = {0.4914, 0.4822, 0.4465};
    float4 std[3] = {0.2471, 0.2435, 0.2616};
    elements = (Datum *) palloc(num_elements_channel * num_channels * sizeof(Datum));
    for (int i = 0; i < num_channels; i++) {
        Datum *elements_channel_head = &elements[i * num_elements_channel];
        for (int j = 0; j < num_elements_channel; j++) {
            elements_channel_head[j] = Float4GetDatum((image_data[j * num_channels + i]*1.0/255-mean[i])/std[i]);
        }
    }

    dims[0] = num_channels;
    dims[1] = num_elements_channel;
    
    image_array = construct_md_array(elements, NULL, 2, dims, lbound, array_type, array_len, array_val, array_align);
    
    HeapTuple ret_tuple;
    Datum res;    
    Datum values[2];  
    bool isnull[2] = {false, false};  
    TupleDesc tupdesc; 

    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)  
        ereport(ERROR,  
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),  
                 errmsg("function returning record called in context that cannot accept type record")));  

    values[0] = PointerGetDatum(cstring_to_text_with_len(file_path, strlen(file_path)));  // 示例中简化处理，实际应存储二进制数据  
    values[1] = PointerGetDatum(image_array);  
    ret_tuple = heap_form_tuple(tupdesc, values, isnull);
    res = HeapTupleGetDatum(ret_tuple); 
    
    //PG_RETURN_ARRAYTYPE_P(image_array);
    PG_RETURN_DATUM(res);  
}

// PG_FUNCTION_INFO_V1(virtual_arr);
// Datum virtual_arr(PG_FUNCTION_ARGS) {
//   ArrayType *image_array = PG_GETARG_ARRAYTYPE_P(0);
//   PG_RETURN_ARRAYTYPE_P(image_array); 
// }