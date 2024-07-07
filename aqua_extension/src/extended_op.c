#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "funcapi.h"

#include <math.h>
#include <time.h>
#include <cblas.h>

#include "libpq/pqformat.h"

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

Datum makeAgg2dResultArr(ArrayBuildStateArr *astate, MemoryContext rcontext, bool release);
static inline void
initReadOnlyStringInfo(StringInfo str, char *data, int len);

PG_FUNCTION_INFO_V1(concat_array);
Datum 
concat_array(PG_FUNCTION_ARGS)
{
	int num_inputs = PG_GETARG_INT32(0);
	float4 *inputs[num_inputs];
	int channels[num_inputs];
	int nelems_channel[num_inputs];
	ArrayType *matrix, *res;
	int *matrix_dims;
	int total_channel = 0, offset = 0, res_size;
	for (int i = 0; i < num_inputs; i++) {
		matrix = PG_GETARG_ARRAYTYPE_P(1+i);
		inputs[i] = (float4 *) ARR_DATA_PTR(matrix);
		matrix_dims = ARR_DIMS(matrix);
		channels[i] = matrix_dims[0];
		nelems_channel[i] = matrix_dims[1];
		total_channel += channels[i];
		// elog(INFO, "[concat_array] input %d: shape: %d,%d", i, channels[i], nelems_channel[i]);
	}
	res_size = total_channel * nelems_channel[0];
	int res_dims[2] = {total_channel, nelems_channel[0]};
	int lbound[2] = {1, 1};
	
	res = float4construct_md_array_ncpy(res_size, 2, res_dims, lbound, FLOAT4OID);
	float4* res_data = (float4 *)ARR_DATA_PTR(res);
	// float* res_data = (float4 *) palloc(sizeof(float4) * res_size);

	for (int i = 0; i < num_inputs; i++) {
		memcpy(res_data + offset, inputs[i], sizeof(float4)*channels[i]*nelems_channel[i]);
		offset += channels[i]*nelems_channel[i];
	}
	 
	//res = float4construct_md_array(res_data, res_size, 2, res_dims, lbound, FLOAT4OID);
	PG_RETURN_ARRAYTYPE_P(res);
}


PG_FUNCTION_INFO_V1(split_array);
Datum 
split_array(PG_FUNCTION_ARGS)
{
    ArrayType *input;
    int split_num;

    int *dims, nitems, ndims;
    Datum *resultElements;
    ArrayType *resultArray;
    int i, j, sub_size, nrows, ncols, rows_per_subarray;
    TupleDesc tupdesc;

   
    // 准备返回值
    FuncCallContext *funcctx;
    MemoryContext oldcontext;
    if (SRF_IS_FIRSTCALL()) 
    {
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
                ereport(ERROR, (errmsg("return type must be a row type")));
        
        input = PG_GETARG_ARRAYTYPE_P(0);
        split_num = PG_GETARG_INT32(1);
        float4 *data = (float4 *) ARR_DATA_PTR(input);
        // 存储所有子数组
        funcctx->user_fctx = palloc(sizeof(Datum) * split_num);
        resultElements = (Datum *)funcctx->user_fctx;

        ndims = ARR_NDIM(input); //assuming it is 2-dimensional
        dims = ARR_DIMS(input);
        nrows = dims[0];
        ncols = dims[1];
        rows_per_subarray = nrows / split_num;
         // check splitting validation
        if (nrows % split_num != 0) {
            ereport(ERROR, (errmsg("The number of rows in the input array is not evenly divisible by split_num")));
        }
        sub_size = rows_per_subarray * ncols;

        for (i = 0; i < split_num; i++) {
            Datum *subarray_data = palloc(sub_size * sizeof(Datum));
            float4 *data_head = data + i * sub_size;
            HeapTuple tuple;
            Datum values[2];
            Datum result;
            bool nulls[2] = {false, false};

            for (j = 0; j < sub_size; j++) {
                subarray_data[j] = Float4GetDatum(data_head[j]);
            }
            resultArray = construct_md_array(subarray_data, NULL, 2, (int[]){rows_per_subarray, ncols}, (int[]){1, 1}, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT);
            values[0] = Int32GetDatum(i+1);
            values[1] = PointerGetDatum(resultArray);
            tuple = heap_form_tuple(tupdesc, values, nulls);
            result = HeapTupleGetDatum(tuple);
            
            resultElements[i] = result;
        }
        funcctx->max_calls = split_num;
        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    resultElements = (Datum *)funcctx->user_fctx;

    if (funcctx->call_cntr < funcctx->max_calls) {
        // return next array 
        Datum res = resultElements[funcctx->call_cntr];
        SRF_RETURN_NEXT(funcctx, res);
    } else {
        SRF_RETURN_DONE(funcctx);
    }
}

PG_FUNCTION_INFO_V1(split_vertical_array);
Datum 
split_vertical_array(PG_FUNCTION_ARGS)
{
    ArrayType *input;
    int split_num;

	input = PG_GETARG_ARRAYTYPE_P(0);
	split_num = PG_GETARG_INT32(1);
	float4 *data = (float4 *) ARR_DATA_PTR(input);
	Datum res_elements[4];  
	bool nulls[4] = {true, true, true, true};
	TupleDesc tuple_desc;  
    HeapTuple ret_tuple;  
    Datum res;  

	for (int i = 0; i < split_num; i++) {  
		res_elements[i] = PointerGetDatum(input);  
		nulls[i] = false;
    } 

	// 获取返回类型的元组描述符  
    if (get_call_result_type(fcinfo, NULL, &tuple_desc) != TYPEFUNC_COMPOSITE) {  
        ereport(ERROR,  
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),  
                 errmsg("function returning record called in context that cannot accept type record")));  
    }
	ret_tuple = heap_form_tuple(tuple_desc, res_elements, nulls);  
	res = HeapTupleGetDatum(ret_tuple); 

	PG_RETURN_DATUM(res); 
}


PG_FUNCTION_INFO_V1(array_2d_agg_transfn);
Datum
array_2d_agg_transfn(PG_FUNCTION_ARGS)
{
    Oid			arg1_typeid = get_fn_expr_argtype(fcinfo->flinfo, 1);
    MemoryContext aggcontext;
    ArrayBuildStateArr *state;

    if (arg1_typeid == InvalidOid)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("could not determine input data type")));

    if (!AggCheckCallContext(fcinfo, &aggcontext))
	{
		/* cannot be called directly because of internal-type argument */
		elog(ERROR, "array_agg_2d+transfn called in non-aggregate context");
	}

    if (PG_ARGISNULL(0))
		state = initArrayResultArr(arg1_typeid, InvalidOid, aggcontext, false);
	else
		state = (ArrayBuildStateArr *) PG_GETARG_POINTER(0);

    state = accumArrayResultArr(state,
								PG_GETARG_DATUM(1),
								PG_ARGISNULL(1),
								arg1_typeid,
								aggcontext);
    PG_RETURN_POINTER(state);
}

PG_FUNCTION_INFO_V1(array_2d_agg_finalfn);
Datum
array_2d_agg_finalfn(PG_FUNCTION_ARGS)
{
    Datum		result;
	ArrayBuildStateArr *state;

    /* cannot be called directly because of internal-type argument */
	Assert(AggCheckCallContext(fcinfo, NULL));
    state = PG_ARGISNULL(0) ? NULL : (ArrayBuildStateArr *) PG_GETARG_POINTER(0);
    if (state == NULL)
		PG_RETURN_NULL();	

    result = makeAgg2dResultArr(state, CurrentMemoryContext, false);
    // result = makeArrayResultArr(state, CurrentMemoryContext, false);

	PG_RETURN_DATUM(result);
}

/* modify the original makeArrayResultArr to satisty the arr agg on the first dimension*/
Datum
makeAgg2dResultArr(ArrayBuildStateArr *astate,
				   MemoryContext rcontext,
				   bool release)
{
    ArrayType *result;
    MemoryContext oldcontext;

    /* Build the final array result in rcontext */
	oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);

    if (astate->ndims == 0)
	{
		/* No inputs, return empty array */
		result = construct_empty_array(astate->element_type);
	}
    else 
    {
        int			dataoffset,
					nbytes;
        /* Check for overflow of the array dimensions */
		(void) ArrayGetNItems(astate->ndims, astate->dims);
        
        /* Compute required space */
		nbytes = astate->nbytes;
		if (astate->nullbitmap != NULL)
		{
			dataoffset = ARR_OVERHEAD_WITHNULLS(astate->ndims-1, astate->nitems);
			nbytes += dataoffset;
		}
		else
		{
			dataoffset = 0;
			nbytes += ARR_OVERHEAD_NONULLS(astate->ndims-1);
		}

        result = (ArrayType *) palloc0(nbytes);
        SET_VARSIZE(result, nbytes);
        // important: modify ndims and dims
        int new_ndims = astate->ndims - 1;
        int new_dims[2];
        int new_lbs[2] = {1, 1};
        new_dims[0] = astate->dims[0] * astate->dims[1];
        new_dims[1] = astate->dims[2];
        
        result->ndim = new_ndims;
		result->dataoffset = dataoffset;
		result->elemtype = astate->element_type;

        memcpy(ARR_DIMS(result), new_dims, new_ndims * sizeof(int));
		memcpy(ARR_LBOUND(result), new_lbs, new_ndims * sizeof(int));
		memcpy(ARR_DATA_PTR(result), astate->data, astate->nbytes);

        if (astate->nullbitmap != NULL)
			array_bitmap_copy(ARR_NULLBITMAP(result), 0,
							  astate->nullbitmap, 0,
							  astate->nitems);
    }

    	MemoryContextSwitchTo(oldcontext);

	/* Clean up all the junk */
	if (release)
	{
		Assert(astate->private_cxt);
		MemoryContextDelete(astate->mcontext);
	}

	return PointerGetDatum(result);
}

PG_FUNCTION_INFO_V1(array_2d_agg_combine);
Datum
array_2d_agg_combine(PG_FUNCTION_ARGS)
{
	elog(INFO, "[agg_combine] agg_combine_start");
	clock_t start, end, tmp;
    double cpu_time_used;
    start = clock(); 
    ArrayBuildStateArr *state1;
	ArrayBuildStateArr *state2;
	MemoryContext agg_context;
	MemoryContext old_context;
    int i;

	if (!AggCheckCallContext(fcinfo, &agg_context))
		elog(ERROR, "aggregate function called in non-aggregate context");

	state1 = PG_ARGISNULL(0) ? NULL : (ArrayBuildStateArr *) PG_GETARG_POINTER(0);
	state2 = PG_ARGISNULL(1) ? NULL : (ArrayBuildStateArr *) PG_GETARG_POINTER(1);

    if (state2 == NULL)
	{
		/*
		 * NULL state2 is easy, just return state1, which we know is already
		 * in the agg_context
		 */
		if (state1 == NULL)
			PG_RETURN_NULL();
		PG_RETURN_POINTER(state1);
	}

    if (state1 == NULL)
	{
		/* We must copy state2's data into the agg_context */
		old_context = MemoryContextSwitchTo(agg_context);

		state1 = initArrayResultArr(state2->array_type, InvalidOid,
									agg_context, false);

		state1->abytes = state2->abytes;
		state1->data = (char *) palloc(state1->abytes);

		if (state2->nullbitmap)
		{
			int			size = (state2->aitems + 7) / 8;

			state1->nullbitmap = (bits8 *) palloc(size);
			memcpy(state1->nullbitmap, state2->nullbitmap, size);
		}

		memcpy(state1->data, state2->data, state2->nbytes);
		state1->nbytes = state2->nbytes;
		state1->aitems = state2->aitems;
		state1->nitems = state2->nitems;
		state1->ndims = state2->ndims;
		memcpy(state1->dims, state2->dims, sizeof(state2->dims));
		memcpy(state1->lbs, state2->lbs, sizeof(state2->lbs));
		state1->array_type = state2->array_type;
		state1->element_type = state2->element_type;

		MemoryContextSwitchTo(old_context);

		PG_RETURN_POINTER(state1);
	}

    /* We only need to combine the two states if state2 has any items */
	else if (state2->nitems > 0)
	{
		MemoryContext oldContext;
		int			reqsize = state1->nbytes + state2->nbytes;
		int			i;

		/*
		 * Check the states are compatible with each other.  Ensure we use the
		 * same error messages that are listed in accumArrayResultArr so that
		 * the same error is shown as would have been if we'd not used the
		 * combine function for the aggregation.
		 */
		if (state1->ndims != state2->ndims)
			ereport(ERROR,
					(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
					 errmsg("cannot accumulate arrays of different dimensionality")));

		/* Check dimensions match ignoring the first dimension. */
		for (i = 1; i < state1->ndims; i++)
		{
			if (state1->dims[i] != state2->dims[i] || state1->lbs[i] != state2->lbs[i])
				ereport(ERROR,
						(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
						 errmsg("cannot accumulate arrays of different dimensionality")));
		}


		oldContext = MemoryContextSwitchTo(state1->mcontext);

		/*
		 * If there's not enough space in state1 then we'll need to reallocate
		 * more.
		 */
		if (state1->abytes < reqsize)
		{
			/* use a power of 2 size rather than allocating just reqsize */
			state1->abytes = pg_nextpower2_32(reqsize);
			state1->data = (char *) repalloc(state1->data, state1->abytes);
		}

		if (state2->nullbitmap)
		{
			int			newnitems = state1->nitems + state2->nitems;

			if (state1->nullbitmap == NULL)
			{
				/*
				 * First input with nulls; we must retrospectively handle any
				 * previous inputs by marking all their items non-null.
				 */
				state1->aitems = pg_nextpower2_32(Max(256, newnitems + 1));
				state1->nullbitmap = (bits8 *) palloc((state1->aitems + 7) / 8);
				array_bitmap_copy(state1->nullbitmap, 0,
								  NULL, 0,
								  state1->nitems);
			}
			else if (newnitems > state1->aitems)
			{
				int			newaitems = state1->aitems + state2->aitems;

				state1->aitems = pg_nextpower2_32(newaitems);
				state1->nullbitmap = (bits8 *)
					repalloc(state1->nullbitmap, (state1->aitems + 7) / 8);
			}
			array_bitmap_copy(state1->nullbitmap, state1->nitems,
							  state2->nullbitmap, 0,
							  state2->nitems);
		}

		memcpy(state1->data + state1->nbytes, state2->data, state2->nbytes);
		state1->nbytes += state2->nbytes;
		state1->nitems += state2->nitems;

		state1->dims[0] += state2->dims[0];
		/* remaining dims already match, per test above */

		Assert(state1->array_type == state2->array_type);
		Assert(state1->element_type == state2->element_type);

		MemoryContextSwitchTo(oldContext);
	}

	end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    elog(INFO, "[agg_combine] took %f ms to execute.\n", cpu_time_used*1000);
    PG_RETURN_POINTER(state1);
}

/*refer to https://github.com/postgres/postgres/blob/master/src/backend/utils/adt/array_userfuncs.c*/
PG_FUNCTION_INFO_V1(array_agg_array_serialize);
Datum
array_agg_array_serialize(PG_FUNCTION_ARGS)
{
	ArrayBuildStateArr *state;
	StringInfoData buf;
	bytea	   *result;

	/* cannot be called directly because of internal-type argument */
	Assert(AggCheckCallContext(fcinfo, NULL));

	state = (ArrayBuildStateArr *) PG_GETARG_POINTER(0);

	pq_begintypsend(&buf);

	/*
	 * element_type. Putting this first is more convenient in deserialization
	 * so that we can init the new state sooner.
	 */
	pq_sendint32(&buf, state->element_type);

	/* array_type */
	pq_sendint32(&buf, state->array_type);

	/* nbytes */
	pq_sendint32(&buf, state->nbytes);

	/* data */
	pq_sendbytes(&buf, state->data, state->nbytes);

	/* abytes */
	pq_sendint32(&buf, state->abytes);

	/* aitems */
	pq_sendint32(&buf, state->aitems);

	/* nullbitmap */
	if (state->nullbitmap)
	{
		Assert(state->aitems > 0);
		pq_sendbytes(&buf, state->nullbitmap, (state->aitems + 7) / 8);
	}

	/* nitems */
	pq_sendint32(&buf, state->nitems);

	/* ndims */
	pq_sendint32(&buf, state->ndims);

	/* dims: XXX should we just send ndims elements? */
	// pq_sendbytes(&buf, state->dims, sizeof(state->dims));
    pq_sendbytes(&buf, (const char *)state->dims, sizeof(state->dims));

	/* lbs */
	// pq_sendbytes(&buf, state->lbs, sizeof(state->lbs));
    pq_sendbytes(&buf, (const char *)state->lbs, sizeof(state->lbs));

	result = pq_endtypsend(&buf);

	PG_RETURN_BYTEA_P(result);
}

PG_FUNCTION_INFO_V1(array_agg_array_deserialize);
Datum
array_agg_array_deserialize(PG_FUNCTION_ARGS)
{
	bytea	   *sstate;
	ArrayBuildStateArr *result;
	StringInfoData buf;
	Oid			element_type;
	Oid			array_type;
	int			nbytes;
	const char *temp;

	/* cannot be called directly because of internal-type argument */
	Assert(AggCheckCallContext(fcinfo, NULL));

	sstate = PG_GETARG_BYTEA_PP(0);

	/*
	 * Initialize a StringInfo so that we can "receive" it using the standard
	 * recv-function infrastructure.
	 */
	initReadOnlyStringInfo(&buf, VARDATA_ANY(sstate),
						   VARSIZE_ANY_EXHDR(sstate));

	/* element_type */
	element_type = pq_getmsgint(&buf, 4);

	/* array_type */
	array_type = pq_getmsgint(&buf, 4);

	/* nbytes */
	nbytes = pq_getmsgint(&buf, 4);

	result = initArrayResultArr(array_type, element_type,
								CurrentMemoryContext, false);

	result->abytes = 1024;
	while (result->abytes < nbytes)
		result->abytes *= 2;

	result->data = (char *) palloc(result->abytes);

	/* data */
	temp = pq_getmsgbytes(&buf, nbytes);
	memcpy(result->data, temp, nbytes);
	result->nbytes = nbytes;

	/* abytes */
	result->abytes = pq_getmsgint(&buf, 4);

	/* aitems: might be 0 */
	result->aitems = pq_getmsgint(&buf, 4);

	/* nullbitmap */
	if (result->aitems > 0)
	{
		int			size = (result->aitems + 7) / 8;

		result->nullbitmap = (bits8 *) palloc(size);
		temp = pq_getmsgbytes(&buf, size);
		memcpy(result->nullbitmap, temp, size);
	}
	else
		result->nullbitmap = NULL;

	/* nitems */
	result->nitems = pq_getmsgint(&buf, 4);

	/* ndims */
	result->ndims = pq_getmsgint(&buf, 4);

	/* dims */
	temp = pq_getmsgbytes(&buf, sizeof(result->dims));
	memcpy(result->dims, temp, sizeof(result->dims));

	/* lbs */
	temp = pq_getmsgbytes(&buf, sizeof(result->lbs));
	memcpy(result->lbs, temp, sizeof(result->lbs));

	pq_getmsgend(&buf);

	PG_RETURN_POINTER(result);
}

static inline void
initReadOnlyStringInfo(StringInfo str, char *data, int len)
{
	str->data = data;
	str->len = len;
	str->maxlen = 0;			/* read-only */
	str->cursor = 0;
}

PG_FUNCTION_INFO_V1(batchnorm);
Datum 
batchnorm(PG_FUNCTION_ARGS)
{
    ArrayType *input_arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *args_arr = PG_GETARG_ARRAYTYPE_P(1);

    float4 *input_data = (float4 *) ARR_DATA_PTR(input_arr);
    int *input_dims = ARR_DIMS(input_arr);  //1024, 27
    int channel = input_dims[0];
    int nelems_channel = input_dims[1];
    int size = channel * nelems_channel;
    
	ArrayType *res;
    int lbound[2] = {1, 1};
    int res_dims[2] = {channel, nelems_channel};

	float4 *args = (float4 *) ARR_DATA_PTR(args_arr);
    float4 *res_data = (float4 *) palloc(sizeof(float4) * size);
	float scale, bias, mean, var;
	int offset = 0;
	for (int c = 0; c < channel; c++) {
		scale = args[c];
		bias = args[channel + c];
		mean = args[2*channel + c];
		var = args[3*channel + c];
		for (int i = offset; i < offset + nelems_channel; i++) {
			res_data[i] = ((input_data[i] - mean) / sqrt(var)) + bias;
		}
		offset += nelems_channel;
	}

    res = float4construct_md_array(res_data, size, 2, res_dims, lbound, FLOAT4OID);
    PG_RETURN_ARRAYTYPE_P(res);
}

