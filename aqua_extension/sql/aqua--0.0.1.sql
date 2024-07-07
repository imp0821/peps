-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION aqua" to load this file. \quit

CREATE FUNCTION test_flops() 
RETURNS REAL
AS '$libdir/aqua', 'test_flops' 
LANGUAGE C STRICT;  

CREATE TYPE virtual AS (  
    raw_data bytea,  
    array_data real[][]   
);  

CREATE FUNCTION virtual_in(cstring)
RETURNS virtual
AS '$libdir/aqua'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;


-- CREATE TYPE virtual (
--   INPUT          = virtual_in,
--   OUTPUT         = virtual_out
-- );

-- CREATE FUNCTION virtual_arr(virtual)
-- RETURNS real[]
-- AS '$libdir/aqua'
-- LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION conv_support(internal) RETURNS internal
AS '$libdir/aqua', 'conv_support'
LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION im2col_init_support(internal) RETURNS internal
AS '$libdir/aqua', 'im2col_init_support'
LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION im2col_support(internal) RETURNS internal
AS '$libdir/aqua', 'im2col_support'
LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION im2col(input_f real[][], kernel INT = 3, padding INT = 1, stride INT = 1)
RETURNS real[][]
AS '$libdir/aqua', 'im2col'
LANGUAGE C STRICT 
SUPPORT im2col_support
PARALLEL SAFE;

CREATE FUNCTION maxpool(input_f real[][], kernel INT = 2, padding INT = 0, stride INT = 2, op_cost REAL = 0.0)
RETURNS real[][]
AS '$libdir/aqua', 'maxpool'
LANGUAGE C STRICT
SUPPORT conv_support
PARALLEL SAFE;

CREATE FUNCTION avgpool(input_f real[][], kernel INT = 2, padding INT = 0, stride INT = 2, op_cost REAL = 0.0)
RETURNS real[][]
AS '$libdir/aqua', 'avgpool'
LANGUAGE C STRICT
SUPPORT conv_support
PARALLEL SAFE;

CREATE FUNCTION avgpool_conv(kmatrix real[][],  fmatrix real[][], bias real[], 
                            kernel INT = 2, padding INT = 0, stride INT = 2)
RETURNS real[][]
AS '$libdir/aqua', 'avgpool_conv'
LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION relu(input_f real[][])
RETURNS real[][]
AS '$libdir/aqua', 'relu'
LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION kfm_support(internal) RETURNS internal
AS '$libdir/aqua', 'kfm_support'
LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION kfm(kmatrix real[][], fmatrix real[][], bias real[])
RETURNS real[][]
AS '$libdir/aqua', 'kfm'
LANGUAGE C STRICT
SUPPORT kfm_support PARALLEL SAFE;

CREATE FUNCTION kfm_nt_support(internal) RETURNS internal
AS '$libdir/aqua', 'kfm_nt_support'
LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION kfm_nt(kmatrix real[][], fmatrix real[][], bias real[], acti BOOLEAN DEFAULT false)
RETURNS real[][]
AS '$libdir/aqua', 'kfm_nt'
LANGUAGE C STRICT
PARALLEL SAFE;

CREATE FUNCTION kfm_im2col(kmatrix real[][], fmatrix real[][], bias real[], 
                    kernel INT = 3, padding INT = 1, stride INT = 1,
                    op_cost REAL = 0.0)
RETURNS real[][]
AS '$libdir/aqua', 'kfm_im2col'
LANGUAGE C STRICT
SUPPORT conv_support
PARALLEL SAFE;

CREATE FUNCTION kfm_im2col_ns(kmatrix real[][], fmatrix real[][], bias real[], kernel_H INT, kernel_W INT, padding_H INT, padding_W INT, stride INT)
RETURNS real[][]
AS '$libdir/aqua', 'kfm_im2col_ns'
LANGUAGE C STRICT
PARALLEL SAFE;

CREATE FUNCTION group_kfm_im2col(kmatrix real[][], fmatrix real[][][], bias real[], kernel INT = 3, padding INT = 1, stride INT = 1, groups INT = 8)
RETURNS real[][]
AS '$libdir/aqua', 'group_kfm_im2col'
LANGUAGE C STRICT
PARALLEL SAFE;


CREATE FUNCTION madd(matrix1 real[][], matrix2 real[][])
RETURNS real[][]
AS '$libdir/aqua', 'madd'
LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION mvm(matrix real[][], vector real[], bias real[], op_cost REAL = 0.0)
RETURNS real[]
AS '$libdir/aqua', 'mvm'
LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION argmax(real[])
RETURNS integer AS $$
SELECT row_number::int
FROM (
  SELECT row_number() OVER() AS row_number, x
  FROM unnest($1) AS t(x) 
  ORDER BY x DESC
  LIMIT 1  
) t;
$$ LANGUAGE sql PARALLEL SAFE;

CREATE TYPE split_type AS (
    id int,
    array_data real[][]
);

CREATE OR REPLACE FUNCTION split_array(input real[][], split_num int)
  RETURNS SETOF split_type
  AS '$libdir/aqua', 'split_array'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE TYPE split_vertical_type AS (
    split1 real[][],
    split2 real[][],
    split3 real[][],
    split4 real[][]
);

CREATE OR REPLACE FUNCTION split_vertical_array(input real[][], split_num int)
  RETURNS split_vertical_type
  AS '$libdir/aqua', 'split_vertical_array'
  LANGUAGE C STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION concat_array(
    num int,
    input1 REAL[][] = ARRAY[]::REAL[],
    input2 REAL[][] = ARRAY[]::REAL[],
    input3 REAL[][] = ARRAY[]::REAL[],
    input4 REAL[][] = ARRAY[]::REAL[],
    input5 REAL[][] = ARRAY[]::REAL[],
    input6 REAL[][] = ARRAY[]::REAL[]
    )
  RETURNS real[][]
  AS '$libdir/aqua', 'concat_array'
  LANGUAGE C STRICT PARALLEL SAFE;


CREATE FUNCTION array_2d_agg_state (internal, real[][])
RETURNS internal
AS '$libdir/aqua','array_2d_agg_transfn'
PARALLEL SAFE
LANGUAGE C;

CREATE FUNCTION array_2d_agg_final_array (internal)
RETURNS real[][]
AS '$libdir/aqua','array_2d_agg_finalfn'
PARALLEL SAFE
LANGUAGE C;

CREATE FUNCTION array_2d_agg_combine(internal, internal)
RETURNS internal 
AS '$libdir/aqua','array_2d_agg_combine'
PARALLEL SAFE
LANGUAGE C;

CREATE FUNCTION array_agg_array_serialize(internal)
RETURNS bytea
AS '$libdir/aqua','array_agg_array_serialize'
PARALLEL SAFE
LANGUAGE C;

CREATE FUNCTION array_agg_array_deserialize(bytea, internal)
RETURNS internal
AS '$libdir/aqua', 'array_agg_array_deserialize'
PARALLEL SAFE
LANGUAGE C;

CREATE AGGREGATE array_2d_agg(real[][]) (
	SFUNC = array_2d_agg_state,
	STYPE = internal,
	FINALFUNC = array_2d_agg_final_array,
  COMBINEFUNC = array_2d_agg_combine,
  SERIALFUNC = array_agg_array_serialize,
  DESERIALFUNC = array_agg_array_deserialize,
	PARALLEL = SAFE
);

CREATE FUNCTION batchnorm(fmatrix real[][], args real[])
RETURNS real[][]
AS '$libdir/aqua', 'batchnorm'
LANGUAGE C STRICT
PARALLEL SAFE;

CREATE FUNCTION conv_text(kmatrix real[][], fmatrix real[][], bias real[], kernel_H INT, kernel_W INT, padding_H INT, padding_W INT, stride INT)
RETURNS real[][]
AS '$libdir/aqua', 'conv_text'
LANGUAGE C STRICT
PARALLEL SAFE;

