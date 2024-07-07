/*-------------------------------------------------------------------------
 *
 * pg_nudf.h
 *	  definition of the "nudf" system catalog (pg_nudf)
 *
 * Author: Yuchen Peng
 *
 * src/include/catalog/pg_nudf.h
 *
 * NOTES
 *	  The Catalog.pm module reads this file and derives schema
 *	  information.
 *
 *-------------------------------------------------------------------------
 */
#ifndef PG_NUDF_H
#define PG_NUDF_H

#include "catalog/genbki.h"
#include "catalog/objectaddress.h"
#include "catalog/pg_nudf_d.h"
#include "commands/nudfcmds.h"

/* ----------------
 *		pg_proc definition.  cpp turns this into
 *		typedef struct FormData_pg_proc
 * ----------------
 */
CATALOG(pg_nudf,8932,NUDFRelationId)
{
    Oid oid; /* oid */
    NameData    nudfname;
    text nudfsrc;  /* equivalent sub-queries*/
} FormData_pg_nudf;

typedef FormData_pg_nudf *Form_pg_nudf;


DECLARE_UNIQUE_INDEX_PKEY(pg_nudf_oid_index, 8933, NUDFOidIndexId, on pg_nudf using btree(oid oid_ops));
DECLARE_UNIQUE_INDEX(pg_nudf_name_index, 8934, NUDFNameIndexId, on pg_nudf using btree(nudfname name_ops));
 
extern ObjectAddress NUDFCreate(const char *nudfName,
            const NUDFInfo *nudfsrc);

#endif /* PG_NUDF_H */