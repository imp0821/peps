/*-------------------------------------------------------------------------
 *
 * pg_nudf.c
 *	  routines to support manipulation of the pg_nudf relation
 *
 * Author: Yuchen Peng
 *
 *
 * IDENTIFICATION
 *	  src/backend/catalog/pg_nudf.c
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "access/htup_details.h"
#include "access/table.h"
#include "access/xact.h"
#include "catalog/catalog.h"
#include "catalog/dependency.h"
#include "catalog/indexing.h"
#include "catalog/objectaccess.h" 
#include "catalog/pg_nudf.h"
#include "commands/defrem.h"
#include "tcop/tcopprot.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/syscache.h"

ObjectAddress
NUDFCreate(const char *nudfName,
            const NUDFInfo *nudfInfo)
{
    Oid         retval;
    bool        nulls[Natts_pg_nudf];
    Datum       values[Natts_pg_nudf];
    NameData    nudfname;
    Relation    rel;
    HeapTuple   tup;
    HeapTuple   oldtup;
    ObjectAddress myself;
    int i;
    /*
	 * sanity checks
	 */
	Assert(PointerIsValid(nudfsrc));

    for (i = 0; i < Natts_pg_nudf; ++i)
    {
        nulls[i] = false;
        values[i] = (Datum) 0;
    }

    namestrcpy(&nudfname, nudfName);
    values[Anum_pg_nudf_nudfname - 1] = NameGetDatum(&nudfname);
    values[Anum_pg_nudf_nudfsrc - 1] = CStringGetTextDatum(nudfInfo->nudf_query);

    rel = table_open(NUDFRelationId, RowExclusiveLock);

    /* Check for pre-existing nudfs*/
    oldtup = SearchSysCache1(NUDFNAME, PointerGetDatum(nudfName));
    if (HeapTupleIsValid(oldtup)) {
        /* default not to replace a pre-existing nudf*/
        ereport(ERROR,
					(errcode(ERRCODE_DUPLICATE_FUNCTION),
					 errmsg("nudf \"%s\" already exists",
							nudfName)));
    }
    else
    {
        /* Creating a new nudf*/
        Oid         newOid;
        newOid = GetNewOidWithIndex(rel, NUDFOidIndexId,
                                    Anum_pg_nudf_oid);
        values[Anum_pg_nudf_oid -1 ] = ObjectIdGetDatum(newOid);
        tup = heap_form_tuple(rel->rd_att, values, nulls);
        CatalogTupleInsert(rel, tup);   
    }

    retval = ((Form_pg_nudf) GETSTRUCT(tup))->oid;

    myself.classId = NUDFRelationId;
    myself.objectId = retval;
    myself.objectSubId = 0;

    heap_freetuple(tup);
    table_close(rel, RowExclusiveLock);
    
    return myself;
}
