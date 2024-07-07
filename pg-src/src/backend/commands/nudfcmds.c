/*-------------------------------------------------------------------------
 *
 * nudfcmds.c
 *
 *	  Routines for CREATE INFERENCE FUNCTION
 *
 *
 * IDENTIFICATION
 *	  src/backend/commands/nudfcmds.h
 *
 * DESCRIPTION
 *	  These routines take the parse tree and pick out the
 *	  appropriate arguments/flags, and pass the results to the
 *	  corresponding "FooDefine" routines (in src/catalog) that do
 *	  the actual catalog-munging.  These routines also verify permission
 *	  of the user to execute the command.
 *
 * NOTES
 *	  These things must be defined and committed in the following order:
 *		"create function":
 *				input/output, recv/send procedures
 *		"create type":
 *				type
 *		"create operator":
 *				operators
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "access/genam.h"
#include "access/htup_details.h"
#include "access/sysattr.h"
#include "access/table.h"
#include "catalog/catalog.h"
#include "catalog/dependency.h"
#include "catalog/indexing.h"
#include "catalog/objectaccess.h"
#include "catalog/pg_nudf.h"
#include "commands/alter.h"
#include "commands/defrem.h"
#include "commands/extension.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "parser/parse_func.h"
#include "pgstat.h"
#include "tcop/pquery.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "commands/nudfcmds.h"

#include <dlfcn.h>

ObjectAddress
CreateNUDF(ParseState *pstate, CreateNUDFStmt *stmt)
{   
    char           *funcname;
    Oid             namespaceId;
    Oid             extId;
    void           *handle;
    char            extPath[MAXPGPATH];
    NUDFInfo       *(*create_inference_function)(const char *func_name, const char *model_path);
    NUDFInfo       *nudfRes;
    /* Convert list of names to a name and namespace */
	namespaceId = QualifiedNameGetCreationNamespace(stmt->funcname,
													&funcname);
    
    snprintf(extPath, MAXPGPATH, "%s/%s", pkglib_path, "aqua.so");
    /* Check if aqua extension is installed */
    extId = get_extension_oid("aqua", true);
    if (extId == InvalidOid) {
        elog(INFO, "[nudfcmd] aqua extension is not installed!");
    } else {
        handle = dlopen(extPath, RTLD_NOW);
        if (!handle) {
            elog(INFO, "[nudfcmd] cannot open aqua link !");
        } else {
            *(void **)(&create_inference_function) = dlsym(handle, "create_inference_function");
            if (!create_inference_function) {
                elog(INFO, "[nudfcmd] symbol does not exist!");
            } else {
                nudfRes = create_inference_function(funcname, stmt->model_path);
            }
            dlclose(handle);
        }
    }

    return NUDFCreate(funcname, nudfRes);
}
