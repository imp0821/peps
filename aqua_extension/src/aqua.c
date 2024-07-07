#include "postgres.h"

#include "aqua_configs.h"
#include "create_nudf.h"

#include "parser/parser.h"
#include "tcop/tcopprot.h"
#include "parser/parse_func.h"
#include "commands/nudfcmds.h"
#include "catalog/pg_nudf.h"
#include "utils/snapmgr.h"
#include "nodes/nodeFuncs.h"
#include "utils/syscache.h"
#include "utils/ruleutils.h"
#include "nodes/value.h"

#include "optimizer/paths.h"
#include "utils/lsyscache.h"
#include "core.c"


 

PG_MODULE_MAGIC;
void _PG_init(void);
void _PG_fini(void);

static List * aqua_parser(const char *query_string);
char *aqua_select_parser(Node *node, const char *old_string);

static bool findFuncCallWalker(Node *node, void *context);
char* StrReplace(const char *query_string, const char *src, const char *dst);
char *ColumnRefToString(ColumnRef *columnRef);

typedef struct NUDFContext {
    char *nudf_name;
    char *nudf_query;  
    char *args;
} NUDFContext;
static NUDFContext *ParseNUDF(List *funcname); 

void aqua_set_rel_pathlist(PlannerInfo * root, RelOptInfo *rel,
								   Index rti, RangeTblEntry *rte);


/* Saved hook values in case of unload */
static prev_parser_hook_type old_prev_parser_hook = NULL;
static set_rel_pathlist_hook_type prev_set_rel_pathlist = NULL;
/*
 * Module load callback
 */
void
_PG_init(void)
{   
    elog(INFO, "[aqua_extension] PG_INIT");
    old_prev_parser_hook = prev_parser_hook;
    prev_parser_hook = aqua_parser;

    // prev_set_rel_pathlist = set_rel_pathlist_hook;
    // set_rel_pathlist_hook = aqua_set_rel_pathlist;
}

/*
 * Module unload callback
 */
void
_PG_fini(void)
{
    prev_parser_hook = old_prev_parser_hook;
    //set_rel_pathlist_hook = prev_set_rel_pathlist;
}

List *
aqua_parser(const char *query_string)
{
    List	   *raw_parsetree_list;
    ListCell   *lc;
    Node       *raw_node;

    raw_parsetree_list = raw_parser(query_string, RAW_PARSE_DEFAULT);
    lc = list_head(raw_parsetree_list);
    raw_node = (Node *) lfirst_node(RawStmt, lc)->stmt;
    if (nodeTag(raw_node) == T_CreateNUDFStmt)
    {   
        char    *funcname;
        char    *model_path;
        Oid     namespaceId;
        NUDFInfo       *nudfRes;
        ObjectAddress address;

        CreateNUDFStmt *stmt = (CreateNUDFStmt *) raw_node;
        /* Convert list of names to a name and namespace */
	    namespaceId = QualifiedNameGetCreationNamespace(stmt->funcname,
													&funcname);
        // return pg_parse_query("SELECT 1;");
    }
    if (IsA(raw_node, SelectStmt))
    {
        char *transformed_query;
        elog(INFO, "[aqua_parser]: %s",query_string);
        transformed_query = aqua_select_parser(raw_node, query_string);
        if (transformed_query != NULL)
        {
            query_string = transformed_query;
        }
    } 
    else if (IsA(raw_node, ExplainStmt))
    {
        ExplainStmt *explainStmt = (ExplainStmt *)raw_node;
        Node *innerStmt = (Node *)explainStmt->query;
        if (IsA(innerStmt, SelectStmt))
        {
            char *transformed_query;
            transformed_query = aqua_select_parser(innerStmt, query_string);
            if (transformed_query != NULL)
            {
                query_string = transformed_query;
            }
        }
    }
    return pg_parse_query(query_string);
}

char *aqua_select_parser(Node *node, const char *old_string)
{
    NUDFContext ctx;
    ctx.nudf_name = NULL;
    ctx.nudf_query = NULL;

    ListCell   *tlistitem;
    raw_expression_tree_walker(node, findFuncCallWalker, &ctx);
    if (ctx.nudf_query != NULL)
    {
        // elog(INFO, "[select parser] %s", ctx.nudf_query);
        StringInfoData full_nudf;
        initStringInfo(&full_nudf);
        appendStringInfo(&full_nudf, "%s(%s)", ctx.nudf_name, ctx.args);
        
        char *transformed_query = StrReplace(old_string, full_nudf.data, ctx.nudf_query);
        // elog(INFO, "transformed_query: %s", transformed_query);
        return transformed_query;
    }
    return NULL;
}

bool findFuncCallWalker(Node *node, void *context) 
{
    if (node == NULL)
        return false;
    //elog(INFO, "[walker] %d", nodeTag(node));
    if (IsA(node, FuncCall)) {
        FuncCall * fn = (FuncCall *) node;
        NUDFContext *nudf_ctx = ParseNUDF(fn->funcname);
        if (nudf_ctx != NULL)
        {
            //elog(INFO, "[walker] %s", nudf_ctx->nudf_query);
            ((NUDFContext *) context)->nudf_name = nudf_ctx->nudf_name;
            ((NUDFContext *) context)->nudf_query = nudf_ctx->nudf_query;
            
            StringInfoData arg_strs;
            initStringInfo(&arg_strs);
            List *args = fn->args;
            ListCell *arg;
            foreach(arg, args)
            {
                Node *expr = (Node *)lfirst(arg);
                char *arg_str;
                if (IsA(expr, ColumnRef)) 
                    arg_str = ColumnRefToString((ColumnRef *)expr);
                if (arg_str != NULL)
                    appendStringInfoString(&arg_strs, arg_str);
            }
            ((NUDFContext *) context)->args = arg_strs.data;
            return true;
        }
        return false;
    }

    return raw_expression_tree_walker(node, findFuncCallWalker, context);
}

NUDFContext *ParseNUDF(List *funcname) 
{
    char	       *schemaname;
    char           *nudfname;
    char           *nudf_query;
    NUDFContext    *ctx;
    HeapTuple	    ftup;
    Form_pg_nudf    pform;

    /* deconstruct the name list */
    DeconstructQualifiedName(funcname, &schemaname, &nudfname);
    /* Search syscache by nudfname only*/
    ftup = SearchSysCache1(NUDFNAME, CStringGetDatum(nudfname));
    if (!HeapTupleIsValid(ftup)) {
        return NULL;
    }
    pform = (Form_pg_nudf) GETSTRUCT(ftup);
    nudf_query = text_to_cstring(&pform->nudfsrc);

    ctx = (NUDFContext *) palloc(sizeof(NUDFContext));
    ctx->nudf_name = nudfname;
    ctx->nudf_query = nudf_query;
    ReleaseSysCache(ftup);
    return ctx;
}

char* StrReplace(const char *query_string, const char *src, const char *dst)
{
    StringInfoData res;
    initStringInfo(&res);

    const char *pos = strstr(query_string, src);
    if (pos == NULL) 
    {
        appendStringInfoString(&res, query_string);
    } else 
    {
        size_t before_len = pos - query_string;
        appendBinaryStringInfo(&res, query_string, before_len);
        appendStringInfo(&res, "(%s)", dst);
        appendStringInfoString(&res, pos + strlen(src));
    }
    return res.data;
}

char *ColumnRefToString(ColumnRef *columnRef) {
    StringInfoData str;
    initStringInfo(&str);
    ListCell *lc;

    foreach(lc, columnRef->fields) {
        Node *field = (Node *)lfirst(lc);
        // elog(INFO, "[column] %d", nodeTag(field));
        if (IsA(field, String)) {
            String *val = (String *)field;
            if (str.len > 0) {
                appendStringInfoChar(&str, '.');
            }
            appendStringInfoString(&str, val->sval);
        }
    }
    return str.data;
}

void
aqua_set_rel_pathlist(PlannerInfo * root, RelOptInfo *rel,
							  Index rti, RangeTblEntry *rte)
{
    ListCell	   *l;
    /* call the previous hook */
	if (prev_set_rel_pathlist)
		prev_set_rel_pathlist(root, rel, rti, rte);

    if (rel->rtekind != RTE_RELATION &&
		 rel->rtekind != RTE_SUBQUERY)
		return;
    
    if (!IS_SIMPLE_REL(rel))
        return;
    Oid relid = rte->relid;
    char *relname = get_rel_name(relid);
    if (strcmp(relname, "conv2d_177_split") == 0) {
        elog(INFO, "Processing relation: %s", relname);
        
        /* Just discard all the paths considered so far */
        list_free_deep(rel->pathlist);
        rel->pathlist = NIL;
        list_free_deep(rel->partial_pathlist);
        rel->partial_pathlist = NIL;

        /* Regenerate paths with the current enforcement */
		set_plain_rel_pathlist(root, rel, rte);

        foreach (l, rel->partial_pathlist)
        {
            Path *path = (Path *) lfirst(l);
            // elog(INFO, "[ppath] %d", path->parallel_workers);
            path->startup_cost = 0;
            path->total_cost = 0;

            if (path->parallel_safe)
				path->parallel_workers	= 4;
            
            if (rel->reloptkind == RELOPT_BASEREL)
                generate_useful_gather_paths(root, rel, false);
        }
    }
     
}
 