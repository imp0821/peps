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

#include "aqua_configs.h"

PG_FUNCTION_INFO_V1(conv_support);
Datum
conv_support(PG_FUNCTION_ARGS)
{
    Node    *rawreq = (Node *) PG_GETARG_POINTER(0);
    Node    *ret = NULL;
    
    if (IsA(rawreq, SupportRequestCost))
    {
        SupportRequestCost *req = (SupportRequestCost *) rawreq;
        float cost_in = 0;
        if (req->node && IsA(req->node, FuncExpr))  
        {  
            FuncExpr *fexpr = (FuncExpr *) req->node; 
            if (list_length(fexpr->args) > 0)  
            {  
                Node *last_arg = (Node *) llast(fexpr->args);  
                if (IsA(last_arg, Const))  
                {  
                    Const *const_arg = (Const *) last_arg; 
                    cost_in = DatumGetFloat4(const_arg->constvalue);
                }
            }
        }
        // elog(INFO, "[kfm_im2col] cost: %s", cost_in);
        req->per_tuple = cost_in;
        ret = (Node *) req;
        PG_RETURN_POINTER(ret);
        //PG_RETURN_POINTER((Node *)0);
    }

    PG_RETURN_POINTER(ret);
}

PG_FUNCTION_INFO_V1(kfm_nt_support);
Datum
kfm_nt_support(PG_FUNCTION_ARGS)
{
    Node    *rawreq = (Node *) PG_GETARG_POINTER(0);
    Node    *ret = NULL;
    
    if (IsA(rawreq, SupportRequestCost))
    {
        SupportRequestCost *req = (SupportRequestCost *) rawreq;
        int cost_in = 0;
        if (req->node && IsA(req->node, FuncExpr))  
        {  
            FuncExpr *fexpr = (FuncExpr *) req->node; 
            if (list_length(fexpr->args) > 0)  
            {  
                Node *last_arg = (Node *) llast(fexpr->args);  
                if (IsA(last_arg, Const))  
                {  
                    Const *const_arg = (Const *) last_arg; 
                    cost_in = DatumGetInt32(const_arg->constvalue);
                }
            }
        }
        // elog(INFO, "[kfm_nt] cost: %d", cost_in);
        req->per_tuple = 40000;
        // ret = (Node *) req;
        //PG_RETURN_POINTER((Node *)0);
    }

    PG_RETURN_POINTER(ret);
}

PG_FUNCTION_INFO_V1(kfm_support);
Datum
kfm_support(PG_FUNCTION_ARGS)
{
    Node    *rawreq = (Node *) PG_GETARG_POINTER(0);
    Node    *ret = NULL;
    // elog(INFO, "[kfm support] support type: %u", nodeTag(rawreq));
    if (IsA(rawreq, SupportRequestCost))
    {
        SupportRequestCost *req = (SupportRequestCost *) rawreq;
        req->startup = 0.25;
        req->per_tuple = 40000;
        ret = (Node *) req;
        //PG_RETURN_POINTER((Node *)0);
    }
    else if (IsA(rawreq, SupportRequestSimplify))
    {
      SupportRequestSimplify *req = (SupportRequestSimplify *) rawreq;      
      PG_RETURN_POINTER((Node *)0);
    }

    PG_RETURN_POINTER(ret);
}

PG_FUNCTION_INFO_V1(im2col_support);
Datum
im2col_support(PG_FUNCTION_ARGS)
{
    Node    *rawreq = (Node *) PG_GETARG_POINTER(0);
    Node    *ret = NULL;
    
    if (IsA(rawreq, SupportRequestCost))
    {
        SupportRequestCost *req = (SupportRequestCost *) rawreq;
        req->startup = 0.25;
        req->per_tuple = 40000;
        ret = (Node *) req;
        //PG_RETURN_POINTER((Node *)0);
    }
    else if (IsA(rawreq, SupportRequestSimplify))
    {
      SupportRequestSimplify *req = (SupportRequestSimplify *) rawreq;      
      PG_RETURN_POINTER((Node *)0);
    }

    PG_RETURN_POINTER(ret);
}


PG_FUNCTION_INFO_V1(im2col_init_support);
Datum
im2col_init_support(PG_FUNCTION_ARGS)
{
    Node    *rawreq = (Node *) PG_GETARG_POINTER(0);
    Node    *ret = NULL;
    // elog(INFO, "[kfm support] support type: %u", nodeTag(rawreq));
    if (IsA(rawreq, SupportRequestCost))
    {
        SupportRequestCost *req = (SupportRequestCost *) rawreq;
        req->startup = 0.25;
        req->per_tuple = 10000;
        ret = (Node *) req;
        //PG_RETURN_POINTER((Node *)0);
    }
    else if (IsA(rawreq, SupportRequestSimplify))
    {
      SupportRequestSimplify *req = (SupportRequestSimplify *) rawreq;      
      PG_RETURN_POINTER((Node *)0);
    }

    PG_RETURN_POINTER(ret);
}