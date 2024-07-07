#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "access/xact.h"
#include "executor/spi.h"
#include "utils/guc.h"
#include "commands/nudfcmds.h"

#include <Python.h>

NUDFInfo *create_inference_function (const char *func_name, const char *model_path);