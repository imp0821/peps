#include "create_nudf.h"

NUDFInfo* 
create_inference_function (const char *func_name, const char *model_path)
{
    PyObject *pName, *pModule, *pDict, *pClass, *pInstance, *pValue;
    NUDFInfo *res;
    elog(INFO, "[aqua_parser] nudf res %s %s",func_name, model_path);
    Py_Initialize();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/pyc/workspace/aqua_v2/aqua_extension/src/dbocg')");

    pName = PyUnicode_DecodeFSDefault("nudf");
    pModule = PyImport_Import(pName);
   
   if (pModule != NULL) {
        pDict = PyModule_GetDict(pModule);
        pClass = PyDict_GetItemString(pDict, "nUDF");

        PyObject *pArgs = PyTuple_New(2);
        PyTuple_SetItem(pArgs, 0, PyUnicode_FromString("color"));
        PyTuple_SetItem(pArgs, 1, PyUnicode_FromString("test.onnx"));

        if (pClass && PyCallable_Check(pClass)) {
            pInstance = PyObject_CallObject(pClass, pArgs);
            Py_DECREF(pArgs);
            if (pInstance == NULL && PyErr_Occurred()) {
                elog(WARNING, "[create_nudf] Python error occurred during instantiation of nUDF class");
                PyErr_Print();  
            }
            pValue = PyObject_CallMethod(pInstance, "export", NULL);
            if (pValue != NULL) {
                const char *result = PyUnicode_AsUTF8(pValue);
                res->nudf_query = result;
                // elog(INFO, "[create_nudf] create_nudf res: \n%s", result);
            } else {
                elog(ERROR, "[create_nudf] fail to call create_nudf");
                PyErr_Print();
            }
        } else {
            elog(ERROR, "[create_nudf] fail to find nUDF class");
        }
    } else {
        elog(ERROR, "[create_nudf] fail to find pModule");
    }
    Py_DECREF(pName);

    return res;
}