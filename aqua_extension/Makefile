EXTENSION = aqua             # the extension name
MODULE_big = aqua
DATA = sql/aqua--0.0.1.sql    # script files to install
# REGRESS   = aqua_test         # our test script file (without extension)
SRCS = $(wildcard src/*.c)
# OBJS = virtual.o nudf.o aqua.o          # our c module file to build
OBJS = $(SRCS:.c=.o)
# Postgres build stuff
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)

# Python configuration
PYTHON_CONFIG = /home/pyc/miniconda2/envs/cpu-aqua/bin/python3-config
PYTHON_INCLUDES = $(shell $(PYTHON_CONFIG) --includes)
# PYTHON_LDFLAGS = $(shell $(PYTHON_CONFIG) --ldflags)
PYTHON_LIB_PATH= /home/pyc/miniconda2/envs/cpu-aqua/lib/

CFLAGS += -I/usr/include/openblas
override CFLAGS += -std=c99 $(PYTHON_INCLUDES) -fopenmp -O3 -march=native
# -I/home/pyc/miniconda2/envs/cpu-aqua/include  
SHLIB_LINK += -L$(PYTHON_LIB_PATH) -lpython3.9
# SHLIB_LINK += -L$(PYTHON_LIB_PATH) -lpython3.9 -lmkl_rt 

LIBJPEG_LIBS := /usr/lib/x86_64-linux-gnu/libjpeg.so.8
SHLIB_LINK += $(LIBJPEG_LIBS)

BLAS_LIBS :=  /usr/lib/libopenblasp-r0.2.18.so
SHLIB_LINK += $(BLAS_LIBS)

include $(PGXS)