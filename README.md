### aqua extension
make clean && make USE_PGXS=1 install

### pg-15
make clean && make -j32  && make install 
make clean && make -j32 > ./make.log && make install  
pg_ctl -D /home/pyc/data/pg15-data -l logfile restart

create inference_function color() model 'test.onnx';