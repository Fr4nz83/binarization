NVCC = nvcc
NVCC_FLAG = -std=c++17 -O3 -w -arch=sm_75 -maxrregcount=64 -rdc=true  # -Xptxas -v
LIBS = -ljpeg -lz -lboost_program_options -lcublas

# For debug
#NVCC_FLAG = -std=c++11 -w -O0 -g -G -arch=sm_70 -maxrregcount=64 -rdc=true -Xptxas -v


all: bmm dense
	
bmm: bmm.cu data.cpp generator.cpp ./cnpy/cnpy.cpp
	$(NVCC) $(NVCC_FLAG) -o $@ bmm.cu data.cpp ./cnpy/cnpy.cpp generator.cpp $(LIBS)
	
dense: dense.cu gpu_mat.cu utils.cpp data.cpp ./cnpy/cnpy.cpp
	$(NVCC) $(NVCC_FLAG) -o $@ dense.cu gpu_mat.cu utils.cpp data.cpp ./cnpy/cnpy.cpp generator.cpp $(LIBS)

clean:
	rm bmm dense
