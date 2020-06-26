TARGET=main
OBJECTS=util.o pix2pix.o matmul.o

CXX=g++
CXXFLAGS=-std=c++11 -Wall -mavx -O3
LDFLAGS=-lm -L/usr/local/cuda/lib64 -lcudart
NVFLAGS=-std=c++11 -O3 -arch sm_61

# If you use MPI, use the following lines instead of above lines
#CXX=mpic++
#CXXFLAGS=-std=c++11 -Wall -O3 -DUSE_MPI

all: $(TARGET)

$(TARGET): $(OBJECTS)

matmul.o: matmul.cu
	nvcc $(NVFLAGS) -c -o $@ $^

pix2pix.o: pix2pix.cu
	nvcc $(NVFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add --name pix2pix_C --device gpu/1080 ./$(TARGET) ../common/edges2cats_AtoB.bin $(ARGS)
