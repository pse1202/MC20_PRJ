TARGET=main
OBJECTS=util.o pix2pix.o kernel.o matmul.o

CXX=mpic++
CXXFLAGS=-std=c++11 -Wall -mavx -O3 -DUSE_MPI
LDFLAGS=-lm -L/usr/local/cuda/lib64 -lcudart
NVFLAGS=-std=c++11 -O3 -arch sm_61

all: $(TARGET)

$(TARGET): $(OBJECTS)

matmul.o: matmul.cu
	nvcc $(NVFLAGS) -c -o $@ $^

kernel.o: kernel.cu
	nvcc $(NVFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add --mode mpi --nodes 4 --slots 1 --name pix2pix_D --device gpu/1080 ./$(TARGET) ../common/edges2cats_AtoB.bin $(ARGS)
