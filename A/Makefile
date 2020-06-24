TARGET=main
OBJECTS=util.o pix2pix.o matmul.o

CXX=g++
CXXFLAGS=-std=c++11 -Wall -mavx -fopenmp -O3
LDFLAGS=-lm -pthread
# If you use MPI, use the following lines instead of above lines
#CXX=mpic++
#CXXFLAGS=-std=c++11 -Wall -O3 -DUSE_MPI

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add --name pix2pix_A ./$(TARGET) ../common/edges2cats_AtoB.bin $(ARGS)
