# -------------------- #
# -- myRem Makefile -- #
# -------------------- #
FILE= 
CUDAFILE= main.cu gpu_sobel.cu container.cu ios.cu pixmap_io.cu

SRC_PATH = src
OBJ_PATH = obj
MODULE_PATH = exe
INC_PATH = include

RM := rm -rf
CC = g++-6
NVCC = nvcc -ccbin $(CC)

CXXFLAGS = -std=c++11 -Xcompiler -fopenmp
CXXCUDAFLAGS =
CUDAFLAGS = arch=compute_50,code=sm_50
LD_FLAGS = 
# CUDAFLAGS = arch=compute_30,code=sm_30
# -Xptxas -dscm=cg
CXX_INC_FLAGS = -I$(INC_PATH)
SPECIALSTUFF =
OPTFLAGS = -O3
TARGET = sobel.exe

# -- src and obj List ----------
OBJ = $(addprefix ${OBJ_PATH}/, $(addsuffix .o, $(basename $(FILE))))
OBJ += $(addprefix ${OBJ_PATH}/, $(addsuffix .o, $(basename $(CUDAFILE))))

# -- Base rules ----------
$(OBJ_PATH)/%.o : $(SRC_PATH)/%.cu
	$(NVCC) -I/usr/local/cuda/include $(CXX_INC_FLAGS) -dc $< -o $@ 
	 # --ptxas-options=-v


#-----Main rule ----------
$(MODULE_PATH)/$(TARGET): $(OBJ)
	$(NVCC) -gencode $(CUDAFLAGS) -I/usr/local/cuda/include -o $@ $^ $(CXXFLAGS) $(OPTFLAGS) $(CXX_INC_FLAGS) $(LD_FLAGS) 


clean:
	rm -f $(OBJ)

debug:
	/usr/local/cuda/bin/nvvp $(MODULE_PATH)/$(TARGET)
