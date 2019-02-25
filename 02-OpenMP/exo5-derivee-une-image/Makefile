# -------------------- #
# -- Label Makefile -- #
# -------------------- #

# -- Lile list ----------
FILE = main.cpp container.cpp ios.cpp pixmap_io.cpp


# -- Paths ----------
SRC_PATH = src
OBJ_PATH = obj
EXE_PATH = exe
INC_PATH = include


# -- Macros ----------
CC = g++
#CC = icpc

# -- Flags ----------
C_OPTIMISATION_FLAGS = -O3 -std=c++11 -fopenmp

C_INC_FLAGS = -I$(INC_PATH)

CFLAGS = $(C_OPTIMISATION_FLAGS) $(C_INC_FLAGS)
LDFLAGS = $(C_OPTIMISATION_FLAGS) $(C_INC_FLAGS)

# -- Final product ----------
PRODUCT   = Sobel.exe

# -- src and obj List ----------
SRC = $(addprefix ${SRC_PATH}/, $(FILE))
OBJ = $(addprefix ${OBJ_PATH}/, $(addsuffix .o, $(basename $(FILE))))

# -- Base rules ----------
$(OBJ_PATH)/%.o : $(SRC_PATH)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

#-----Main rule ----------
$(EXE_PATH)/$(PRODUCT): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) -lm 

# -- Other stuff ----------
clean:
	rm -f $(OBJ) *.pgm

