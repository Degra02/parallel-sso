# Compilers
CC := gcc
MPICC ?= mpicc

# Directories
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build

# Source files
COMMON_SRCS := $(SRC_DIR)/sso/parse_args.c $(SRC_DIR)/sso/ofuncs.c $(SRC_DIR)/sso/sso.c
HEADERS := $(wildcard $(INC_DIR)/*.h)

# Main source files for each executable
SERIAL_MAIN := $(SRC_DIR)/serial_main.c
MPI_SHARKS_MAIN := $(SRC_DIR)/mpi_sharks_main.c

# Object files
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(COMMON_SRCS))
SERIAL_OBJS := $(BUILD_DIR)/serial_main.o $(COMMON_OBJS)
SERIAL_V2_OBJS := $(BUILD_DIR)/serial_main_parallel_ready.o $(COMMON_OBJS)
PAR_OPENMP_SHARKS_OBJS := $(BUILD_DIR)/openmp_sharks_main.o $(COMMON_OBJS)
MPI_SHARKS_OBJS := $(BUILD_DIR)/mpi_sharks_main.o $(COMMON_OBJS)

# Executables
SERIAL_BIN := sso_serial
SERIAL_V2_BIN := sso_serial_v2
PAR_OPENMP_SHARKS_BIN := sso_openmp_sharks
MPI_SHARKS_BIN := sso_mpi_sharks

# Compiler flags
CFLAGS := -O2 -Wall -Wextra -Wpedantic -I$(INC_DIR)
LDFLAGS := -lm
OPENMP_FLAGS := -fopenmp

# Default target
all: $(SERIAL_BIN) $(SERIAL_V2_BIN) $(PAR_OPENMP_SHARKS_BIN) $(MPI_SHARKS_BIN)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)/sso

# Pattern rule for object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# OpenMP object file
$(BUILD_DIR)/openmp_sharks_main.o: $(SRC_DIR)/openmp_sharks_main.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c $< -o $@

# Serial executable
$(SERIAL_BIN): $(SERIAL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Serial executable
$(SERIAL_V2_BIN): $(SERIAL_V2_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# OpenMP parallel executable
$(PAR_OPENMP_SHARKS_BIN): $(PAR_OPENMP_SHARKS_OBJS)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(LDFLAGS) $(OPENMP_FLAGS)

# MPI executable
$(MPI_SHARKS_BIN): $(MPI_SHARKS_OBJS)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Convenience targets
clean:
	rm -rf $(BUILD_DIR) $(SERIAL_BIN) $(SERIAL_V2_BIN) $(PAR_OPENMP_SHARKS_BIN) $(MPI_SHARKS_BIN)

distclean: clean

help:
	@echo "Available targets:"
	@echo "  all       - Build all executables (default)"
	@echo "  $(SERIAL_BIN)  - Build serial version"
	@echo "  $(PAR_OPENMP_BIN)  - Build OpenMP sharks version"
	@echo "  $(MPI_SHARKS_BIN)    - Build MPI sharks version"
	@echo "  clean     - Remove build artifacts and executables"
	@echo "  distclean - Same as clean"
	@echo "  help      - Show this help message"

.PHONY: all clean distclean help
