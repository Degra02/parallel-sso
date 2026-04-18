# Compilers
CC := gcc
MPICC ?= mpicc

# Directories
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build

# Source files
COMMON_SRCS := $(SRC_DIR)/parse_args.c $(SRC_DIR)/ofuncs.c
HEADERS := $(wildcard $(INC_DIR)/*.h)

# Main source files for each executable
SERIAL_MAIN := $(SRC_DIR)/serial_main.c
MPI_MAIN := $(SRC_DIR)/mpi_main.c

# Object files
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(COMMON_SRCS))
SERIAL_OBJS := $(BUILD_DIR)/serial_main.o $(COMMON_OBJS)
MPI_OBJS := $(BUILD_DIR)/mpi_main.o $(COMMON_OBJS)

# Executables
SERIAL_BIN := sso_serial
MPI_BIN := sso_mpi

# Compiler flags
CFLAGS := -O2 -Wall -Wextra -I$(INC_DIR)
LDFLAGS := -lm

# Default target
all: $(SERIAL_BIN) $(MPI_BIN)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Pattern rule for object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Serial executable
$(SERIAL_BIN): $(SERIAL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# MPI executable
$(MPI_BIN): $(MPI_OBJS)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Convenience targets
clean:
	rm -rf $(BUILD_DIR) $(SERIAL_BIN) $(MPI_BIN)

distclean: clean

help:
	@echo "Available targets:"
	@echo "  all       - Build all executables (default)"
	@echo "  $(SERIAL_BIN)  - Build serial version"
	@echo "  $(MPI_BIN)    - Build MPI version"
	@echo "  clean     - Remove build artifacts and executables"
	@echo "  distclean - Same as clean"
	@echo "  help      - Show this help message"

.PHONY: all clean distclean help
