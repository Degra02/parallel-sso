# Compilers
CC := mpicc

# Directories
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build

# Source files
COMMON_SRCS := $(wildcard $(SRC_DIR)/*/*.c)
HEADERS := $(wildcard $(SRC_DIR)/**.h)

# Main source files for each executable
SERIAL_MAIN := $(SRC_DIR)/serial_main.c
MPI_SHARKS_MAIN := $(SRC_DIR)/mpi_sharks_main.c

# Object files
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(COMMON_SRCS))

# Compiler flags
CFLAGS := -O2 -Wall -Wextra -Wpedantic -I$(INC_DIR) -march=native
LDFLAGS := -lm
OPENMP_FLAGS := -fopenmp



# Default target
help:
	@echo "Available targets:"
	@echo "  all       - Build all executables (default)"
	@echo "  sso_<type>_<section>  - Build parallel version: <type> = [mpi, openmp, hybrid], <section> = [sharks, dim]"
	@echo "  clean     - Remove build artifacts and executables"
	@echo "  distclean - Same as clean"
	@echo "  help      - Show this help message"

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)/sso/

# Pattern rule for object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c $< -o $@

sso_%: $(COMMON_OBJS) $(BUILD_DIR)/main_%.o
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(LDFLAGS)

# Convenience targets
clean:
	rm -rf $(BUILD_DIR) sso_*

distclean: clean

.PHONY: clean distclean help
