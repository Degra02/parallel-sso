CC := gcc
MPICC ?= mpicc
CFLAGS := -O3 -Wall -Wextra -Iinclude
LDFLAGS := -lm

SRC_COMMON := src/common.c

all: sso_serial sso_mpi

sso_serial: src/serial_main.c $(SRC_COMMON) include/common.h
	$(CC) $(CFLAGS) -o $@ src/serial_main.c $(SRC_COMMON) $(LDFLAGS)

sso_mpi: src/mpi_main.c $(SRC_COMMON) include/common.h
	$(MPICC) $(CFLAGS) -o $@ src/mpi_main.c $(SRC_COMMON) $(LDFLAGS)

clean:
	rm -f sso_serial sso_mpi

.PHONY: all clean
