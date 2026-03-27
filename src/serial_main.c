#include "common.h"

int main(int argc, char const *argv[]) {
    SSOConfig cfg;
    parse_args(argc, (char **)argv, &cfg);

    printf("Configuration:\n");
    printf("  np: %u\n", cfg.np);

    return 0;
}