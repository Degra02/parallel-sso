#include "sso/parse_args.h"

int main(int argc, char *argv[]) {
    struct SSOConfig cfg;
    if (parse_args(argc, argv, &cfg) != 0) {
        return -1;
    }

    return 0;
}
