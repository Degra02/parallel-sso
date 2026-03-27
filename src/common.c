#include "common.h"

ObjectiveFunction lookup_obj(const char *name) {
    for (size_t i = 0; obj_registry[i].name != NULL; ++i)
        if (strcmp(obj_registry[i].name, name) == 0)
            return obj_registry[i].value;

    fprintf(stderr, "Unknown objective: '%s'. Available:\n", name);
    for (size_t i = 0; obj_registry[i].name != NULL; ++i)
        fprintf(stderr, "  %s\n", obj_registry[i].name);
    exit(1);
}

void parse_args(int argc, char **argv, SSOConfig *cfg) {
    int opt, idx;
    while ((opt = getopt_long(argc, argv, "", long_opts, &idx)) != -1) {
        switch (opt) {
            case 1: cfg->np   = (uint32_t)atoi(optarg); break;
            case 2: cfg->nd   = (uint32_t)atoi(optarg); break;
            case 3: cfg->k_max = (uint32_t)atoi(optarg); break;
            case 4: cfg->m    = (uint32_t)atoi(optarg); break;
            case 5: cfg->n_k  = atof(optarg); break;
            case 6: cfg->a_k  = atof(optarg); break;
            case 7: cfg->b_k  = atof(optarg); break;
            case 8: cfg->seed = (uint64_t)strtoull(optarg, NULL, 10); break;
            case 9: cfg->obj = lookup_obj(optarg); break;
            default:
                fprintf(stderr, "Unknown option\n");
                exit(1);
        }
    }
}