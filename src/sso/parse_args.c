#include "parse_args.h"
#include <argp.h>

/**
 *  @brief The parsing callback needed by argp, called for each option / parameter.
 *
 *  @see https://sourceware.org/glibc/manual/latest/html_node/Argp-Parser-Functions.html
 */
static error_t parser(int key, char *arg, struct argp_state *state);

static const char doc[] = "Program to perform shark smelling optimization.";
static const char args_doc[] = "";

static const struct argp_option options[] = {
    {"np",          'p', "int"  , 0, "The population size.",                    1},
    {"nd",          'd', "int"  , 0, "The number of decision variables.",       0},
    {"k_max",       'k', "int"  , 0, "The number of stages.",                   0},
    {"rotations",   'm', "int"  , 0, "Rotational points to check at each step.",2},
    {"eta",         'n', "[0-1]", 0, "Gradient scaling factor.",                0},
    {"alpha",       'a', "[0-1]", 0, "Momentum (inertia) rate.",                0},
    {"beta",        'b', "[0-1]", 0, "Velocity limiter ratio.",                 0},
    {"seed",        's', "int"  , 0, "PRNG seed (0 for random).",               3},
    {"obj",         'o', "int"  , 0, "The objective function.",                 0},
    { 0 } // This is needed to "terminate" the array.
};

static struct argp argp = { options, parser, args_doc, doc, 0, 0, 0 };

/**
 *  @brief Parse the command line arguments in a struct.
 *
 *  @param argc The number of command line arguments.
 *  @param argv The command line arguments.
 *
 *  @return The parsed argument. This function doesn't return in case of errors.
 */
error_t parse_args(int argc, char *argv[], struct SSOConfig *config) {
    *config = (struct SSOConfig) {
        .np         = 100,
        .nd         = 30,
        .k_max      = 1000,
        .rotations  = 10,
        .eta        = 0.9,
        .alpha      = 0.1,
        .beta       = 4.0,
        .obj        = OBJ_RASTRIGIN,
        .seed       = 1,
    };

    argp.children = NULL;

    return argp_parse(&argp, argc, argv, 0, 0, config);
}

error_t parse_args_extend(int argc, char *argv[], struct SSOConfig *config,
                          const struct argp *extension, void *ext_state) {
    *config = (struct SSOConfig) {
        .np         = 100,
        .nd         = 30,
        .k_max      = 1000,
        .rotations  = 10,
        .eta        = 0.9,
        .alpha      = 0.1,
        .beta       = 4.0,
        .obj        = OBJ_RASTRIGIN,
        .seed       = 1,
        .extension  = ext_state,
    };

    struct argp_child children[] = {{extension, 0, 0, 0}, {0}};

    argp.children = children;

    return argp_parse(&argp, argc, argv, 0, 0, config);
}

#define RET_PARSE_BOUNDED(size, args, field, val, min, max, ...) do {          \
        char *end;                                                             \
        (args)->field = strto##size((val), &end __VA_OPT__(,) __VA_ARGS__);    \
        if (*(val) == 0 || *end != 0) {                                        \
            perror("Couldn't parse " #field);                                  \
            return -1;                                                         \
        }                                                                      \
        if (*(val) < min || *(val) > max) {                                    \
            return -1;                                                         \
        }                                                                      \
        return 0;                                                              \
    } while(0)

#define RET_PARSE(size, args, field, val, ...) do {                            \
        char *end;                                                             \
        (args)->field = strto##size((val), &end __VA_OPT__(,) __VA_ARGS__);    \
        if (*(val) == 0 || *end != 0) {                                        \
            perror("Couldn't parse " #field);                                  \
            return -1;                                                         \
        }                                                                      \
        return 0;                                                              \
    } while(0)

#define RET_PARSE_ULL(args, field, val) RET_PARSE(ull, args, field, val, 0)
#define RET_PARSE_D(args, field, val) RET_PARSE(d, args, field, val)
#define RET_PARSE_D_BOUNDED(args, field, val, min, max)\
            RET_PARSE_BOUNDED(d, args, field, val, min, max)

static error_t parse_obj(struct SSOConfig *args, char *name) {
    for (size_t i = 0; obj_registry[i].name != NULL; ++i) {
        if (strcmp(obj_registry[i].name, name) == 0) {
            args->obj = obj_registry[i].value;
            return 0;
        }
    }

    fprintf(stderr, "Unknown objective function '%s'.\nChoose between:\n", name);
    for (size_t i = 0; obj_registry[i].name != NULL; ++i) {
        fprintf(stderr, "\t%s\n", obj_registry[i].name);
    }

    return EINVAL;
}

static error_t parser(int key, char *arg, struct argp_state *state) {
    struct SSOConfig *args = state->input;

    switch (key) {
        case ARGP_KEY_INIT:
            if (state->child_inputs != NULL && args != NULL) {
                state->child_inputs[0] = args->extension;
            }
            return 0;
        case 'p':
            RET_PARSE_ULL(args, np, arg);
        case 'd':
            RET_PARSE_ULL(args, nd, arg);
        case 'k':
            RET_PARSE_ULL(args, k_max, arg);
        case 'm':
            RET_PARSE_ULL(args, rotations, arg);
        case 'n':
            RET_PARSE_D_BOUNDED(args, eta, arg, 0.0, 1.0);
        case 'a':
            RET_PARSE_D_BOUNDED(args, alpha, arg, 0.0, 1.0);
        case 'b':
            RET_PARSE_D_BOUNDED(args, beta, arg, 0.0, 1.0);
        case 's':
            RET_PARSE_ULL(args, seed, arg);
        case 'o':
            return parse_obj(args, arg);
        case ARGP_KEY_END:
            return 0;
    }

    return ARGP_ERR_UNKNOWN;
}
