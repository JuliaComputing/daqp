#ifndef DAQP_BNB_PARALLEL_H
# define DAQP_BNB_PARALLEL_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "types.h"
#include "constants.h"

/*
 * Parallel branch and bound with a portfolio of workspaces (compiled with DAQP_PARALLEL).
 *
 * The n workspaces hold the same mixed-integer problem, set up separately, and may differ in their
 * settings (e.g., DAQPSettings.branching), which must not be shared between them. Each workspace is
 * solved by a thread of its own with a complete branch-and-bound search. The threads share the
 * objective of the best incumbent that any of them has found, which bounds the nodes of all of them,
 * and the search ends when the first thread has explored its tree, which proves that the best
 * incumbent is optimal (within abs_subopt and rel_subopt). The other threads are then stopped.
 *
 * The solution in res is that of the best incumbent. res->iter and res->nodes are the totals of all
 * threads, and res->solve_time is the elapsed (wall-clock) time. Returns the exit flag (also in
 * res->exitflag): DAQP_EXIT_UNSUPPORTED if the workspaces share settings or differ in their
 * dimensions. Without DAQP_PARALLEL, or if the problem has no binary constraints, the first workspace
 * is solved with daqp_solve. Declared in api.h:
 *   int daqp_solve_parallel(DAQPResult* res, DAQPWorkspace** works, const int n);
 */

#if defined(DAQP_PARALLEL) && !defined(__cplusplus)
#include <stdatomic.h>

typedef struct{
    _Atomic c_float J_bound; // Bound on the objective J of the nodes from the best incumbent of all threads
    atomic_int done; // Set when a thread has explored its tree
    _Atomic(DAQPWorkspace*) winner; // The workspace of the first thread that explored its tree
}DAQPBnBShared;
#endif

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif //ifndef DAQP_BNB_PARALLEL_H
