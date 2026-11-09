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
 * The n workspaces hold the same mixed-integer problem, set up separately, each with settings of its
 * own (not shared between them). The settings may only differ in the selection of the nodes
 * (DAQPSettings.branching) and in the limits time_limit and iter_limit: the search of a thread is a
 * certificate of optimality for the incumbents of all threads only if they search the same problem
 * (the same cutoff fval_bound, regularization, equality reduction and tolerances). Each workspace is
 * solved by a thread of its own with a complete branch-and-bound search. The threads share the
 * objective of the best incumbent that any of them has found, which bounds the nodes of all of them,
 * and the search ends when the first thread has explored its tree, which proves that the best
 * incumbent is optimal (within abs_subopt and rel_subopt). The other threads are then stopped.
 *
 * The solution in res is that of the best incumbent. res->iter and res->nodes are the totals of all
 * threads, and res->solve_time is the elapsed (wall-clock) time. Returns the exit flag (also in
 * res->exitflag): DAQP_EXIT_UNSUPPORTED if n < 1, or if the workspaces share settings, differ in
 * their dimensions or state (equality reduction, regularization), or differ in other settings.
 * Without DAQP_PARALLEL, or if the problem has no binary constraints, the first workspace is solved
 * with daqp_solve. Declared in api.h:
 *   int daqp_solve_parallel(DAQPResult* res, DAQPWorkspace** works, const int n);
 *
 * Groups (a split of the search space):
 *   int daqp_solve_parallel_groups(DAQPResult* res, DAQPWorkspace** works, const int* groups, const int n);
 * Workspace i belongs to the group groups[i] (the groups are numbered 0, 1, ..., n_groups-1, each with
 * at least one workspace; groups == NULL is one group, as daqp_solve_parallel). The workspaces of a group
 * hold the same problem, as above. The problems of different groups are restrictions of one problem P:
 * they have the same dimensions, H, f, A, soft and binary constraints and settings (except branching,
 * time_limit, iter_limit and the settings that only change the order of the search), and differ in the
 * bounds, e.g., binaries fixed to different values. The caller is responsible that the feasible set of
 * each group is a subset of that of P and that their union is the feasible set of P; only the data
 * listed above is compared.
 *
 * Certification: an incumbent of any group is feasible for P and has the objective of P, so the bound
 * on J from the best incumbent of all groups is valid in every group (unless the Hessian is singular,
 * see below). A group is complete when one of its threads has explored its tree, which proves that no
 * point of the feasible set of the group improves on the best incumbent (by more than abs_subopt and
 * rel_subopt), or when one of its threads proves its problem infeasible. The completion of a group
 * certifies its own subset only; the other threads of the group are then stopped. When all groups are
 * complete, the best incumbent of all threads is optimal for P (DAQP_EXIT_INFEASIBLE if there is none).
 * A thread that ends early (limits, failure of the inner solver) does not complete its group; if all
 * threads of a group end early, the group cannot be completed, all threads are stopped and the best
 * incumbent is reported as DAQP_EXIT_FEASIBLE (or the exit flag of such a thread if there is none).
 *
 * With a singular Hessian (daqp_bnb_prox), the regularization depends on the bounds and differs between
 * the groups, so the bound is only shared within each group. The result is the best of the groups
 * (in the objective without the regularization), and it is reported as DAQP_EXIT_OPTIMAL_INEXACT
 * unless, for every group, the thread that found its best incumbent has explored its tree itself and
 * found it exact (or the group is infeasible). Groups require DAQP_PARALLEL and binary constraints
 * (otherwise DAQP_EXIT_UNSUPPORTED if there is more than one group).
 */

#if defined(DAQP_PARALLEL) && !defined(__cplusplus)
#include <stdatomic.h>

// The state of a group of threads of a parallel solve (work->bnb->shared)
typedef struct{
    _Atomic c_float* J_bound; // Bound on the objective J of the nodes from the best incumbent (of all groups, or of this group)
    atomic_int* stop; // Set when a group cannot be completed (all threads stop)
    atomic_int done; // Set when a thread of the group has explored its tree
    _Atomic(DAQPWorkspace*) winner; // The workspace of the first thread of the group that explored its tree
    atomic_int n_ended; // Threads of the group that ended without completing it
    int n_threads;
    _Atomic c_float J_group; // The bound J_bound of the group if it is not shared between the groups
}DAQPBnBShared;
#endif

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif //ifndef DAQP_BNB_PARALLEL_H
