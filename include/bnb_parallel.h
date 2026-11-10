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
 * Reusable solver (declared in api.h):
 *   DAQPParallel* daqp_parallel_setup(DAQPWorkspace** works, const int n);
 *   int daqp_parallel_solve(DAQPParallel* par, DAQPResult* res);
 *   void daqp_parallel_free(DAQPParallel* par);
 * daqp_parallel_setup checks the workspaces, allocates the results of the threads and starts n-1
 * worker threads, which wait on a condition variable between the solves; the calling thread of
 * daqp_parallel_solve solves the first workspace. It returns NULL if n < 1, if the workspaces are
 * incompatible (see below) or if an allocation fails. DAQPParallel is opaque; it is allocated by the
 * library so that its layout (threads, synchronization) is not part of the interface.
 *
 * Between the solves, the workspaces may be updated (daqp_update, settings), also in their
 * dimensions. The compatibility checks are repeated by every solve, at a cost of O(n^2) comparisons
 * of pointers and O(n) comparisons of settings and dimensions (the problem data is not compared, except
 * between groups, see below). The workspaces must remain allocated until daqp_parallel_free, which does
 * not access them. A DAQPParallel must not be solved by two threads at the same time.
 *
 * daqp_parallel_solve: the solution in res is that of the best incumbent. res->iter and res->nodes are
 * the totals of all threads, and res->solve_time is the elapsed (wall-clock) time. Returns the exit
 * flag (also in res->exitflag): DAQP_EXIT_UNSUPPORTED if the workspaces share settings, differ in their
 * dimensions or state (equality reduction, regularization), or differ in other settings, or if an
 * allocation fails. If the problem has no binary constraints, the first workspace is solved with
 * daqp_solve. Without DAQP_PARALLEL (no threads or no C11 atomics, see CMakeLists.txt), the workspaces
 * are checked in the same way and the first one is solved with daqp_solve.
 *
 * Polling (off by default):
 *   void daqp_parallel_set_spin(DAQPParallel* par, const int spin);
 * Between the solves, each worker thread polls for the start of the next solve spin times before it
 * waits on the condition variable, and the calling thread polls for the end of the solve as often
 * before it waits. This reduces the latency of the start and of the end of a solve at the cost of
 * processor time while the threads poll. Without DAQP_PARALLEL it has no effect.
 *
 * One-shot solve (setup, solve and free, declared in api.h):
 *   int daqp_solve_parallel(DAQPResult* res, DAQPWorkspace** works, const int n);
 * Returns DAQP_EXIT_UNSUPPORTED also if n < 1.
 *
 * Groups (a split of the search space):
 *   DAQPParallel* daqp_parallel_setup_groups(DAQPWorkspace** works, const int* groups, const int n);
 *   int daqp_solve_parallel_groups(DAQPResult* res, DAQPWorkspace** works, const int* groups, const int n);
 * Workspace i belongs to the group groups[i] (the groups are numbered 0, 1, ..., n_groups-1, each with
 * at least one workspace; groups == NULL is one group, as daqp_parallel_setup). The groups are copied by
 * the setup. The workspaces of a group hold the same problem, as above. The problems of different groups
 * are restrictions of one problem P: they have the same dimensions, H, f, A, soft and binary constraints
 * and settings (except branching, time_limit, iter_limit and the settings that only change the order
 * of the search), and differ in the bounds, e.g., binaries fixed to different values. The caller is
 * responsible that the feasible set of each group is a subset of that of P and that their union is the
 * feasible set of P; only the data listed above is compared, between the first workspaces of the
 * groups. H and A are compared by the setup only (O(n^2 + mn) per group): an update of H or A must be
 * made in all workspaces before the next solve. f and the soft and binary constraints (which updates of
 * f and of the senses may change) are compared by every solve (O(n + m) per group). The bounds may
 * differ between the groups and change between the solves.
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
 * (otherwise the setup of more than one group returns NULL, and a solve returns DAQP_EXIT_UNSUPPORTED).
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
