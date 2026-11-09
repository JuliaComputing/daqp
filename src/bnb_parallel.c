#include "bnb_parallel.h"
#include "api.h"
#include <stdlib.h>

#ifdef DAQP_PARALLEL
#include <pthread.h>
#ifdef PROFILING
#include "utils.h"
#endif

typedef struct{
    DAQPResult res;
    DAQPWorkspace* work;
    DAQPBnBShared* group;
}DAQPThreadData;

// A search that ends with one of these exit flags has explored its tree (or proved the problem of
// the workspace infeasible before the tree search) and completes its group. A search that is stopped,
// or ends at a limit, ends with another exit flag (DAQP_EXIT_FEASIBLE or a negative one)
static int daqp_parallel_certified(const int exitflag){
    return exitflag == DAQP_EXIT_OPTIMAL || exitflag == DAQP_EXIT_OPTIMAL_INEXACT ||
        exitflag == DAQP_EXIT_INFEASIBLE;
}

static void* daqp_solve_thread(void* arg){
    DAQPThreadData* data = (DAQPThreadData*)arg;
    DAQPBnBShared* g = data->group;
    DAQPWorkspace* none = NULL;
    daqp_solve(&data->res, data->work);
    if(daqp_parallel_certified(data->res.exitflag)){
        // Also a proof before the tree search (which does not call daqp_bnb_finish)
        atomic_compare_exchange_strong(&g->winner, &none, data->work);
        atomic_store(&g->done, 1);
    }
    else if(atomic_fetch_add(&g->n_ended, 1)+1 == g->n_threads)
        atomic_store(g->stop, 1); // No thread of the group can complete it
    return NULL;
}

// Number of primal variables and constraints of the problem of work (before an equality elimination)
static void daqp_problem_dims(DAQPWorkspace* work, int* n, int* m){
    *n = work->n; *m = work->m;
    if(work->eq != NULL){
        if(work->eq->n > *n) *n = work->eq->n;
        if(work->eq->m > *m) *m = work->eq->m;
    }
    if(work->qp != NULL){
        if(work->qp->n > *n) *n = work->qp->n;
        if(work->qp->m > *m) *m = work->qp->m;
    }
}

// Whether the threads of a and b search the same problem, so that the search of either is a certificate
// for the incumbents of both: the settings may only differ in the selection of the nodes and in the
// limits of the effort, which only end a search early (it is then not a certificate). A new setting that
// changes the problem or its tolerances must be compared here
static int daqp_parallel_compatible(DAQPWorkspace* a, DAQPWorkspace* b){
    const DAQPSettings *s = a->settings, *t = b->settings;
    return a->n == b->n && a->m == b->m && a->ms == b->ms && a->n_prox == b->n_prox &&
        DAQP_IS_REDUCED(a) == DAQP_IS_REDUCED(b) &&
        s->primal_tol == t->primal_tol && s->dual_tol == t->dual_tol && s->zero_tol == t->zero_tol &&
        s->pivot_tol == t->pivot_tol && s->progress_tol == t->progress_tol &&
        s->cycle_tol == t->cycle_tol && s->fval_bound == t->fval_bound &&
        s->eps_prox == t->eps_prox && s->eta_prox == t->eta_prox && s->rho_soft == t->rho_soft &&
        s->rel_subopt == t->rel_subopt && s->abs_subopt == t->abs_subopt &&
        s->sing_tol == t->sing_tol && s->refactor_tol == t->refactor_tol && s->w_soft == t->w_soft &&
        s->eq_reduction == t->eq_reduction && s->presolve == t->presolve;
}

// The problem given to the setup of work (not the reduced one of an equality elimination)
static const DAQPProblem* daqp_parallel_problem(DAQPWorkspace* work){
    if(work->eq != NULL && work->eq->installed) return work->eq->other.qp;
    return work->qp;
}

static int daqp_same_data(const c_float* a, const c_float* b, const size_t len){
    size_t i;
    if(a == b) return 1;
    if(a == NULL || b == NULL) return 0;
    for(i = 0; i < len; i++) if(a[i] != b[i]) return 0;
    return 1;
}

// Whether the problems of a and b are restrictions of one problem: the same objective, constraint
// matrix, soft and binary constraints (the bounds may differ). The workspaces are compatible
static int daqp_parallel_restrictions(DAQPWorkspace* a, DAQPWorkspace* b){
    int i;
    const DAQPProblem *p = daqp_parallel_problem(a), *q = daqp_parallel_problem(b);
    size_t nH;
    if(p == NULL || q == NULL) return 0;
    if(p == q) return 1;
    if(p->n != q->n || p->m != q->m || p->ms != q->ms || p->problem_type != q->problem_type ||
            p->nh != q->nh) return 0;
    nH = p->problem_type == 2 ? (size_t)p->n*(p->n+1)/2 : (size_t)p->n*p->n;
    if(!daqp_same_data(p->H, q->H, nH) || !daqp_same_data(p->f, q->f, p->n) ||
            !daqp_same_data(p->A, q->A, (size_t)(p->m-p->ms)*p->n)) return 0;
    if((p->sense == NULL) != (q->sense == NULL)) return 0;
    for(i = 0; p->sense != NULL && i < p->m; i++)
        if((p->sense[i]^q->sense[i]) & (DAQP_SOFT+DAQP_BINARY)) return 0;
    return 1;
}

int daqp_solve_parallel(DAQPResult* res, DAQPWorkspace** works, const int n){
    return daqp_solve_parallel_groups(res, works, NULL, n);
}

int daqp_solve_parallel_groups(DAQPResult* res, DAQPWorkspace** works, const int* groups, const int n){
    int i, j, g, n_x, n_lam, n_groups = 1, best = -1, complete, failed = -1, ok = 1;
    DAQPThreadData* data;
    DAQPBnBShared* shared = NULL;
    pthread_t* threads;
    int* started;
    int* first; // The first workspace of each group
    _Atomic c_float J_bound;
    atomic_int stop;
#ifdef PROFILING
    DAQPtimer timer;
#endif

    if(n <= 0 || works == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    for(i = 0; groups != NULL && i < n; i++){
        if(groups[i] < 0 || groups[i] >= n){
            res->exitflag = DAQP_EXIT_UNSUPPORTED;
            return res->exitflag;
        }
        if(groups[i]+1 > n_groups) n_groups = groups[i]+1;
    }
    if(n_groups == 1 && (n == 1 || works[0]->bnb == NULL)){
        daqp_solve(res, works[0]);
        return res->exitflag;
    }

    first = malloc(n_groups*sizeof(int));
    if(first == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    for(g = 0; g < n_groups; g++) first[g] = -1;
    for(i = 0; i < n; i++){
        g = groups != NULL ? groups[i] : 0;
        if(first[g] < 0) first[g] = i;
    }
    for(g = 0; g < n_groups; g++) if(first[g] < 0) ok = 0; // An empty group
    for(i = 0; ok && i < n; i++){
        g = groups != NULL ? groups[i] : 0;
        if(works[i]->bnb == NULL || works[i]->settings == NULL || works[first[g]]->settings == NULL ||
                !daqp_parallel_compatible(works[i], works[first[g]])) ok = 0;
        for(j = 0; j < i; j++) if(works[i]->settings == works[j]->settings) ok = 0;
    }
    for(g = 1; ok && g < n_groups; g++)
        if(!daqp_parallel_compatible(works[first[g]], works[first[0]]) ||
                !daqp_parallel_restrictions(works[first[g]], works[first[0]])) ok = 0;
    if(!ok){
        free(first);
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }

    daqp_problem_dims(works[0], &n_x, &n_lam);
    data = calloc(n, sizeof(DAQPThreadData));
    threads = calloc(n, sizeof(pthread_t));
    started = calloc(n, sizeof(int));
    shared = calloc(n_groups, sizeof(DAQPBnBShared));
    for(i = 0; data != NULL && i < n; i++){
        data[i].work = works[i];
        data[i].res.x = malloc(n_x*sizeof(c_float));
        data[i].res.lam = res->lam != NULL ? malloc(n_lam*sizeof(c_float)) : NULL;
        if(data[i].res.x == NULL || (res->lam != NULL && data[i].res.lam == NULL)) ok = 0;
    }
    if(data == NULL || threads == NULL || started == NULL || shared == NULL || !ok){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        goto cleanup;
    }

    // With a singular Hessian, the regularization depends on the bounds of the group (see
    // daqp_bnb_prox), so the bound of an incumbent is only valid in its own group
    atomic_init(&J_bound, DAQP_INF);
    atomic_init(&stop, 0);
    for(g = 0; g < n_groups; g++){
        atomic_init(&shared[g].J_group, DAQP_INF);
        shared[g].J_bound = (works[0]->n_prox > 0 && n_groups > 1) ? &shared[g].J_group : &J_bound;
        shared[g].stop = &stop;
        atomic_init(&shared[g].done, 0);
        atomic_init(&shared[g].winner, NULL);
        atomic_init(&shared[g].n_ended, 0);
        shared[g].n_threads = 0;
    }
    for(i = 0; i < n; i++){
        g = groups != NULL ? groups[i] : 0;
        data[i].group = &shared[g];
        shared[g].n_threads++;
        works[i]->bnb->shared = &shared[g];
    }

#ifdef PROFILING
    tic(&timer);
#endif
    // The first workspace is solved by the calling thread
    for(i = 1; i < n; i++)
        started[i] = pthread_create(&threads[i], NULL, daqp_solve_thread, &data[i]) == 0;
    daqp_solve_thread(&data[0]);
    for(i = 1; i < n; i++){
        if(started[i]) pthread_join(threads[i], NULL);
        else daqp_solve_thread(&data[i]); // Thread could not be created
    }
#ifdef PROFILING
    toc(&timer);
#endif
    for(i = 0; i < n; i++) works[i]->bnb->shared = NULL;

    // The best incumbent of all threads (the thread that found it may have been stopped)
    for(i = 0; i < n; i++)
        if(data[i].res.exitflag > 0 && (best < 0 || data[i].res.fval < data[best].res.fval)) best = i;
    complete = 1;
    for(g = 0; g < n_groups; g++){
        if(atomic_load(&shared[g].winner) != NULL) continue;
        complete = 0;
        // The exit flag of a thread of a group that could not be completed
        if(failed < 0 || atomic_load(&shared[g].n_ended) == shared[g].n_threads)
            for(i = 0; i < n; i++) if(data[i].group == &shared[g] && data[i].res.exitflag < 0){ failed = i; break; }
    }

    i = best >= 0 ? best : (failed >= 0 ? failed : 0);
    for(int k = 0; k < n_x; k++) res->x[k] = data[i].res.x[k];
    for(int k = 0; res->lam != NULL && k < n_lam; k++) res->lam[k] = data[i].res.lam[k];
    res->fval = data[i].res.fval;
    res->soft_slack = data[i].res.soft_slack;
    res->exitflag = data[i].res.exitflag;
    if(!complete){
        if(best >= 0) res->exitflag = DAQP_EXIT_FEASIBLE;
        else if(res->exitflag >= 0) res->exitflag = DAQP_EXIT_TIMELIMIT;
    }
    else if(best < 0) res->exitflag = DAQP_EXIT_INFEASIBLE;
    else if(works[0]->n_prox > 0){
        // A singular Hessian (daqp_bnb_prox) can make an incumbent inexact, which only the thread that
        // found it can tell when it has explored its own tree. The result is exact if the best incumbent
        // of every group is (with the bound shared by all threads of a single group, that of the best)
        res->exitflag = DAQP_EXIT_OPTIMAL;
        for(g = 0; g < n_groups; g++){
            int best_g = -1;
            if(n_groups == 1) best_g = best;
            else for(j = 0; j < n; j++)
                if(data[j].group == &shared[g] && data[j].res.exitflag > 0 &&
                        (best_g < 0 || data[j].res.fval < data[best_g].res.fval)) best_g = j;
            if(best_g >= 0 && data[best_g].res.exitflag != DAQP_EXIT_OPTIMAL)
                res->exitflag = DAQP_EXIT_OPTIMAL_INEXACT;
        }
    }
    else
        res->exitflag = data[best].res.exitflag == DAQP_EXIT_OPTIMAL_INEXACT ?
            DAQP_EXIT_OPTIMAL_INEXACT : DAQP_EXIT_OPTIMAL;

    res->iter = 0; res->nodes = 0;
    for(i = 0; i < n; i++){
        res->iter += data[i].res.iter;
        res->nodes += data[i].res.nodes;
    }
#ifdef PROFILING
    res->solve_time = get_time(&timer);
#else
    res->solve_time = 0;
#endif

cleanup:
    for(i = 0; data != NULL && i < n; i++){
        free(data[i].res.x);
        free(data[i].res.lam);
    }
    free(data);
    free(threads);
    free(started);
    free(shared);
    free(first);
    return res->exitflag;
}

#else // DAQP_PARALLEL

int daqp_solve_parallel(DAQPResult* res, DAQPWorkspace** works, const int n){
    return daqp_solve_parallel_groups(res, works, NULL, n);
}

// The groups search different problems, which a single workspace cannot replace
int daqp_solve_parallel_groups(DAQPResult* res, DAQPWorkspace** works, const int* groups, const int n){
    int i;
    if(n <= 0 || works == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    for(i = 0; groups != NULL && i < n; i++)
        if(groups[i] != 0){
            res->exitflag = DAQP_EXIT_UNSUPPORTED;
            return res->exitflag;
        }
    daqp_solve(res, works[0]);
    return res->exitflag;
}

#endif // DAQP_PARALLEL
