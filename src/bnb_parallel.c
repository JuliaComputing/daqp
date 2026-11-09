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
}DAQPThreadData;

static void* daqp_solve_thread(void* arg){
    DAQPThreadData* data = (DAQPThreadData*)arg;
    daqp_solve(&data->res, data->work);
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

int daqp_solve_parallel(DAQPResult* res, DAQPWorkspace** works, const int n){
    int i, j, n_x, n_lam, best = -1, ok = 1;
    DAQPThreadData* data;
    pthread_t* threads;
    int* started;
    DAQPBnBShared shared;
    DAQPWorkspace* winner;
#ifdef PROFILING
    DAQPtimer timer;
#endif

    if(n <= 0 || works == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    if(n == 1 || works[0]->bnb == NULL){
        daqp_solve(res, works[0]);
        return res->exitflag;
    }
    for(i = 0; i < n; i++){
        if(works[i]->bnb == NULL || works[i]->settings == NULL || works[0]->settings == NULL ||
                !daqp_parallel_compatible(works[i], works[0])) ok = 0;
        for(j = 0; j < i; j++) if(works[i]->settings == works[j]->settings) ok = 0;
    }
    if(!ok){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }

    daqp_problem_dims(works[0], &n_x, &n_lam);
    data = calloc(n, sizeof(DAQPThreadData));
    threads = calloc(n, sizeof(pthread_t));
    started = calloc(n, sizeof(int));
    for(i = 0; data != NULL && i < n; i++){
        data[i].work = works[i];
        data[i].res.x = malloc(n_x*sizeof(c_float));
        data[i].res.lam = res->lam != NULL ? malloc(n_lam*sizeof(c_float)) : NULL;
        if(data[i].res.x == NULL || (res->lam != NULL && data[i].res.lam == NULL)) ok = 0;
    }
    if(data == NULL || threads == NULL || started == NULL || !ok){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        goto cleanup;
    }

    atomic_init(&shared.J_bound, DAQP_INF);
    atomic_init(&shared.done, 0);
    atomic_init(&shared.winner, NULL);
    for(i = 0; i < n; i++) works[i]->bnb->shared = &shared;

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
    winner = atomic_load(&shared.winner);
    for(j = 0; winner != NULL && j < n; j++) if(works[j] == winner) break;

    i = best >= 0 ? best : (winner != NULL ? j : 0);
    for(int k = 0; k < n_x; k++) res->x[k] = data[i].res.x[k];
    for(int k = 0; res->lam != NULL && k < n_lam; k++) res->lam[k] = data[i].res.lam[k];
    res->fval = data[i].res.fval;
    res->soft_slack = data[i].res.soft_slack;
    res->exitflag = data[i].res.exitflag;
    if(winner != NULL && best >= 0){
        // The search of the winner proves that the best incumbent is optimal. A singular Hessian
        // (daqp_bnb_prox) can make it inexact, which only the thread that found it can tell when
        // it has explored its own tree, otherwise it is reported as inexact.
        if(works[best]->n_prox > 0)
            res->exitflag = data[best].res.exitflag == DAQP_EXIT_OPTIMAL ?
                DAQP_EXIT_OPTIMAL : DAQP_EXIT_OPTIMAL_INEXACT;
        else
            res->exitflag = data[best].res.exitflag == DAQP_EXIT_OPTIMAL_INEXACT ?
                DAQP_EXIT_OPTIMAL_INEXACT : DAQP_EXIT_OPTIMAL;
    }
    else if(winner != NULL) res->exitflag = data[j].res.exitflag; // No incumbent: infeasible

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
    return res->exitflag;
}

#else // DAQP_PARALLEL

int daqp_solve_parallel(DAQPResult* res, DAQPWorkspace** works, const int n){
    if(n <= 0 || works == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    daqp_solve(res, works[0]);
    return res->exitflag;
}

#endif // DAQP_PARALLEL
