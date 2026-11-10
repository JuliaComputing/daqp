#include "bnb_parallel.h"
#include "api.h"
#include <stdlib.h>

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

// Whether the workspaces can be solved in parallel: they have binary constraints and settings of their
// own, and are compatible
static int daqp_parallel_check(DAQPWorkspace** works, const int n){
    int i, j;
    if(works[0]->settings == NULL) return 0;
    for(i = 0; i < n; i++){
        if(works[i] == NULL || works[i]->bnb == NULL || works[i]->settings == NULL ||
                !daqp_parallel_compatible(works[i], works[0])) return 0;
        for(j = 0; j < i; j++) if(works[i]->settings == works[j]->settings) return 0;
    }
    return 1;
}

#ifdef DAQP_PARALLEL
#include <pthread.h>
#ifdef PROFILING
#include "utils.h"
#endif

typedef struct{
    DAQPResult res;
    c_float* lam; // Allocation of res.lam (res.lam is NULL if the caller requests no dual solution)
    DAQPWorkspace* work;
    DAQPParallel* par;
}DAQPThreadData;

struct DAQPParallel{
    int n;
    DAQPWorkspace** works;
    DAQPThreadData* data;
    int n_x, n_lam; // Allocated lengths of the solutions of the threads
    DAQPBnBShared shared;

    // Worker threads of the workspaces 1, ..., n-1 (workspace 0 is solved by the calling thread)
    pthread_t* threads;
    int* started; // Whether the thread of the workspace was created (otherwise solved by the caller)
    int n_started;
    pthread_mutex_t lock;
    pthread_cond_t start; // Signals a new solve (generation) or the end (quit) to the workers
    pthread_cond_t done; // Signals that n_running has reached zero
    // Written under lock; read without it while a thread polls them (see daqp_parallel_set_spin)
    atomic_ulong generation; // Number of solves started
    atomic_int n_running; // Workers that have not finished the current solve
    atomic_int quit;
    atomic_int spin; // Number of polls before a thread waits on a condition variable
};

static void* daqp_parallel_worker(void* arg){
    DAQPThreadData* data = (DAQPThreadData*)arg;
    DAQPParallel* par = data->par;
    unsigned long generation = 0;
    int k, spin;
    for(;;){
        spin = atomic_load_explicit(&par->spin, memory_order_relaxed);
        for(k = 0; k < spin && atomic_load(&par->generation) == generation && !atomic_load(&par->quit); k++);
        pthread_mutex_lock(&par->lock);
        while(atomic_load(&par->generation) == generation && !atomic_load(&par->quit))
            pthread_cond_wait(&par->start, &par->lock);
        if(atomic_load(&par->quit)){
            pthread_mutex_unlock(&par->lock);
            return NULL;
        }
        generation = atomic_load(&par->generation);
        pthread_mutex_unlock(&par->lock);

        daqp_solve(&data->res, data->work);

        pthread_mutex_lock(&par->lock);
        if(atomic_fetch_sub(&par->n_running, 1) == 1) pthread_cond_signal(&par->done);
        pthread_mutex_unlock(&par->lock);
    }
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

// Allocate the solutions of the threads for the dimensions of the problem (if they have grown)
static int daqp_parallel_alloc(DAQPParallel* par){
    int i, n_x, n_lam;
    c_float* ptr;
    daqp_problem_dims(par->works[0], &n_x, &n_lam);
    if(par->data[0].res.x != NULL && n_x <= par->n_x && n_lam <= par->n_lam) return 1;
    if(n_x < par->n_x) n_x = par->n_x;
    if(n_lam < par->n_lam) n_lam = par->n_lam;
    for(i = 0; i < par->n; i++){
        ptr = realloc(par->data[i].res.x, (n_x > 0 ? n_x : 1)*sizeof(c_float));
        if(ptr == NULL) return 0;
        par->data[i].res.x = ptr;
        ptr = realloc(par->data[i].lam, (n_lam > 0 ? n_lam : 1)*sizeof(c_float));
        if(ptr == NULL) return 0;
        par->data[i].lam = ptr;
    }
    par->n_x = n_x; par->n_lam = n_lam;
    return 1;
}

static void daqp_parallel_dealloc(DAQPParallel* par){
    int i;
    for(i = 0; par->data != NULL && i < par->n; i++){
        free(par->data[i].res.x);
        free(par->data[i].lam);
    }
    free(par->works); free(par->data); free(par->threads); free(par->started); free(par);
}

DAQPParallel* daqp_parallel_setup(DAQPWorkspace** works, const int n){
    int i;
    DAQPParallel* par;
    if(n <= 0 || works == NULL || works[0] == NULL) return NULL;
    if(n > 1 && works[0]->bnb != NULL && !daqp_parallel_check(works, n)) return NULL;

    par = calloc(1, sizeof(DAQPParallel));
    if(par == NULL) return NULL;
    par->n = n;
    par->works = malloc(n*sizeof(DAQPWorkspace*));
    par->data = calloc(n, sizeof(DAQPThreadData));
    par->threads = calloc(n, sizeof(pthread_t));
    par->started = calloc(n, sizeof(int));
    if(par->works == NULL || par->data == NULL || par->threads == NULL || par->started == NULL){
        daqp_parallel_dealloc(par);
        return NULL;
    }
    for(i = 0; i < n; i++){
        par->works[i] = works[i];
        par->data[i].work = works[i];
        par->data[i].par = par;
    }
    if(!daqp_parallel_alloc(par) || pthread_mutex_init(&par->lock, NULL) != 0){
        daqp_parallel_dealloc(par);
        return NULL;
    }
    if(pthread_cond_init(&par->start, NULL) != 0){
        pthread_mutex_destroy(&par->lock);
        daqp_parallel_dealloc(par);
        return NULL;
    }
    if(pthread_cond_init(&par->done, NULL) != 0){
        pthread_cond_destroy(&par->start);
        pthread_mutex_destroy(&par->lock);
        daqp_parallel_dealloc(par);
        return NULL;
    }
    atomic_init(&par->shared.J_bound, DAQP_INF);
    atomic_init(&par->shared.done, 0);
    atomic_init(&par->shared.winner, NULL);
    atomic_init(&par->generation, 0);
    atomic_init(&par->n_running, 0);
    atomic_init(&par->quit, 0);
    atomic_init(&par->spin, 0);
    // A workspace whose thread cannot be created is solved by the calling thread
    for(i = 1; i < n; i++){
        par->started[i] = pthread_create(&par->threads[i], NULL, daqp_parallel_worker, &par->data[i]) == 0;
        par->n_started += par->started[i];
    }
    return par;
}

void daqp_parallel_free(DAQPParallel* par){
    int i;
    if(par == NULL) return;
    pthread_mutex_lock(&par->lock);
    atomic_store(&par->quit, 1);
    pthread_cond_broadcast(&par->start);
    pthread_mutex_unlock(&par->lock);
    for(i = 1; i < par->n; i++) if(par->started[i]) pthread_join(par->threads[i], NULL);
    pthread_cond_destroy(&par->done);
    pthread_cond_destroy(&par->start);
    pthread_mutex_destroy(&par->lock);
    daqp_parallel_dealloc(par);
}

void daqp_parallel_set_spin(DAQPParallel* par, const int spin){
    if(par != NULL) atomic_store_explicit(&par->spin, spin > 0 ? spin : 0, memory_order_relaxed);
}

int daqp_parallel_solve(DAQPParallel* par, DAQPResult* res){
    int i, j, k, n, n_x, n_lam, spin, best = -1;
    DAQPThreadData* data;
    DAQPWorkspace** works;
    DAQPWorkspace* winner;
#ifdef PROFILING
    DAQPtimer timer;
#endif
    if(par == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    n = par->n; works = par->works; data = par->data;
    if(n == 1 || works[0]->bnb == NULL){
        daqp_solve(res, works[0]);
        return res->exitflag;
    }
    // The workspaces may have been updated since the setup
    if(!daqp_parallel_check(works, n) || !daqp_parallel_alloc(par)){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    daqp_problem_dims(works[0], &n_x, &n_lam);
    for(i = 0; i < n; i++) data[i].res.lam = res->lam != NULL ? data[i].lam : NULL;

    // The workers are waiting (the previous solve has ended), so the shared state can be reset
    atomic_store(&par->shared.J_bound, DAQP_INF);
    atomic_store(&par->shared.done, 0);
    atomic_store(&par->shared.winner, NULL);
    for(i = 0; i < n; i++) works[i]->bnb->shared = &par->shared;

#ifdef PROFILING
    tic(&timer);
#endif
    pthread_mutex_lock(&par->lock);
    atomic_store(&par->n_running, par->n_started);
    atomic_fetch_add(&par->generation, 1);
    pthread_cond_broadcast(&par->start);
    pthread_mutex_unlock(&par->lock);
    daqp_solve(&data[0].res, works[0]);
    for(i = 1; i < n; i++) if(!par->started[i]) daqp_solve(&data[i].res, works[i]);
    spin = atomic_load_explicit(&par->spin, memory_order_relaxed);
    for(k = 0; k < spin && atomic_load(&par->n_running) > 0; k++);
    pthread_mutex_lock(&par->lock);
    while(atomic_load(&par->n_running) > 0) pthread_cond_wait(&par->done, &par->lock);
    pthread_mutex_unlock(&par->lock);
#ifdef PROFILING
    toc(&timer);
#endif
    for(i = 0; i < n; i++) works[i]->bnb->shared = NULL;

    // The best incumbent of all threads (the thread that found it may have been stopped)
    for(i = 0; i < n; i++)
        if(data[i].res.exitflag > 0 && (best < 0 || data[i].res.fval < data[best].res.fval)) best = i;
    winner = atomic_load(&par->shared.winner);
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
    return res->exitflag;
}

#else // DAQP_PARALLEL

// The workspaces are checked as in the parallel build, and the first one is solved
struct DAQPParallel{
    int n;
    DAQPWorkspace** works;
};

DAQPParallel* daqp_parallel_setup(DAQPWorkspace** works, const int n){
    int i;
    DAQPParallel* par;
    if(n <= 0 || works == NULL || works[0] == NULL) return NULL;
    if(n > 1 && works[0]->bnb != NULL && !daqp_parallel_check(works, n)) return NULL;
    par = malloc(sizeof(DAQPParallel));
    if(par == NULL) return NULL;
    par->works = malloc(n*sizeof(DAQPWorkspace*));
    if(par->works == NULL){
        free(par);
        return NULL;
    }
    par->n = n;
    for(i = 0; i < n; i++) par->works[i] = works[i];
    return par;
}

int daqp_parallel_solve(DAQPParallel* par, DAQPResult* res){
    if(par == NULL || (par->n > 1 && par->works[0]->bnb != NULL && !daqp_parallel_check(par->works, par->n))){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    daqp_solve(res, par->works[0]);
    return res->exitflag;
}

void daqp_parallel_free(DAQPParallel* par){
    if(par != NULL) free(par->works);
    free(par);
}

void daqp_parallel_set_spin(DAQPParallel* par, const int spin){
    (void)par; (void)spin;
}

#endif // DAQP_PARALLEL

int daqp_solve_parallel(DAQPResult* res, DAQPWorkspace** works, const int n){
    DAQPParallel* par;
    if(n <= 0 || works == NULL || works[0] == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    if(n == 1 || works[0]->bnb == NULL){
        daqp_solve(res, works[0]);
        return res->exitflag;
    }
    par = daqp_parallel_setup(works, n);
    if(par == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    daqp_parallel_solve(par, res);
    daqp_parallel_free(par);
    return res->exitflag;
}
