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

// Whether the problems of a and b have the same dimensions, H and A (compared by the setup)
static int daqp_parallel_same_matrices(DAQPWorkspace* a, DAQPWorkspace* b){
    const DAQPProblem *p = daqp_parallel_problem(a), *q = daqp_parallel_problem(b);
    size_t nH;
    if(p == NULL || q == NULL) return 0;
    if(p == q) return 1;
    if(p->n != q->n || p->m != q->m || p->ms != q->ms || p->problem_type != q->problem_type ||
            p->nh != q->nh) return 0;
    nH = p->problem_type == 2 ? (size_t)p->n*(p->n+1)/2 : (size_t)p->n*p->n;
    return daqp_same_data(p->H, q->H, nH) && daqp_same_data(p->A, q->A, (size_t)(p->m-p->ms)*p->n);
}

// Whether the problems of a and b have the same f and the same soft and binary constraints (compared
// by every solve, since updates of f and of the senses may change them)
static int daqp_parallel_same_objective(DAQPWorkspace* a, DAQPWorkspace* b){
    int i;
    const DAQPProblem *p = daqp_parallel_problem(a), *q = daqp_parallel_problem(b);
    if(p == NULL || q == NULL) return 0;
    if(p == q) return 1;
    if(p->n != q->n || p->m != q->m || !daqp_same_data(p->f, q->f, p->n)) return 0;
    if((p->sense == NULL) != (q->sense == NULL)) return 0;
    for(i = 0; p->sense != NULL && i < p->m; i++)
        if((p->sense[i]^q->sense[i]) & (DAQP_SOFT+DAQP_BINARY)) return 0;
    return 1;
}

// The number of groups and the first workspace of each group (first has n entries). Returns 0 if a
// group is out of range or empty
static int daqp_parallel_groups(const int* groups, const int n, int* n_groups, int* first){
    int i, g;
    *n_groups = 1;
    for(i = 0; groups != NULL && i < n; i++){
        if(groups[i] < 0 || groups[i] >= n) return 0;
        if(groups[i]+1 > *n_groups) *n_groups = groups[i]+1;
    }
    for(g = 0; g < *n_groups; g++) first[g] = -1;
    for(i = 0; i < n; i++){
        g = groups != NULL ? groups[i] : 0;
        if(first[g] < 0) first[g] = i;
    }
    for(g = 0; g < *n_groups; g++) if(first[g] < 0) return 0;
    return 1;
}

// Whether the workspaces can be solved in parallel: they have binary constraints and settings of their
// own, the workspaces of a group are compatible with its first one, and the first workspaces of the
// groups are compatible and hold restrictions of one problem (H and A are compared if setup is set)
static int daqp_parallel_check(DAQPWorkspace** works, const int n, const int* groups,
        const int* first, const int n_groups, const int setup){
    int i, j, g;
    for(i = 0; i < n; i++){
        if(works[i] == NULL || works[i]->bnb == NULL || works[i]->settings == NULL) return 0;
        for(j = 0; j < i; j++) if(works[i]->settings == works[j]->settings) return 0;
    }
    for(i = 0; i < n; i++){
        g = groups != NULL ? groups[i] : 0;
        if(!daqp_parallel_compatible(works[i], works[first[g]])) return 0;
    }
    for(g = 1; g < n_groups; g++)
        if(!daqp_parallel_compatible(works[first[g]], works[first[0]]) ||
                !daqp_parallel_same_objective(works[first[g]], works[first[0]]) ||
                (setup && !daqp_parallel_same_matrices(works[first[g]], works[first[0]]))) return 0;
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
    DAQPBnBShared* group;
    DAQPParallel* par;
}DAQPThreadData;

struct DAQPParallel{
    int n;
    DAQPWorkspace** works;
    DAQPThreadData* data;
    int n_x, n_lam; // Allocated lengths of the solutions of the threads
    int n_groups;
    int* groups; // The group of each workspace (NULL for one group)
    int* first; // The first workspace of each group
    DAQPBnBShared* shared; // The state of each group
    _Atomic c_float J_bound; // Bound on J shared by all groups (see daqp_parallel_solve)
    atomic_int stop; // Set when a group cannot be completed

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

// A search that ends with one of these exit flags has explored its tree (or proved the problem of
// the workspace infeasible before the tree search) and completes its group. A search that is stopped,
// or ends at a limit, ends with another exit flag (DAQP_EXIT_FEASIBLE or a negative one)
static int daqp_parallel_certified(const int exitflag){
    return exitflag == DAQP_EXIT_OPTIMAL || exitflag == DAQP_EXIT_OPTIMAL_INEXACT ||
        exitflag == DAQP_EXIT_INFEASIBLE;
}

// Solve the workspace of a thread and complete its group if the search is a certificate
static void daqp_parallel_run(DAQPThreadData* data){
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
}

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

        daqp_parallel_run(data);

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
    free(par->works); free(par->data); free(par->groups); free(par->first); free(par->shared);
    free(par->threads); free(par->started); free(par);
}

DAQPParallel* daqp_parallel_setup_groups(DAQPWorkspace** works, const int* groups, const int n){
    int i, g, n_groups;
    int* first;
    DAQPParallel* par;
    if(n <= 0 || works == NULL || works[0] == NULL) return NULL;
    first = malloc(n*sizeof(int));
    if(first == NULL) return NULL;
    if(!daqp_parallel_groups(groups, n, &n_groups, first) ||
            ((n_groups > 1 || (n > 1 && works[0]->bnb != NULL)) &&
             !daqp_parallel_check(works, n, groups, first, n_groups, 1))){
        free(first);
        return NULL;
    }

    par = calloc(1, sizeof(DAQPParallel));
    if(par == NULL){
        free(first);
        return NULL;
    }
    par->n = n;
    par->n_groups = n_groups;
    par->first = first;
    par->works = malloc(n*sizeof(DAQPWorkspace*));
    par->data = calloc(n, sizeof(DAQPThreadData));
    par->groups = groups != NULL ? malloc(n*sizeof(int)) : NULL;
    par->shared = calloc(n_groups, sizeof(DAQPBnBShared));
    par->threads = calloc(n, sizeof(pthread_t));
    par->started = calloc(n, sizeof(int));
    if(par->works == NULL || par->data == NULL || (groups != NULL && par->groups == NULL) ||
            par->shared == NULL || par->threads == NULL || par->started == NULL){
        daqp_parallel_dealloc(par);
        return NULL;
    }
    for(i = 0; i < n; i++){
        g = groups != NULL ? groups[i] : 0;
        if(groups != NULL) par->groups[i] = g;
        par->works[i] = works[i];
        par->data[i].work = works[i];
        par->data[i].group = &par->shared[g];
        par->data[i].par = par;
        par->shared[g].n_threads++;
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
    atomic_init(&par->J_bound, DAQP_INF);
    atomic_init(&par->stop, 0);
    for(g = 0; g < n_groups; g++){
        atomic_init(&par->shared[g].J_group, DAQP_INF);
        par->shared[g].J_bound = &par->J_bound;
        par->shared[g].stop = &par->stop;
        atomic_init(&par->shared[g].done, 0);
        atomic_init(&par->shared[g].winner, NULL);
        atomic_init(&par->shared[g].n_ended, 0);
    }
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

DAQPParallel* daqp_parallel_setup(DAQPWorkspace** works, const int n){
    return daqp_parallel_setup_groups(works, NULL, n);
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
    int i, j, g, k, n, n_x, n_lam, n_groups, spin, best = -1, complete, failed = -1;
    DAQPThreadData* data;
    DAQPWorkspace** works;
    DAQPBnBShared* shared;
#ifdef PROFILING
    DAQPtimer timer;
#endif
    if(par == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    n = par->n; works = par->works; data = par->data; n_groups = par->n_groups; shared = par->shared;
    if(n_groups == 1 && (n == 1 || works[0]->bnb == NULL)){
        daqp_solve(res, works[0]);
        return res->exitflag;
    }
    // The workspaces may have been updated since the setup
    if(!daqp_parallel_check(works, n, par->groups, par->first, n_groups, 0) || !daqp_parallel_alloc(par)){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    daqp_problem_dims(works[0], &n_x, &n_lam);
    for(i = 0; i < n; i++) data[i].res.lam = res->lam != NULL ? data[i].lam : NULL;

    // The workers are waiting (the previous solve has ended), so the shared state can be reset. With a
    // singular Hessian, the regularization depends on the bounds of the group (see daqp_bnb_prox), so
    // the bound of an incumbent is only valid in its own group
    atomic_store(&par->J_bound, DAQP_INF);
    atomic_store(&par->stop, 0);
    for(g = 0; g < n_groups; g++){
        atomic_store(&shared[g].J_group, DAQP_INF);
        shared[g].J_bound = (works[0]->n_prox > 0 && n_groups > 1) ? &shared[g].J_group : &par->J_bound;
        atomic_store(&shared[g].done, 0);
        atomic_store(&shared[g].winner, NULL);
        atomic_store(&shared[g].n_ended, 0);
    }
    for(i = 0; i < n; i++) works[i]->bnb->shared = data[i].group;

#ifdef PROFILING
    tic(&timer);
#endif
    pthread_mutex_lock(&par->lock);
    atomic_store(&par->n_running, par->n_started);
    atomic_fetch_add(&par->generation, 1);
    pthread_cond_broadcast(&par->start);
    pthread_mutex_unlock(&par->lock);
    daqp_parallel_run(&data[0]);
    for(i = 1; i < n; i++) if(!par->started[i]) daqp_parallel_run(&data[i]);
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
    complete = 1;
    for(g = 0; g < n_groups; g++){
        if(atomic_load(&shared[g].winner) != NULL) continue;
        complete = 0;
        // The exit flag of a thread of a group that could not be completed
        if(failed < 0 || atomic_load(&shared[g].n_ended) == shared[g].n_threads)
            for(i = 0; i < n; i++) if(data[i].group == &shared[g] && data[i].res.exitflag < 0){ failed = i; break; }
    }

    i = best >= 0 ? best : (failed >= 0 ? failed : 0);
    for(k = 0; k < n_x; k++) res->x[k] = data[i].res.x[k];
    for(k = 0; res->lam != NULL && k < n_lam; k++) res->lam[k] = data[i].res.lam[k];
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
    return res->exitflag;
}

#else // DAQP_PARALLEL

// The workspaces are checked as in the parallel build, and the first one is solved. The groups search
// different problems, which a single workspace cannot replace
struct DAQPParallel{
    int n;
    DAQPWorkspace** works;
    int* first;
};

DAQPParallel* daqp_parallel_setup_groups(DAQPWorkspace** works, const int* groups, const int n){
    int i, n_groups;
    DAQPParallel* par;
    if(n <= 0 || works == NULL || works[0] == NULL) return NULL;
    par = calloc(1, sizeof(DAQPParallel));
    if(par == NULL) return NULL;
    par->works = malloc(n*sizeof(DAQPWorkspace*));
    par->first = malloc(n*sizeof(int));
    if(par->works == NULL || par->first == NULL || !daqp_parallel_groups(groups, n, &n_groups, par->first) ||
            n_groups > 1 ||
            (n > 1 && works[0]->bnb != NULL && !daqp_parallel_check(works, n, NULL, par->first, 1, 1))){
        free(par->works); free(par->first); free(par);
        return NULL;
    }
    par->n = n;
    for(i = 0; i < n; i++) par->works[i] = works[i];
    return par;
}

DAQPParallel* daqp_parallel_setup(DAQPWorkspace** works, const int n){
    return daqp_parallel_setup_groups(works, NULL, n);
}

int daqp_parallel_solve(DAQPParallel* par, DAQPResult* res){
    if(par == NULL || (par->n > 1 && par->works[0]->bnb != NULL &&
                !daqp_parallel_check(par->works, par->n, NULL, par->first, 1, 0))){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    daqp_solve(res, par->works[0]);
    return res->exitflag;
}

void daqp_parallel_free(DAQPParallel* par){
    if(par != NULL){
        free(par->works);
        free(par->first);
    }
    free(par);
}

void daqp_parallel_set_spin(DAQPParallel* par, const int spin){
    (void)par; (void)spin;
}

#endif // DAQP_PARALLEL

int daqp_solve_parallel_groups(DAQPResult* res, DAQPWorkspace** works, const int* groups, const int n){
    int i, single = 1; // One group
    DAQPParallel* par;
    if(n <= 0 || works == NULL || works[0] == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    for(i = 0; groups != NULL && i < n; i++) if(groups[i] != 0) single = 0;
    if(single && (n == 1 || works[0]->bnb == NULL)){
        daqp_solve(res, works[0]);
        return res->exitflag;
    }
    par = daqp_parallel_setup_groups(works, groups, n);
    if(par == NULL){
        res->exitflag = DAQP_EXIT_UNSUPPORTED;
        return res->exitflag;
    }
    daqp_parallel_solve(par, res);
    daqp_parallel_free(par);
    return res->exitflag;
}

int daqp_solve_parallel(DAQPResult* res, DAQPWorkspace** works, const int n){
    return daqp_solve_parallel_groups(res, works, NULL, n);
}
