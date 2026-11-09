#include "bnb.h"
#ifdef PROFILING
#include "utils.h"
#endif

#ifdef DAQP_PARALLEL
#include "bnb_parallel.h"

// The bound fval_bound (in the scale 0.5*fval = J + c, see daqp_bnb_bound) is shared between the
// threads of daqp_solve_parallel as the bound fval_bound - c on J, which does not depend on c.
// Returns 1 if another thread has explored its tree (the search is stopped)
static int daqp_bnb_sync(DAQPWorkspace* work, const c_float c){
    DAQPBnBShared* sh = (DAQPBnBShared*)work->bnb->shared;
    c_float bound;
    if(sh == NULL) return 0;
    if(atomic_load_explicit(&sh->done, memory_order_relaxed)) return 1;
    bound = atomic_load_explicit(&sh->J_bound, memory_order_relaxed) + c;
    if(bound < work->settings->fval_bound) work->settings->fval_bound = bound;
    return 0;
}

// Make the bound of a new incumbent known to the other threads
static void daqp_bnb_publish(DAQPWorkspace* work, const c_float c){
    DAQPBnBShared* sh = (DAQPBnBShared*)work->bnb->shared;
    c_float J, cur;
    if(sh == NULL) return;
    J = work->settings->fval_bound - c;
    cur = atomic_load_explicit(&sh->J_bound, memory_order_relaxed);
    while(J < cur && !atomic_compare_exchange_weak(&sh->J_bound, &cur, J));
}

// The tree of work has been explored: stop the other threads
static void daqp_bnb_finish(DAQPWorkspace* work){
    DAQPBnBShared* sh = (DAQPBnBShared*)work->bnb->shared;
    DAQPWorkspace* none = NULL;
    if(sh == NULL) return;
    atomic_compare_exchange_strong(&sh->winner, &none, work);
    atomic_store(&sh->done, 1);
}
#else
#define daqp_bnb_sync(work,c) 0
#define daqp_bnb_publish(work,c)
#define daqp_bnb_finish(work)
#endif

static c_float daqp_binary_diff(const int id, DAQPWorkspace* work){
    int j, disp;
    c_float diff = 0.5*(work->dupper[id]+work->dlower[id]);

    if(id < work->ms){//Simple bound
        if(work->Rinv==NULL) diff -= work->u[id]; //Hessian is identity
        else{
            for(j=id,disp=id+DAQP_R_OFFSET(id,work->n);j<work->n;j++)
                diff -= work->Rinv[disp++]*work->u[j];
        }
    }
    else{//General bound; daqp_add_infeasible already computed M*u
        diff -= work->Mu[id-work->ms];
    }
    return diff;
}

// Store the free part of the working set in ids. Returns the number stored.
static int daqp_bnb_store_ws(int* ids, DAQPWorkspace* work){
    int i, n_ids = 0;
    for(i=work->bnb->neq; i<work->n_active;i++){
        if((work->sense[work->WS[i]]&(DAQP_IMMUTABLE+DAQP_BINARY))!=DAQP_IMMUTABLE+DAQP_BINARY)
            ids[n_ids++] = work->WS[i]+(DAQP_IS_LOWER(work->WS[i]) << (DAQP_LOWER_BIT-1));
    }
    return n_ids;
}

// Add the constraints in ids to the working set (aborted if the basis gets singular)
static void daqp_bnb_load_ws(const int* ids, const int n_ids, DAQPWorkspace* work){
    int i;
    for(i=0; i < n_ids; i++){
        // E.g., a binary that the new bounds fix is already active (as an equality)
        if(DAQP_IS_ACTIVE(DAQP_REMOVE_LOWER_FLAG(ids[i]))) continue;
        daqp_add_upper_lower(ids[i],work);
        if(work->sing_ind != DAQP_EMPTY_IND) {
            work->n_active--;
            DAQP_SET_INACTIVE(work->WS[work->n_active]);
            work->sing_ind = DAQP_EMPTY_IND;
            break;
        }
    }
}

// Add fixed binary id; a dependent one is left out: returns 1 if added, 0 if
// consistent, and -1 if inconsistent (the node is infeasible)
static int daqp_bnb_add_fixed(const int id, DAQPWorkspace* work){
    int j;
    c_float residual = 0, term;
    // Same tolerance as for the binary in daqp_get_branch_id
    const c_float tol = work->settings->primal_tol*
        (work->scaling != NULL ? work->scaling[DAQP_REMOVE_LOWER_FLAG(id)] : 1);
    daqp_add_upper_lower(id,work);
    if(work->sing_ind == DAQP_EMPTY_IND){
        DAQP_SET_IMMUTABLE(DAQP_REMOVE_LOWER_FLAG(id));
        return 1;
    }
    daqp_compute_singular_direction(work);
    for(j = 0; j < work->n_active; j++){
        const int k = work->WS[j];
        term = work->lam_star[j]*(DAQP_IS_LOWER(k) ? work->dlower[k] : work->dupper[k]);
        residual += term;
    }
    daqp_drop_singular_last(work);
    return (residual > tol || residual < -tol) ? -1 : 0;
}

// Add the fixed binaries of node (n_clean falls back to neq if one is left
// out). Returns -1 if the node is infeasible
static int daqp_bnb_add_fixed_ids(DAQPNode* node, DAQPWorkspace* work){
    int i, flag, dropped = 0;
    for(i=work->bnb->n_clean - work->bnb->neq; i< node->depth+1;i++){
        flag = daqp_bnb_add_fixed(work->bnb->fixed_ids[i],work);
        if(flag < 0){
            work->bnb->n_clean = work->bnb->neq;
            return -1;
        }
        dropped |= flag == 0;
    }
    work->bnb->n_clean = dropped ? work->bnb->neq : work->bnb->neq+node->depth;
    return 0;
}

// The immutable constraints (e.g., equalities) are kept fixed as a prefix of
// the working set throughout the tree. Returns the length of that prefix.
static int daqp_bnb_setup_root(DAQPWorkspace* work){
    int i, j, error_flag;
    for(i = 0; i < work->n_active; i++)
        if(!DAQP_IS_IMMUTABLE(work->WS[i])) break;
    for(j = i; j < work->n_active; j++)
        if(DAQP_IS_IMMUTABLE(work->WS[j])) break;
    if(j == work->n_active) return i; // Mutable constraints are a warm start

    // Immutable after mutable => only activate immutable constraints
    for(i = 0; i < work->n_active; i++)
        if(!DAQP_IS_IMMUTABLE(work->WS[i])) DAQP_SET_INACTIVE(work->WS[i]);
    reset_daqp_workspace(work);
    error_flag = daqp_activate_constraints(work);
    return error_flag < 0 ? error_flag : work->n_active;
}

// Use the candidate in work->x as incumbent. Returns its objective
// (internal scale) with u = R*x+v stored in work->xold or -1 if infeasible.
static c_float daqp_bnb_incumbent(DAQPWorkspace* work){
    int i, j, disp;
    const int n = work->n;
    c_float val, fval = 0, tol = work->settings->primal_tol;
    c_float *x = work->x, *u = work->xold;
    DAQPProblem* qp = work->qp;
    if(qp == NULL) return -1;

    // Check feasibility (soft constraints are treated as hard)
    for(i=0, disp=0; i < work->m; i++){
        if(i < work->ms) val = x[i];
        else for(j=0, val=0; j < n; j++) val += qp->A[disp++]*x[j];
        if(DAQP_IS_IMMUTABLE(i) && !DAQP_IS_ACTIVE(i) && !DAQP_IS_BINARY(i)) continue; // Ignored
        if(val > qp->bupper[i]+tol || val < qp->blower[i]-tol) return -1;
        if(DAQP_IS_BINARY(i) && val > qp->blower[i]+tol && val < qp->bupper[i]-tol) return -1;
    }

    // Invert ldp2qp_solution: u = R*x + v
    for(i=0; i < n; i++) u[i] = x[i];
    if(work->Rinv != NULL){
        if(work->scaling != NULL)
            for(i=0; i < work->ms; i++) u[i] *= work->scaling[i];
        for(i=n-1; i >= 0; i--){ // Back substitution with upper triangular Rinv
            disp = i+DAQP_R_OFFSET(i,n);
            for(j=i+1; j < n; j++) u[i] -= work->Rinv[disp+j-i]*u[j];
            u[i] /= work->Rinv[disp];
        }
    }
    else if(work->RinvD != NULL)
        for(i=0; i < n; i++) u[i] /= work->RinvD[i];
    if(work->v != NULL)
        for(i=0; i < n; i++) u[i] += work->v[i];

    for(i=0; i < n; i++) fval += u[i]*u[i];
    return fval;
}

// The regularization sum_j reg_j/2*x_j^2 of daqp_bnb_prox at u, which is not part of
// the objective J (only needed if rel_subopt > 0 or for the lower bound lb)
static c_float daqp_bnb_reg(const c_float* u, const c_float lb, DAQPWorkspace* work){
    int i;
    c_float val = 0, *x = work->bnb->xtmp;
    if(work->n_prox == 0 || x == NULL || (work->settings->rel_subopt <= 0 && lb <= -DAQP_INF))
        return 0;
    for(i = 0; i < work->n; i++) x[i] = work->v != NULL ? u[i]-work->v[i] : u[i];
    daqp_apply_Rinv(work,x);
    for(i = 0; i < work->n; i++) val += 0.5*work->bnb->reg[i]*x[i]*x[i];
    return val;
}

// Nodes must improve on the objective J by more than abs_subopt + rel_subopt*|J|.
// The bound is in the scale 0.5*fval = J + c used by daqp_ldp. If J is within this of
// the lower bound lb, no node can, and all are pruned
static c_float daqp_bnb_bound(c_float fval, c_float c, const c_float lb, DAQPSettings* settings){
    const c_float J = 0.5*fval - c;
    const c_float tol = settings->abs_subopt + settings->rel_subopt*(J < 0 ? -J : J);
    return (lb > -DAQP_INF && J - lb <= tol) ? -DAQP_INF : 0.5*fval - tol;
}

// Fixing binaries that dominate a row of M (e.g., big-M or small curvature on the binaries)
// leaves the fraction r of its squared norm that is on the other variables. Pivots of such
// rows are hence about r, and the singularity tolerance is made relative to the smallest r
static c_float daqp_bnb_sing_tol(DAQPWorkspace* work){
    int i, k;
    c_float rmin = 1, sb, tol;
    if(work->n_prox > 0) return work->settings->sing_tol; // Not for daqp_bnb_prox
    for(i = 0; i < work->m-work->ms; i++){ // The rows of M have unit norm => r = 1-sb
        const c_float* Mi = work->M+(size_t)i*work->n;
        for(k = 0, sb = 0; k < work->bnb->nb; k++)
            if(work->bnb->bin_ids[k] < work->ms) sb += Mi[work->bnb->bin_ids[k]]*Mi[work->bnb->bin_ids[k]];
        if(sb > 0 && 1-sb > 0 && 1-sb < rmin) rmin = 1-sb;
    }
    tol = rmin*work->settings->sing_tol;
    if(tol < DAQP_BNB_MIN_SING_TOL) // Limit by the accuracy of the pivots
        tol = work->settings->sing_tol < DAQP_BNB_MIN_SING_TOL ? work->settings->sing_tol : DAQP_BNB_MIN_SING_TOL;
    return tol;
}

// Store the relaxation of the node being processed (in work->u) as the incumbent
static void daqp_bnb_new_incumbent(DAQPWorkspace* work, const c_float c, const c_float lb,
        c_float* fval_best, c_float** swp_ptr){
    *fval_best = work->fval;
    work->settings->fval_bound = daqp_bnb_bound(*fval_best,c+daqp_bnb_reg(work->u,lb,work),
            lb,work->settings);
    daqp_bnb_publish(work,c);
    *swp_ptr=work->xold; work->xold= work->u; work->u=*swp_ptr; // Store feasible sol
}

// Individual relaxations are often too short to reach the timer check
// in daqp_ldp, so also enforce the limit across the BnB tree.
static int daqp_bnb_time_exceeded(DAQPWorkspace* work){
#ifdef PROFILING
    if(work->timer != NULL && (work->bnb->nodecount&31)==0){
        toc((DAQPtimer*)work->timer);
        if(get_time((DAQPtimer*)work->timer) > work->settings->time_limit) return 1;
    }
#else
    (void)work;
#endif
    return 0;
}

static int daqp_bnb_best_first(DAQPWorkspace* work, const c_float c, const c_float lb,
        c_float* fval_best, c_float** swp_ptr, int* explored);

int daqp_bnb(DAQPWorkspace* work){
    return daqp_bnb_lb(work,-DAQP_INF);
}

int daqp_bnb_lb(DAQPWorkspace* work, const c_float lb){
    int branch_id, exitflag;
    DAQPNode* node;
    c_float *swp_ptr = NULL;

    exitflag = daqp_bnb_setup_root(work);
    if(exitflag < 0) return exitflag;
    work->bnb->neq = exitflag;
    const c_float sing_tol0 = work->settings->sing_tol;
    work->settings->sing_tol = daqp_bnb_sing_tol(work);

    // Warm start the root with the root working set of the previous solve
    // (unless a warm start has been provided)
    if(work->n_active == work->bnb->neq)
        daqp_bnb_load_ws(work->bnb->root_WS,work->bnb->n_root_WS,work);

    // Objective J = 0.5*fval - c, with c = 0.5*||v||^2 - fp, where fp is the objective
    // removed by equality elimination (only needed for rel_subopt, fval_bound and lb)
    c_float fval_bound0 = work->settings->fval_bound, fval_best = 0, c = 0;
    if(work->settings->rel_subopt > 0 || fval_bound0 < DAQP_INF || lb > -DAQP_INF ||
            work->bnb->shared != NULL){
        if(work->v != NULL) for(int i=0; i < work->n; i++) c += 0.5*work->v[i]*work->v[i];
        if(work->eq != NULL && work->eq->installed) c -= work->eq->fp;
    }
    work->settings->fval_bound = fval_bound0 + c;

    // Start from a user-provided integer-feasible solution
    if(work->state & DAQP_STATE_INCUMBENT){
        work->state &= ~DAQP_STATE_INCUMBENT;
        fval_best = daqp_bnb_incumbent(work);
        if(fval_best >= 0 && 0.5*fval_best < work->settings->fval_bound){
            work->settings->fval_bound = daqp_bnb_bound(fval_best,c+daqp_bnb_reg(work->xold,lb,work),
                    lb,work->settings);
            daqp_bnb_publish(work,c);
            swp_ptr = work->xold; // Marks that a feasible solution is stored in xold
        }
    }

    work->bnb->itercount=0;
    work->bnb->nodecount=0;
    // Setup root node
    work->bnb->tree[0].depth=-1;
    work->bnb->tree[0].WS_start=0;
    work->bnb->tree[0].WS_end=0;
    work->bnb->tree[0].bin_id=0;
    work->bnb->tree[0].lb=-DAQP_INF; // Always process the root
    work->bnb->n_nodes=1;
    work->bnb->n_clean=work->bnb->neq;
    work->bnb->nWS=0;

    exitflag = DAQP_EXIT_INFEASIBLE;
    int explored = 1; // The tree is explored (not stopped early)
    if(work->settings->node_selection == DAQP_NODE_BEST_FIRST_DIVE && work->bnb->pool != NULL)
        exitflag = daqp_bnb_best_first(work,c,lb,&fval_best,&swp_ptr,&explored);
    else while( work->bnb->n_nodes > 0 ){ // Depth-first tree exploration
        // Bound from the incumbents of other threads (see daqp_solve_parallel)
        if(daqp_bnb_sync(work,c)){
            explored = 0;
            exitflag = DAQP_EXIT_TIMELIMIT;
            break;
        }

        node = work->bnb->tree+(--work->bnb->n_nodes);
        if(node->lb > 2*work->settings->fval_bound) continue;
        exitflag = daqp_process_node(node,work); // Solve relaxation
        if(node->depth < 0 && exitflag > 0 && work->bnb->root_WS != NULL)
            work->bnb->n_root_WS = daqp_bnb_store_ws(work->bnb->root_WS,work);
        if(daqp_bnb_time_exceeded(work)){
            exitflag = DAQP_EXIT_TIMELIMIT;
            explored = 0;
            break;
        }
        // Cut conditions
        if(exitflag==DAQP_EXIT_INFEASIBLE) continue; // Dominance cut
        if(exitflag<0){ // Inner solver failed => exit loop
            explored = 0;
            break;
        }

        // Find index to branch over
        branch_id = daqp_get_branch_id(work);
        if(branch_id==DAQP_EMPTY_IND) // Nothing to branch over => integer feasible
            daqp_bnb_new_incumbent(work,c,lb,&fval_best,&swp_ptr);
        else
            daqp_spawn_children(node,branch_id, work);
    }

    // Exploration completed
    if(explored) daqp_bnb_finish(work);
    work->iterations = work->bnb->itercount;
    // Restore the root state (unfix binaries etc.) so that the workspace can
    // be reused for subsequent solves
    daqp_node_cleanup_workspace(work->bnb->neq,work);
    work->bnb->n_clean = work->bnb->neq;
    work->settings->fval_bound = fval_bound0;
    work->settings->sing_tol = sing_tol0;
    if(swp_ptr==NULL){
        return exitflag < 0 ? exitflag : DAQP_EXIT_INFEASIBLE;
    }
    else{
        work->fval = fval_best;
        // Let work->u point to the best feasible solution
        swp_ptr=work->u; work->u= work->xold; work->xold=swp_ptr;
        // Exploration ended early (time/iteration limit, cycling) => not proven optimal
        return exitflag < DAQP_EXIT_INFEASIBLE ? DAQP_EXIT_FEASIBLE : DAQP_EXIT_OPTIMAL;
    }
}

// Solve the relaxation of node, for which the working set has been set up
static int daqp_bnb_solve_relaxation(DAQPNode* node, DAQPWorkspace* work){
    int exitflag = daqp_ldp(work);
    work->bnb->itercount += work->iterations;

    if(exitflag == DAQP_EXIT_CYCLE){// Try to repair (cold start)
        // A cycle can be caused by stale cached forward-substitution data.
        // Force the retained fixed prefix to be recomputed during repair.
        work->reuse_ind=0;
        if(daqp_setup_cold_bnb(node,work) < 0) return DAQP_EXIT_INFEASIBLE;
        exitflag = daqp_ldp(work);
        work->bnb->itercount += work->iterations;
    }
    return exitflag;
}

int daqp_process_node(DAQPNode* node, DAQPWorkspace* work){
    work->bnb->nodecount+=1;
    if(node->depth >=0){
        // Fix a binary constraints
        work->bnb->fixed_ids[node->depth] = node->bin_id;
        // Setup relaxation
        if(work->bnb->n_nodes==0 || (node-1)->depth!=node->depth){
            // Sibling has been processed => need to fix workspace state
            // Retain only fixed constraints of common ancestors (robust to pruned nodes)
            if(work->bnb->n_clean > work->bnb->neq+node->depth)
                work->bnb->n_clean = work->bnb->neq+node->depth;
            daqp_node_cleanup_workspace(work->bnb->n_clean,work);
            if(daqp_warmstart_node(node,work) < 0) return DAQP_EXIT_INFEASIBLE;
        }
        else{
            daqp_add_upper_lower(node->bin_id,work);
            work->sense[DAQP_REMOVE_LOWER_FLAG(node->bin_id)] |= DAQP_IMMUTABLE; //Equality
            if(work->sing_ind != DAQP_EMPTY_IND){ // Need to cold start to not miss integer feasible
                if(daqp_setup_cold_bnb(node,work) < 0) return DAQP_EXIT_INFEASIBLE;
            }
        }
        // Add binary constraint
    }
    return daqp_bnb_solve_relaxation(node,work);
}

// Whether the binary constraint id needs branching at the node being processed: it is not fixed and the
// relaxation does not satisfy it. diff is the signed distance of the relaxation from the midpoint of the
// bounds, dist the distance to the nearest bound and tol the (scaled) primal tolerance
static int daqp_branch_candidate(const int id, DAQPNode* node, DAQPWorkspace* work,
        c_float* diff, c_float* dist, c_float* tol){
    int j;
    if(!(work->sense[id] & DAQP_BINARY)) return 0;
    // Skip fixed binary constraints (also those dropped from the working set since
    // their bounds are implied, which may still be off by rounding)
    if(DAQP_IS_ACTIVE(id)) return 0;
    for(j = 0; j <= node->depth; j++)
        if(DAQP_REMOVE_LOWER_FLAG(work->bnb->fixed_ids[j]) == id) return 0;

    // Compute signed distance from midpoint between bounds
    *diff = daqp_binary_diff(id,work);

    // A zero-dual binary constraint can lie at an endpoint without being
    // active. It is already integer feasible and does not need branching.
    *dist = 0.5*(work->dupper[id]-work->dlower[id]) -(*diff < 0 ? -*diff : *diff);
    *tol = work->settings->primal_tol;
    if(work->scaling != NULL) *tol *= work->scaling[id];
    return *dist > *tol;
}

// Branch over the binary constraint id (see daqp_branch_candidate for diff, dist and tol)
static int daqp_branch_on(const int id, const c_float diff, c_float dist, const c_float tol,
        DAQPNode* node, DAQPWorkspace* work){
    // Bounds for the children (spawned in node/node+1) from KKT at u*:
    // ||u||^2 >= ||u*||^2 + (m_i'(u-u*))^2 since ||m_i|| = 1 (unless soft)
    const c_float dist_far = DAQP_IS_SOFT(id) ? 0 : work->dupper[id]-work->dlower[id]-dist-tol;
    dist = DAQP_IS_SOFT(id) ? 0 : dist-tol;
    (node+1)->lb = work->fval + dist*dist; // Nearest endpoint
    node->lb = work->fval + dist_far*dist_far;
    // Explore the endpoint nearest to the relaxation first.
    return diff < 0 ? id : DAQP_ADD_LOWER_FLAG(id);
}

// The binary constraint to branch over (DAQPSettings.branching), with the lower flag if the lower bound is
// explored first, or DAQP_EMPTY_IND if the relaxation satisfies all binary constraints
int daqp_get_branch_id(DAQPWorkspace* work){
    int i, k, id, best = DAQP_EMPTY_IND;
    const int nb = work->bnb->nb, rule = work->settings->branching;
    c_float diff, dist, tol, frac, best_diff = 0, best_dist = 0, best_tol = 0, best_frac = -1;
    DAQPNode* node = work->bnb->tree+work->bnb->n_nodes; // Node being processed

    if(rule == DAQP_BRANCHING_PRIORITY){
        for(k = 0; k < work->bnb->n_priority; k++){
            id = work->bnb->priority[k];
            if(daqp_branch_candidate(id,node,work,&diff,&dist,&tol))
                return daqp_branch_on(id,diff,dist,tol,node,work);
        }
    }
    for(k = 0; k < nb; k++){
        i = rule == DAQP_BRANCHING_LAST ? nb-1-k : k;
        id = work->bnb->bin_ids[i];
        if(!daqp_branch_candidate(id,node,work,&diff,&dist,&tol)) continue;
        if(rule != DAQP_BRANCHING_MOST_FRACTIONAL)
            return daqp_branch_on(id,diff,dist,tol,node,work);
        // Distance to the nearest bound relative to half the distance between the bounds (1 at the midpoint)
        frac = dist/(0.5*(work->dupper[id]-work->dlower[id]));
        if(frac > best_frac){
            best = id; best_diff = diff; best_dist = dist; best_tol = tol; best_frac = frac;
        }
    }
    return best == DAQP_EMPTY_IND ? DAQP_EMPTY_IND :
        daqp_branch_on(best,best_diff,best_dist,best_tol,node,work);
}

void daqp_spawn_children(DAQPNode* node, const int branch_id, DAQPWorkspace* work){

    daqp_save_warmstart(node,work);

    // Update child1 (reuse current node)
    node->bin_id = DAQP_TOGGLE_LOWER_FLAG(branch_id);
    node->depth +=1;

    // Update child2
    (node+1)->bin_id = branch_id;
    (node+1)->depth = node->depth;
    (node+1)->WS_start = node->WS_start;
    (node+1)->WS_end= node->WS_end;

    work->bnb->n_nodes+=2;
}

void daqp_node_cleanup_workspace(int n_clean, DAQPWorkspace* work){
    int i;
    // Cleanup sense
    for(i=n_clean; i<work->n_active; i++)
        work->sense[work->WS[i]]&= DAQP_IS_BINARY(work->WS[i]) ?
            ~(DAQP_ACTIVE+DAQP_IMMUTABLE): ~DAQP_ACTIVE;
    // Reset workspace
    work->sing_ind=DAQP_EMPTY_IND;
    work->n_active=n_clean;
    // Only the retained prefix can still have a valid cached substitution.
    if(work->reuse_ind > n_clean)
        work->reuse_ind=n_clean;
}


int daqp_warmstart_node(DAQPNode* node, DAQPWorkspace* work){
    // Add fixed constraints
    if(daqp_bnb_add_fixed_ids(node,work) < 0) return -1;
    // Add free constraints
    daqp_bnb_load_ws(work->bnb->tree_WS+node->WS_start,node->WS_end-node->WS_start,work);
    work->bnb->nWS = node->WS_start; // always move up tree after warmstart
    return 0;
}

void daqp_save_warmstart(DAQPNode* node, DAQPWorkspace* work){
    node->WS_start = work->bnb->nWS;
    work->bnb->nWS += daqp_bnb_store_ws(work->bnb->tree_WS+work->bnb->nWS,work);
    node->WS_end = work->bnb->nWS;
}

int daqp_add_upper_lower(const int add_id, DAQPWorkspace* work){
    int true_add_id = DAQP_REMOVE_LOWER_FLAG(add_id);
    // Setup new binary constraint
    if(DAQP_EXTRACT_LOWER_FLAG(add_id)){
        DAQP_SET_LOWER(true_add_id);
        daqp_add_constraint(work,true_add_id,-1.0);
    }
    else{
        DAQP_SET_UPPER(true_add_id);
        daqp_add_constraint(work,true_add_id,1.0);
    }
    return 1;
}

int daqp_setup_cold_bnb(DAQPNode* node,DAQPWorkspace *work){
    daqp_node_cleanup_workspace(work->bnb->n_clean,work);
    return daqp_bnb_add_fixed_ids(node,work);
}

/*
 * Best-first search with diving (DAQPSettings.node_selection = DAQP_NODE_BEST_FIRST_DIVE).
 *
 * The open node with the smallest lower bound is selected from the pool (see DAQPNodePool), and
 * the search dives from it: the child nearest to the relaxation is processed next, which only adds
 * one constraint to the working set, until the dive is pruned, infeasible or integer feasible. The
 * other child is stored as an open node with its path of fixed binary constraints and the working
 * set of its parent. When an open node is selected, the fixed binary constraints that its path has
 * in common with the path of the previous node are retained in the working set, the remaining ones
 * are added, and the working set of its parent is loaded as a warm start.
 *
 * The open nodes are pruned with the bound of the incumbent when they are selected. The lower
 * bounds in the heap are hence not below the smallest one, and the search ends when it is pruned
 * (which applies abs_subopt and rel_subopt to the global lower bound).
 *
 * At most max_cap nodes are stored. A depth-first search from a selected node stores at most one
 * node per depth (nb). If storing a node would leave less room than that, the subsequent nodes are
 * stored on a stack instead, which is emptied before the next node is selected from the heap: the
 * subtree of each selected node is then explored depth first, and the heap does not grow.
 */

static int* daqp_pool_path(DAQPNodePool* pool, const int slot){
    return pool->data+(size_t)slot*(pool->path_len+pool->ws_len);
}

// Whether open node a is selected before b: a smaller lower bound, or deeper for equal bounds
static int daqp_pool_before(DAQPNodePool* pool, const int a, const int b){
    const DAQPPoolNode *na = pool->nodes+a, *nb = pool->nodes+b;
    return na->lb < nb->lb || (na->lb == nb->lb && na->depth > nb->depth);
}

static void daqp_pool_heap_push(DAQPNodePool* pool, const int slot){
    int i = pool->n_heap++, parent;
    while(i > 0 && daqp_pool_before(pool,slot,pool->heap[parent = (i-1)/2])){
        pool->heap[i] = pool->heap[parent];
        i = parent;
    }
    pool->heap[i] = slot;
}

static int daqp_pool_heap_pop(DAQPNodePool* pool){
    const int top = pool->heap[0], last = pool->heap[--pool->n_heap];
    int i = 0, child;
    while((child = 2*i+1) < pool->n_heap){
        if(child+1 < pool->n_heap && daqp_pool_before(pool,pool->heap[child+1],pool->heap[child])) child++;
        if(!daqp_pool_before(pool,pool->heap[child],last)) break;
        pool->heap[i] = pool->heap[child];
        i = child;
    }
    pool->heap[i] = last;
    return top;
}

// Store the open node with the path fixed_ids[0..depth-1] followed by bin_id and the lower bound
// lb, to be warm started from the current working set. *lifo is set when the nodes are to be
// stored on the stack (see above). Returns 0 if there is no room (not reached, see above).
static int daqp_pool_store(DAQPWorkspace* work, const int depth, const int bin_id, const c_float lb,
        int* lifo){
    DAQPNodePool* pool = work->bnb->pool;
    const int n_open = pool->n_heap+pool->n_stack, reserve = work->bnb->nb;
    int i, slot, *path;
    // Allocate the room for this node and a subsequent depth-first search
    while(pool->cap < pool->max_cap && n_open+1+reserve > pool->cap)
        if(!pool->grow(pool)) pool->max_cap = pool->cap; // Out of memory
    if(n_open+1+reserve > pool->max_cap) *lifo = 1;
    if(n_open >= pool->max_cap) return 0;

    slot = pool->n_free > 0 ? pool->free_slots[--pool->n_free] : pool->n_slots++;
    path = daqp_pool_path(pool,slot);
    for(i = 0; i < depth; i++) path[i] = work->bnb->fixed_ids[i];
    path[depth] = bin_id;
    pool->nodes[slot].depth = depth;
    pool->nodes[slot].lb = lb;
    pool->nodes[slot].n_ws = daqp_bnb_store_ws(path+pool->path_len,work);
    if(*lifo) pool->stack[pool->n_stack++] = slot;
    else daqp_pool_heap_push(pool,slot);
    if(n_open+1 > pool->peak) pool->peak = n_open+1;
    return 1;
}

// Select an open node that is not pruned: from the stack if it is not empty, else the one with the
// smallest lower bound. Returns its slot, or -1 if there is none (the search is completed).
static int daqp_pool_select(DAQPWorkspace* work){
    DAQPNodePool* pool = work->bnb->pool;
    const c_float bound = 2*work->settings->fval_bound;
    int slot;
    while(pool->n_stack > 0){
        slot = pool->stack[--pool->n_stack];
        if(pool->nodes[slot].lb <= bound) return slot;
        pool->free_slots[pool->n_free++] = slot;
    }
    if(pool->n_heap == 0) return -1;
    slot = daqp_pool_heap_pop(pool);
    if(pool->nodes[slot].lb <= bound) return slot;
    pool->n_heap = 0; // The bounds of all other open nodes are at least as large
    return -1;
}

// Set up the working set for the open node in slot (as node) and solve its relaxation
static int daqp_bnb_process_open(const int slot, DAQPNode* node, DAQPWorkspace* work){
    DAQPBnB* bnb = work->bnb;
    DAQPNodePool* pool = bnb->pool;
    const int* path = daqp_pool_path(pool,slot);
    const int depth = pool->nodes[slot].depth;
    int i, common, infeasible;
    bnb->nodecount+=1;
    // The fixed binary constraints in the common prefix of the paths of the previous node and this
    // node are retained (those that are in the working set, see daqp_bnb_add_fixed_ids)
    for(common = 0; common <= node->depth && common <= depth; common++)
        if(bnb->fixed_ids[common] != path[common]) break;
    if(bnb->n_clean > bnb->neq+common) bnb->n_clean = bnb->neq+common;
    daqp_node_cleanup_workspace(bnb->n_clean,work);
    for(i = common; i <= depth; i++) bnb->fixed_ids[i] = path[i];
    node->depth = depth;
    node->bin_id = path[depth];
    node->lb = pool->nodes[slot].lb;
    infeasible = daqp_bnb_add_fixed_ids(node,work) < 0;
    if(!infeasible) daqp_bnb_load_ws(path+pool->path_len,pool->nodes[slot].n_ws,work);
    pool->free_slots[pool->n_free++] = slot;
    return infeasible ? DAQP_EXIT_INFEASIBLE : daqp_bnb_solve_relaxation(node,work);
}

// Add the binary constraint of node, a child of the previous node, and solve its relaxation
static int daqp_bnb_process_child(DAQPNode* node, DAQPWorkspace* work){
    work->bnb->nodecount+=1;
    if(node->depth >= 0){
        work->bnb->fixed_ids[node->depth] = node->bin_id;
        daqp_add_upper_lower(node->bin_id,work);
        work->sense[DAQP_REMOVE_LOWER_FLAG(node->bin_id)] |= DAQP_IMMUTABLE; //Equality
        if(work->sing_ind != DAQP_EMPTY_IND){ // Need to cold start to not miss integer feasible
            if(daqp_setup_cold_bnb(node,work) < 0) return DAQP_EXIT_INFEASIBLE;
        }
    }
    return daqp_bnb_solve_relaxation(node,work);
}

// The search from the root, which daqp_bnb_lb has set up as tree[0]. Returns the exit flag of the
// latest node (DAQP_EXIT_TIMELIMIT if the search is stopped)
static int daqp_bnb_best_first(DAQPWorkspace* work, const c_float c, const c_float lb,
        c_float* fval_best, c_float** swp_ptr, int* explored){
    DAQPBnB* bnb = work->bnb;
    DAQPNodePool* pool = bnb->pool;
    // The node being processed; daqp_get_branch_id sets the lower bound of the child that is
    // explored first in tree[1] and that of the other child in tree[0]
    DAQPNode* node = bnb->tree;
    int exitflag = DAQP_EXIT_INFEASIBLE, branch_id, slot = -1, lifo = 0, dive = 1;
    pool->n_heap = pool->n_stack = pool->n_free = pool->n_slots = pool->peak = 0;
    bnb->n_nodes = 0;
    while(1){
        if(daqp_bnb_sync(work,c)){
            *explored = 0;
            exitflag = DAQP_EXIT_TIMELIMIT;
            break;
        }
        if(dive){ // The next node of the dive (the root first)
            dive = 0;
            if(node->lb > 2*work->settings->fval_bound) continue;
            exitflag = daqp_bnb_process_child(node,work);
        }
        else{
            if((slot = daqp_pool_select(work)) < 0) break;
            exitflag = daqp_bnb_process_open(slot,node,work);
        }
        if(node->depth < 0 && exitflag > 0 && bnb->root_WS != NULL)
            bnb->n_root_WS = daqp_bnb_store_ws(bnb->root_WS,work);
        if(daqp_bnb_time_exceeded(work)){
            exitflag = DAQP_EXIT_TIMELIMIT;
            *explored = 0;
            break;
        }
        if(exitflag==DAQP_EXIT_INFEASIBLE) continue; // Dominance cut
        if(exitflag<0){ // Inner solver failed => exit loop
            *explored = 0;
            break;
        }

        branch_id = daqp_get_branch_id(work);
        if(branch_id==DAQP_EMPTY_IND){ // Integer feasible
            daqp_bnb_new_incumbent(work,c,lb,fval_best,swp_ptr);
            continue;
        }
        // Store the child that is explored later (unless pruned) and dive into the other one
        if(node->lb <= 2*work->settings->fval_bound &&
                !daqp_pool_store(work,node->depth+1,DAQP_TOGGLE_LOWER_FLAG(branch_id),node->lb,&lifo)){
            exitflag = DAQP_EXIT_UNSUPPORTED;
            *explored = 0;
            break;
        }
        node->depth += 1;
        node->bin_id = branch_id;
        node->lb = (node+1)->lb;
        dive = 1;
    }
    pool->n_heap = pool->n_stack = 0;
    return exitflag;
}
