#include "api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// Best-first node selection with diving (DAQPSettings.node_selection = DAQP_NODE_BEST_FIRST_DIVE): it
// finds the optimal solution of the depth-first search for every branching rule, with and without
// equality reduction, for a singular Hessian (daqp_bnb_prox), with a pool that is too small for the
// search (which then continues depth first), with an incumbent, repeated solves and abs_subopt, and in
// a portfolio of daqp_solve_parallel. An infeasible problem is detected.

namespace {

constexpr int N = 30;  // Variables
constexpr int NB = 22; // The first NB variables are binary
constexpr int M = N + 2;

// As in 30_bnb_parallel: H = I + c*ones + diagonal, the unconstrained minimizer has entries in
// (0.2, 0.8) for the binary variables, sum(x[0:NB]) <= s and x[NB] + x[NB+1] = 0.1
struct Problem {
    c_float H[N*N], f[N], A[2*N], bu[M], bl[M];
    int sense[M];
    DAQPProblem qp{};
    explicit Problem(unsigned seed, c_float diag = 1) {
        unsigned s = seed*2654435761u + 12345u;
        auto rnd = [&s]() { s = s*1664525u + 1013904223u; return (s >> 8)*(1.0/16777216.0); };
        const c_float c = 0.1 + 0.3*rnd();
        c_float t[N];
        for(int i = 0; i < N; i++) t[i] = i < NB ? 0.2 + 0.6*rnd() : 0.1;
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++) H[i*N+j] = (i == j)*diag*(1 + rnd()) + c;
        for(int i = 0; i < N; i++){
            f[i] = 0;
            for(int j = 0; j < N; j++) f[i] -= H[i*N+j]*t[j];
        }
        for(int k = 0; k < 2*N; k++) A[k] = 0;
        for(int j = 0; j < NB; j++) A[j] = 1;
        A[N+NB] = 1; A[N+NB+1] = 1;
        for(int i = 0; i < N; i++){
            bu[i] = i < NB ? 1 : 2;
            bl[i] = i < NB ? 0 : -2;
            sense[i] = i < NB ? DAQP_BINARY : 0;
        }
        bu[N] = 4.5 + 6*rnd(); bl[N] = -DAQP_INF; sense[N] = 0;
        bu[N+1] = bl[N+1] = 0.1; sense[N+1] = DAQP_ACTIVE | DAQP_IMMUTABLE;
        qp = {N, M, N, H, f, A, bu, bl, sense, nullptr, 1, 0};
    }
};

struct Solution {
    int exitflag, nodes, peak;
    c_float fval;
    c_float x[N];
    c_float lam[M];
};

void free_work(DAQPWorkspace* work){
    work->settings = nullptr;
    free_daqp_workspace(work);
    free_daqp_ldp(work);
}

DAQPSettings make_settings(int node_selection, int branching = DAQP_BRANCHING_FIRST,
        int eq_reduction = DAQP_EQ_REDUCTION_OFF){
    DAQPSettings s;
    daqp_default_settings(&s);
    s.node_selection = node_selection;
    s.branching = branching;
    s.eq_reduction = eq_reduction;
    return s;
}

// Priority list (DAQP_BRANCHING_PRIORITY): the binary variables in reverse order
void set_priority(DAQPWorkspace* work){
    int ids[NB];
    for(int i = 0; i < NB; i++) ids[i] = NB-1-i;
    daqp_set_branch_priority(work, ids, NB);
}

// Solve with one workspace per entry of settings (in parallel if there is more than one). If
// incumbent is given, it is the primal start of the first workspace. The problem is solved
// repeat times with the same workspaces (the latest solution is returned).
Solution solve(Problem& p, std::vector<DAQPSettings>& settings, const c_float* incumbent = nullptr,
        int repeat = 1){
    Solution s{};
    const int n = (int)settings.size();
    std::vector<DAQPWorkspace> works(n);
    std::vector<DAQPWorkspace*> ptrs(n);
    for(int i = 0; i < n; i++){
        works[i] = DAQPWorkspace{};
        works[i].settings = &settings[i];
        setup_daqp(&p.qp, &works[i], nullptr);
        if(settings[i].branching == DAQP_BRANCHING_PRIORITY) set_priority(&works[i]);
        ptrs[i] = &works[i];
    }
    DAQPResult res{};
    res.x = s.x;
    res.lam = s.lam;
    for(int r = 0; r < repeat; r++){
        if(incumbent != nullptr){
            c_float x0[N];
            std::copy(incumbent, incumbent+N, x0);
            daqp_set_primal_start(ptrs[0], x0);
        }
        if(n == 1) daqp_solve(&res, ptrs[0]);
        else daqp_solve_parallel(&res, ptrs.data(), n);
    }
    s.exitflag = res.exitflag;
    s.nodes = res.nodes;
    s.fval = res.fval;
    s.peak = daqp_bnb_pool_peak(ptrs[0]);
    for(int i = 0; i < n; i++) free_work(&works[i]);
    return s;
}

Solution solve1(Problem& p, DAQPSettings settings, const c_float* incumbent = nullptr, int repeat = 1){
    std::vector<DAQPSettings> one(1, settings);
    return solve(p, one, incumbent, repeat);
}

bool integer(const Solution& s){
    for(int i = 0; i < NB; i++)
        if(std::min(std::fabs(s.x[i]), std::fabs(s.x[i] - 1)) > 1e-8) return false;
    return true;
}

bool same(const Solution& a, const Solution& b, c_float tol = 1e-9){
    return a.exitflag == b.exitflag && integer(a) && std::fabs(a.fval - b.fval) <= tol*(1 + std::fabs(b.fval));
}

bool report(const char* name, bool pass){
    std::printf("%s: %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

int fail_growth(DAQPNodePool*){ return 0; }

} // namespace

int main(){
    bool all_pass = true;
    constexpr int n_seeds = 10;
    const int rules[] = {DAQP_BRANCHING_FIRST, DAQP_BRANCHING_LAST, DAQP_BRANCHING_MOST_FRACTIONAL,
        DAQP_BRANCHING_PRIORITY};

    // The optimum of depth first for every branching rule, with and without equality reduction
    for(int eq_reduction : {DAQP_EQ_REDUCTION_OFF, DAQP_EQ_REDUCTION_ON}){
        for(int rule : rules){
            bool pass = true;
            long nodes_df = 0, nodes_bf = 0;
            int peak = 0;
            for(unsigned seed = 0; seed < n_seeds; seed++){
                Problem p(seed);
                const Solution df = solve1(p, make_settings(DAQP_NODE_DEPTH_FIRST, rule, eq_reduction));
                const Solution bf = solve1(p, make_settings(DAQP_NODE_BEST_FIRST_DIVE, rule, eq_reduction));
                pass = pass && df.exitflag == DAQP_EXIT_OPTIMAL && integer(df) && same(bf, df);
                pass = pass && df.peak == 0 && bf.peak > 0;
                nodes_df += df.nodes; nodes_bf += bf.nodes;
                peak = std::max(peak, bf.peak);
            }
            std::printf("eq_reduction = %2d, branching = %d: nodes depth first %ld, best first %ld, "
                    "largest pool %d: %s\n", eq_reduction, rule, nodes_df, nodes_bf, peak, pass ? "PASS" : "FAIL");
            all_pass &= pass;
        }
    }

    // A pool that is too small for the search: the remainder is depth first, and at most
    // max(node_pool_size, nb+1) open nodes are stored. With the smallest pool, the search is
    // depth first throughout and processes the nodes of the depth-first search.
    {
        bool pass = true;
        for(int pool_size : {0, 30, 60}){
            for(unsigned seed = 0; seed < n_seeds; seed++){
                Problem p(seed);
                const Solution df = solve1(p, make_settings(DAQP_NODE_DEPTH_FIRST));
                DAQPSettings s = make_settings(DAQP_NODE_BEST_FIRST_DIVE);
                s.node_pool_size = pool_size;
                const Solution bf = solve1(p, s);
                pass = pass && same(bf, df) && bf.peak <= std::max(pool_size, NB+1);
                if(pool_size == 0) pass = pass && bf.nodes == df.nodes;
            }
        }
        all_pass &= report("Pool exhaustion", pass);
    }

    // Singular Hessian (daqp_bnb_prox)
    {
        bool pass = true;
        for(unsigned seed = 0; seed < 5; seed++){
            Problem p(seed, 0);
            const Solution df = solve1(p, make_settings(DAQP_NODE_DEPTH_FIRST));
            const Solution bf = solve1(p, make_settings(DAQP_NODE_BEST_FIRST_DIVE));
            pass = pass && df.exitflag > 0 && bf.exitflag > 0 && integer(bf) &&
                std::fabs(bf.fval - df.fval) <= 1e-6*(1 + std::fabs(df.fval));
        }
        all_pass &= report("Singular Hessian", pass);
    }

    // Infeasible: the binary variables must sum to a value in (1,2)
    {
        bool pass = true;
        for(unsigned seed = 0; seed < 5; seed++){
            Problem p(seed);
            p.bu[N] = 1.5; p.bl[N] = 1.2;
            pass = pass && solve1(p, make_settings(DAQP_NODE_BEST_FIRST_DIVE)).exitflag == DAQP_EXIT_INFEASIBLE;
        }
        all_pass &= report("Infeasible", pass);
    }

    // Incumbent (the optimal solution: no node improves on it), repeated solves (the root is warm
    // started from the previous solve) and abs_subopt
    {
        bool pass = true;
        for(unsigned seed = 0; seed < n_seeds; seed++){
            Problem p(seed);
            const Solution df = solve1(p, make_settings(DAQP_NODE_DEPTH_FIRST));
            const DAQPSettings s = make_settings(DAQP_NODE_BEST_FIRST_DIVE);
            const Solution inc = solve1(p, s, df.x);
            pass = pass && same(inc, df);
            const Solution rep = solve1(p, s, nullptr, 3);
            pass = pass && same(rep, df);
            DAQPSettings sub = s;
            sub.abs_subopt = 0.05;
            const Solution so = solve1(p, sub);
            pass = pass && so.exitflag == DAQP_EXIT_OPTIMAL && integer(so) && so.fval <= df.fval + 0.05 + 1e-9;
        }
        all_pass &= report("Incumbent, repeated solves and abs_subopt", pass);
    }

    // Portfolio of daqp_solve_parallel with depth-first threads and a best-first thread
    for(int eq_reduction : {DAQP_EQ_REDUCTION_OFF, DAQP_EQ_REDUCTION_ON}){
        bool pass = true;
        for(unsigned seed = 0; seed < n_seeds; seed++){
            Problem p(seed);
            const Solution df = solve1(p, make_settings(DAQP_NODE_DEPTH_FIRST, DAQP_BRANCHING_FIRST, eq_reduction));
            std::vector<DAQPSettings> all = {
                make_settings(DAQP_NODE_BEST_FIRST_DIVE, DAQP_BRANCHING_MOST_FRACTIONAL, eq_reduction),
                make_settings(DAQP_NODE_DEPTH_FIRST, DAQP_BRANCHING_FIRST, eq_reduction),
                make_settings(DAQP_NODE_DEPTH_FIRST, DAQP_BRANCHING_LAST, eq_reduction),
                make_settings(DAQP_NODE_BEST_FIRST_DIVE, DAQP_BRANCHING_FIRST, eq_reduction)};
            pass = pass && same(solve(p, all), df);
            Problem pi(seed);
            pi.bu[N] = 1.5; pi.bl[N] = 1.2;
            pass = pass && solve(pi, all).exitflag == DAQP_EXIT_INFEASIBLE;
        }
        std::printf("Parallel portfolio, eq_reduction = %2d: %s\n", eq_reduction, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    // The pool cannot be enlarged (allocation failure, simulated through grow) for more binary constraints
    // than the initial room: the search falls back to depth first and finds the optimum
    {
        constexpr int NV = 65;
        std::vector<c_float> H(NV*NV, 0), f(NV, -0.49), bu(NV, 1), bl(NV, 0), x(NV);
        std::vector<int> sense(NV, DAQP_BINARY);
        for(int i = 0; i < NV; i++) H[i*NV+i] = 1;
        DAQPProblem qp = {NV, NV, NV, H.data(), f.data(), nullptr, bu.data(), bl.data(), sense.data(), nullptr, 0, 0};
        bool pass = true;
        for(int pool_size : {NV+1, 16384}){
            DAQPSettings st;
            daqp_default_settings(&st);
            st.node_selection = DAQP_NODE_BEST_FIRST_DIVE;
            st.node_pool_size = pool_size;
            st.presolve = 0;
            st.abs_subopt = 100;
            DAQPWorkspace w{};
            w.settings = &st;
            setup_daqp(&qp, &w, nullptr);
            pass = pass && daqp_bnb_setup_pool(&w) == 1 && w.bnb->pool->cap >= NV+1;
            w.bnb->pool->grow = fail_growth;
            DAQPResult res{};
            res.x = x.data();
            daqp_solve(&res, &w);
            const int flag_bf = res.exitflag;
            const c_float f_bf = res.fval;
            st.node_selection = DAQP_NODE_DEPTH_FIRST;
            daqp_solve(&res, &w);
            pass = pass && flag_bf == DAQP_EXIT_OPTIMAL && res.exitflag == DAQP_EXIT_OPTIMAL &&
                std::fabs(f_bf - res.fval) <= 1e-9*(1 + std::fabs(res.fval));
            w.settings = nullptr;
            free_daqp_workspace(&w);
            free_daqp_ldp(&w);
        }
        std::printf("Pool that cannot be enlarged: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass ? 0 : 1;
}
