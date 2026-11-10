#include "api.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

// Parallel branch and bound (daqp_solve_parallel) with a portfolio of branching rules: the result is the
// optimal solution that every rule finds on its own, also with equality reduction, for an infeasible
// problem and for a singular Hessian. Workspaces that share settings are rejected. The reusable solver
// (daqp_parallel_setup) gives the same results over repeated solves with updates of the workspaces,
// rejects incompatible workspaces at the setup and at a solve, and can be freed without a solve. The
// overhead of a solve with the persistent worker threads is compared with the one-shot solve, which
// creates the threads in every call.

namespace {

constexpr int N = 30;  // Variables
constexpr int NB = 22; // The first NB variables are binary
constexpr int M = N + 2;

// As in 29_bnb_branching: H = I + c*ones + diagonal, the unconstrained minimizer has entries in
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
    int exitflag, nodes;
    c_float fval, time;
    c_float x[N];
    c_float lam[M];
};

void free_work(DAQPWorkspace* work){
    work->settings = nullptr;
    free_daqp_workspace(work);
    free_daqp_ldp(work);
}

double now(){
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Solve with one workspace per entry of settings (in parallel if there is more than one)
Solution solve(Problem& p, std::vector<DAQPSettings>& settings, bool share_settings = false){
    Solution s{};
    const int n = (int)settings.size();
    std::vector<DAQPWorkspace> works(n);
    std::vector<DAQPWorkspace*> ptrs(n);
    for(int i = 0; i < n; i++){
        works[i] = DAQPWorkspace{};
        works[i].settings = &settings[share_settings ? 0 : i];
        setup_daqp(&p.qp, &works[i], nullptr);
        ptrs[i] = &works[i];
    }
    DAQPResult res{};
    res.x = s.x;
    res.lam = s.lam;
    const double t0 = now();
    if(n == 1) daqp_solve(&res, ptrs[0]);
    else daqp_solve_parallel(&res, ptrs.data(), n);
    s.time = now() - t0;
    s.exitflag = res.exitflag;
    s.nodes = res.nodes;
    s.fval = res.fval;
    for(int i = 0; i < n; i++) free_work(&works[i]);
    return s;
}

bool integer(const Solution& s){
    for(int i = 0; i < NB; i++)
        if(std::min(std::fabs(s.x[i]), std::fabs(s.x[i] - 1)) > 1e-8) return false;
    return true;
}

bool same(const Solution& a, const Solution& b, c_float tol = 1e-9){
    return a.exitflag == b.exitflag && integer(a) && std::fabs(a.fval - b.fval) <= tol*(1 + std::fabs(b.fval));
}

// Small problem of the structure of Problem (n variables, the first nb binary), solved in a fraction
// of a millisecond, for the measurement of the overhead of a parallel solve
struct SmallProblem {
    int n, nb, m;
    std::vector<c_float> H, f, A, bu, bl;
    std::vector<int> sense;
    DAQPProblem qp{};
    SmallProblem(int n_, int nb_, unsigned seed) : n(n_), nb(nb_), m(n_ + 1),
        H(n_*n_), f(n_), A(n_), bu(n_ + 1), bl(n_ + 1), sense(n_ + 1) {
        unsigned s = seed*2654435761u + 12345u;
        auto rnd = [&s]() { s = s*1664525u + 1013904223u; return (s >> 8)*(1.0/16777216.0); };
        std::vector<c_float> t(n);
        for(int i = 0; i < n; i++) t[i] = i < nb ? 0.2 + 0.6*rnd() : 0.1;
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++) H[i*n+j] = (i == j)*(1 + rnd()) + 0.2;
        for(int i = 0; i < n; i++){
            f[i] = 0;
            for(int j = 0; j < n; j++) f[i] -= H[i*n+j]*t[j];
            A[i] = i < nb;
            bu[i] = i < nb ? 1 : 2;
            bl[i] = i < nb ? 0 : -2;
            sense[i] = i < nb ? DAQP_BINARY : 0;
        }
        bu[n] = 0.4*nb; bl[n] = -DAQP_INF; sense[n] = 0;
        qp = {n, m, n, H.data(), f.data(), A.data(), bu.data(), bl.data(), sense.data(), nullptr, 0, 0};
    }
};

std::vector<DAQPSettings> portfolio(int eq_reduction){
    std::vector<DAQPSettings> settings(3);
    for(int r = 0; r < 3; r++){
        daqp_default_settings(&settings[r]);
        settings[r].eq_reduction = eq_reduction;
        settings[r].branching = r; // FIRST, LAST, MOST_FRACTIONAL
    }
    return settings;
}

} // namespace

int main(){
    bool all_pass = true;
    for(int eq_reduction : {DAQP_EQ_REDUCTION_OFF, DAQP_EQ_REDUCTION_ON}){
        bool pass = true;
        double t_rule[3] = {0, 0, 0}, t_min = 0, t_par = 0;
        for(unsigned seed = 0; seed < 20; seed++){
            Problem p(seed);
            std::vector<DAQPSettings> all = portfolio(eq_reduction);
            Solution single[3];
            double t_best = 1e30;
            for(int r = 0; r < 3; r++){
                std::vector<DAQPSettings> one(1, all[r]);
                single[r] = solve(p, one);
                pass = pass && single[r].exitflag == DAQP_EXIT_OPTIMAL && integer(single[r]);
                t_rule[r] += single[r].time;
                t_best = std::min<double>(t_best, single[r].time);
            }
            t_min += t_best;
            const Solution par = solve(p, all);
            t_par += par.time;
            pass = pass && same(par, single[0]);
        }
        std::printf("eq_reduction = %2d: time [ms] over 20 problems: first %.1f, last %.1f, most fractional %.1f, "
                "fastest rule %.1f, parallel %.1f: %s\n", eq_reduction, 1e3*t_rule[0], 1e3*t_rule[1],
                1e3*t_rule[2], 1e3*t_min, 1e3*t_par, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Infeasible: the binary variables must sum to a value in (1,2)
    {
        bool pass = true;
        for(unsigned seed = 0; seed < 5; seed++){
            Problem p(seed);
            p.bu[N] = 1.5; p.bl[N] = 1.2;
            std::vector<DAQPSettings> all = portfolio(DAQP_EQ_REDUCTION_OFF);
            const Solution par = solve(p, all);
            pass = pass && par.exitflag == DAQP_EXIT_INFEASIBLE;
        }
        std::printf("Infeasible problems: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Singular Hessian (daqp_bnb_prox): the objective of the parallel solve is that of the single rule
    {
        bool pass = true;
        for(unsigned seed = 0; seed < 5; seed++){
            Problem p(seed, 0);
            std::vector<DAQPSettings> all = portfolio(DAQP_EQ_REDUCTION_OFF);
            std::vector<DAQPSettings> one(1, all[0]);
            const Solution ref = solve(p, one);
            const Solution par = solve(p, all);
            pass = pass && ref.exitflag > 0 && par.exitflag > 0 && integer(par) &&
                std::fabs(par.fval - ref.fval) <= 1e-6*(1 + std::fabs(ref.fval));
        }
        std::printf("Singular Hessian: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Settings that change the problem a thread searches (cutoff, equality reduction, regularization)
    // must agree, since the search of one thread certifies the incumbents of all; limits of the effort
    // and the branching rule may differ. An empty portfolio is rejected.
    {
        c_float H = 1, f = -0.4, bu = 1, bl = 0, x = 0;
        int sense = DAQP_BINARY;
        DAQPProblem qp = {1, 1, 1, &H, &f, nullptr, &bu, &bl, &sense, nullptr, 0, 0};
        auto run = [&](void (*modify)(DAQPSettings*, int)){
            DAQPSettings st[2];
            DAQPWorkspace w[2] = {}, *ws[] = {&w[0], &w[1]};
            for(int i = 0; i < 2; i++){
                daqp_default_settings(&st[i]);
                modify(&st[i], i);
                w[i].settings = &st[i];
                setup_daqp(&qp, &w[i], nullptr);
            }
            DAQPResult r{};
            r.x = &x;
            daqp_solve_parallel(&r, ws, 2);
            for(auto& wi : w) free_work(&wi);
            return r;
        };
        bool pass = run([](DAQPSettings* s, int i){ if(i == 0) s->fval_bound = -0.2; }).exitflag == DAQP_EXIT_UNSUPPORTED &&
            run([](DAQPSettings* s, int i){ s->eq_reduction = i ? DAQP_EQ_REDUCTION_OFF : DAQP_EQ_REDUCTION_ON; }).exitflag == DAQP_EXIT_UNSUPPORTED &&
            run([](DAQPSettings* s, int i){ if(i == 1) s->eps_prox = -10; }).exitflag == DAQP_EXIT_UNSUPPORTED;
        // A common cutoff gives the result of a single solve
        DAQPResult r = run([](DAQPSettings* s, int i){ s->fval_bound = -0.2; s->branching = i; });
        pass = pass && r.exitflag == DAQP_EXIT_INFEASIBLE;
        r = run([](DAQPSettings* s, int i){ s->branching = i; s->time_limit = 1 + i; s->iter_limit = 1000 + i; });
        pass = pass && r.exitflag == DAQP_EXIT_OPTIMAL && std::fabs(x) < 1e-12;
        DAQPResult empty{};
        empty.x = &x;
        pass = pass && daqp_solve_parallel(&empty, nullptr, 0) == DAQP_EXIT_UNSUPPORTED;
        std::printf("Incompatible settings and empty portfolio rejected: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // A relaxation objective above the default bound DAQP_INF (finite): the parallel solve agrees with
    // the single solve for each equality-reduction policy
    {
        c_float H[] = {1, 0, 0, 1}, f[] = {-0.4, 1e16}, A[] = {0, 1}, bu[] = {1, 1, 0}, bl[] = {0, -1, 0}, x[2];
        int sense[] = {DAQP_BINARY, 0, DAQP_ACTIVE | DAQP_IMMUTABLE};
        DAQPProblem qp = {2, 3, 2, H, f, A, bu, bl, sense, nullptr, 0, 0};
        bool pass = true;
        for(int eq : {DAQP_EQ_REDUCTION_OFF, DAQP_EQ_REDUCTION_ON}){
            DAQPSettings st[2];
            DAQPWorkspace w[2] = {}, *ws[] = {&w[0], &w[1]};
            for(int i = 0; i < 2; i++){
                daqp_default_settings(&st[i]);
                st[i].eq_reduction = eq; st[i].presolve = 0; st[i].branching = i;
                w[i].settings = &st[i];
                setup_daqp(&qp, &w[i], nullptr);
            }
            DAQPResult single{}, par{};
            single.x = x; par.x = x;
            daqp_solve(&single, &w[0]);
            daqp_solve_parallel(&par, ws, 2);
            pass = pass && single.exitflag == par.exitflag;
            for(auto& wi : w) free_work(&wi);
        }
        std::printf("Default bound with a large objective offset: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Workspaces that share settings are rejected
    {
        Problem p(0);
        std::vector<DAQPSettings> all = portfolio(DAQP_EQ_REDUCTION_OFF);
        const Solution par = solve(p, all, true);
        const bool pass = par.exitflag == DAQP_EXIT_UNSUPPORTED;
        std::printf("Shared settings rejected: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Reusable solver: repeated solves with updates of f, of the bounds and of H give the solutions of
    // single solves of fresh workspaces. A setting that makes the workspaces incompatible between two
    // solves is rejected by the solve, and the solver is usable again once it is restored
    for(int eq_reduction : {DAQP_EQ_REDUCTION_OFF, DAQP_EQ_REDUCTION_ON}){
        bool pass = true;
        Problem p(3);
        std::vector<DAQPSettings> all = portfolio(eq_reduction);
        std::vector<DAQPWorkspace> works(all.size());
        std::vector<DAQPWorkspace*> ptrs(all.size());
        for(size_t i = 0; i < all.size(); i++){
            works[i] = DAQPWorkspace{};
            works[i].settings = &all[i];
            setup_daqp(&p.qp, &works[i], nullptr);
            ptrs[i] = &works[i];
        }
        DAQPParallel* par = daqp_parallel_setup(ptrs.data(), (int)ptrs.size());
        pass = pass && par != nullptr;
        for(int k = 0; par != nullptr && k < 8; k++){
            int mask = DAQP_UPDATE_v + DAQP_UPDATE_d;
            const Problem q(100 + k, k == 5 ? 2 : 1); // Data of another problem of the same structure
            for(int i = 0; i < N; i++) p.f[i] = q.f[i];
            p.bu[N] = q.bu[N];
            if(k >= 5){ // A Hessian of another scale from k = 5
                for(int i = 0; i < N*N; i++) p.H[i] = q.H[i];
                mask += DAQP_UPDATE_Rinv;
            }
            for(DAQPWorkspace* w : ptrs) pass = pass && daqp_update_ldp(mask, w, &p.qp) >= 0;
            for(size_t i = 0; i < all.size(); i++) all[i].branching = (int)((i + k) % 3);

            Solution s{};
            DAQPResult res{};
            res.x = s.x;
            res.lam = k % 2 ? s.lam : nullptr;
            daqp_parallel_solve(par, &res);
            std::vector<DAQPSettings> one(1, all[0]);
            const Solution ref = solve(p, one);
            s.exitflag = res.exitflag; s.fval = res.fval;
            pass = pass && ref.exitflag == DAQP_EXIT_OPTIMAL && same(s, ref);
            for(int i = 0; i < N; i++) pass = pass && std::fabs(s.x[i] - ref.x[i]) < 1e-6;

            if(k == 3){ // A cutoff in one workspace only
                all[1].fval_bound = 0;
                pass = pass && daqp_parallel_solve(par, &res) == DAQP_EXIT_UNSUPPORTED;
                all[1].fval_bound = DAQP_INF;
                pass = pass && daqp_parallel_solve(par, &res) == DAQP_EXIT_OPTIMAL &&
                    std::fabs(res.fval - ref.fval) <= 1e-9*(1 + std::fabs(ref.fval));
            }
        }
        daqp_parallel_free(par);
        for(DAQPWorkspace& w : works) free_work(&w);
        std::printf("Reusable solver with updates, eq_reduction = %2d: %s\n", eq_reduction, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Reusable solver: incompatible workspaces are rejected by the setup; a solver can be freed without
    // a solve; a single workspace is solved by daqp_solve
    {
        Problem p(1), p2(2);
        p2.bu[N+1] = p2.bl[N+1] = 0.2; // Another problem of the same dimensions
        std::vector<DAQPSettings> st = portfolio(DAQP_EQ_REDUCTION_OFF);
        DAQPWorkspace w[3] = {};
        DAQPWorkspace* ws[3] = {&w[0], &w[1], &w[2]};
        for(int i = 0; i < 3; i++){
            w[i].settings = &st[i];
            setup_daqp(&p.qp, &w[i], nullptr);
        }
        bool pass = daqp_parallel_setup(nullptr, 3) == nullptr && daqp_parallel_setup(ws, 0) == nullptr;
        // Free without a solve (the worker threads are joined)
        for(int n = 1; n <= 3; n++){
            DAQPParallel* par = daqp_parallel_setup(ws, n);
            pass = pass && par != nullptr;
            daqp_parallel_free(par);
        }
        daqp_parallel_free(nullptr);
        // Shared settings
        w[2].settings = &st[0];
        pass = pass && daqp_parallel_setup(ws, 3) == nullptr;
        w[2].settings = &st[2];
        // A cutoff, a regularization or a policy of equality reduction in one workspace only
        st[1].fval_bound = -1;
        pass = pass && daqp_parallel_setup(ws, 3) == nullptr;
        st[1].fval_bound = DAQP_INF;
        st[2].eps_prox = 1e-3;
        pass = pass && daqp_parallel_setup(ws, 3) == nullptr;
        st[2].eps_prox = st[0].eps_prox;
        // Workspaces of different dimensions
        SmallProblem small(8, 4, 0);
        DAQPSettings st_small;
        daqp_default_settings(&st_small);
        DAQPWorkspace w_small{};
        w_small.settings = &st_small;
        setup_daqp(&small.qp, &w_small, nullptr);
        DAQPWorkspace* mixed[2] = {&w[0], &w_small};
        pass = pass && daqp_parallel_setup(mixed, 2) == nullptr;
        // A single workspace and a compatible portfolio
        Solution s{}, ref{};
        DAQPResult res{}, res_ref{};
        res.x = s.x; res_ref.x = ref.x;
        DAQPParallel* par = daqp_parallel_setup(ws, 1);
        pass = pass && par != nullptr && daqp_parallel_solve(par, &res) == DAQP_EXIT_OPTIMAL;
        daqp_parallel_free(par);
        par = daqp_parallel_setup(ws, 3);
        pass = pass && par != nullptr && daqp_parallel_solve(par, &res_ref) == DAQP_EXIT_OPTIMAL &&
            std::fabs(res.fval - res_ref.fval) <= 1e-9*(1 + std::fabs(res.fval));
        daqp_parallel_free(par);
        // The solve of a solver whose setup failed
        DAQPResult none{};
        pass = pass && daqp_parallel_solve(nullptr, &none) == DAQP_EXIT_UNSUPPORTED;
        for(auto& wi : w) free_work(&wi);
        free_work(&w_small);
        std::printf("Reusable solver: setup rejects incompatible workspaces, free without solve: %s\n",
                pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Overhead of a parallel solve: persistent worker threads (reusable solver) against the creation
    // of the threads in every call (daqp_solve_parallel), on a problem that is solved in a fraction of a
    // millisecond. Only reported, not tested
    {
        const int reps = 2000;
        for(int size : {4, 12, 17})
        for(int n : {2, 4, 8}){
            SmallProblem small(size, 2*size/3, 7);
            std::vector<DAQPSettings> st(n);
            std::vector<DAQPWorkspace> works(n);
            std::vector<DAQPWorkspace*> ptrs(n);
            for(int i = 0; i < n; i++){
                daqp_default_settings(&st[i]);
                st[i].branching = i % 3;
                works[i] = DAQPWorkspace{};
                works[i].settings = &st[i];
                setup_daqp(&small.qp, &works[i], nullptr);
                ptrs[i] = &works[i];
            }
            std::vector<c_float> x(small.n);
            DAQPResult res{};
            res.x = x.data();
            int nodes = 0;
            double t0 = now();
            for(int r = 0; r < reps; r++) daqp_solve(&res, ptrs[0]);
            const double t_single = (now() - t0)/reps;
            t0 = now();
            for(int r = 0; r < reps; r++) daqp_solve_parallel(&res, ptrs.data(), n);
            const double t_oneshot = (now() - t0)/reps;
            DAQPParallel* par = daqp_parallel_setup(ptrs.data(), n);
            t0 = now();
            for(int r = 0; r < reps; r++){
                daqp_parallel_solve(par, &res);
                nodes += res.nodes;
            }
            const double t_pool = (now() - t0)/reps;
            daqp_parallel_set_spin(par, 100000);
            t0 = now();
            for(int r = 0; r < reps; r++) daqp_parallel_solve(par, &res);
            const double t_spin = (now() - t0)/reps;
            daqp_parallel_free(par);
            std::printf("Overhead, %d variables, %d workspaces: time per solve [us]: single %.1f, one-shot parallel %.1f, "
                    "reusable parallel %.1f, with polling %.1f (%.1f nodes per parallel solve)\n", size, n,
                    1e6*t_single, 1e6*t_oneshot, 1e6*t_pool, 1e6*t_spin, (double)nodes/reps);
            for(DAQPWorkspace& w : works) free_work(&w);
        }
    }
    return all_pass ? 0 : 1;
}
