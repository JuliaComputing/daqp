#include "api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

// Parallel branch and bound (daqp_solve_parallel) with a portfolio of branching rules: the result is the
// optimal solution that every rule finds on its own, also with equality reduction, for an infeasible
// problem and for a singular Hessian. Workspaces that share settings are rejected.

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
    return all_pass ? 0 : 1;
}
