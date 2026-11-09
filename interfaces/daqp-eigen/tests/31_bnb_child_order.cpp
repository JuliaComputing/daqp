#include "api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// Child order (DAQPSettings.child_order) and the branching rule LEAST_FRACTIONAL: every combination of a
// branching rule and a child order must find the optimal solution of mixed-integer problems
// (abs_subopt = 0), with equality reduction off and on. Since the lower bounds of the children are used
// for pruning, a lower bound attached to the wrong child would prune the optimal solution. The default
// child order explores the same nodes as NEAREST, and a parallel portfolio that mixes rules and child
// orders finds the same optimal solution.

namespace {

constexpr int N = 14;  // Variables
constexpr int NB = 10; // The first NB variables are binary
constexpr int M = N + 2;

// H = I + c*ones + diagonal, the unconstrained minimizer t has entries in (0.2, 0.8) for the binary variables,
// with the general constraints sum(x[0:NB]) <= s (fractional, active at the root) and x[NB] + x[NB+1] = 0.1
struct Problem {
    c_float H[N*N], f[N], A[2*N], bu[M], bl[M];
    int sense[M];
    DAQPProblem qp{};
    explicit Problem(unsigned seed) {
        unsigned s = seed*2654435761u + 12345u;
        auto rnd = [&s]() { s = s*1664525u + 1013904223u; return (s >> 8)*(1.0/16777216.0); };
        const c_float c = 0.1 + 0.3*rnd();
        c_float t[N];
        for(int i = 0; i < N; i++) t[i] = i < NB ? 0.2 + 0.6*rnd() : 0.1;
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++) H[i*N+j] = (i == j)*(1 + rnd()) + c;
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
        bu[N] = 2.5 + 3*rnd(); bl[N] = -DAQP_INF; sense[N] = 0;
        bu[N+1] = bl[N+1] = 0.1; sense[N+1] = DAQP_ACTIVE | DAQP_IMMUTABLE;
        qp = {N, M, N, H, f, A, bu, bl, sense, nullptr, 1, 0};
    }
};

struct Solution {
    int exitflag, nodes;
    c_float fval;
    c_float x[N];
};

void free_work(DAQPWorkspace* work){
    work->settings = nullptr;
    free_daqp_workspace(work);
    free_daqp_ldp(work);
}

// Solve with one workspace per entry of settings (in parallel if there is more than one)
Solution solve(Problem& p, std::vector<DAQPSettings>& settings){
    Solution s{};
    const int n = (int)settings.size();
    std::vector<DAQPWorkspace> works(n);
    std::vector<DAQPWorkspace*> ptrs(n);
    for(int i = 0; i < n; i++){
        works[i] = DAQPWorkspace{};
        works[i].settings = &settings[i];
        setup_daqp(&p.qp, &works[i], nullptr);
        ptrs[i] = &works[i];
    }
    DAQPResult res{};
    res.x = s.x;
    res.lam = nullptr;
    if(n == 1) daqp_solve(&res, ptrs[0]);
    else daqp_solve_parallel(&res, ptrs.data(), n);
    s.exitflag = res.exitflag;
    s.nodes = res.nodes;
    s.fval = res.fval;
    for(int i = 0; i < n; i++) free_work(&works[i]);
    return s;
}

Solution solve(Problem& p, const DAQPSettings& settings){
    std::vector<DAQPSettings> one(1, settings);
    return solve(p, one);
}

bool integer(const Solution& s){
    for(int i = 0; i < NB; i++)
        if(std::min(std::fabs(s.x[i]), std::fabs(s.x[i] - 1)) > 1e-8) return false;
    return true;
}

bool same(const Solution& a, const Solution& b){
    return a.exitflag == DAQP_EXIT_OPTIMAL && b.exitflag == DAQP_EXIT_OPTIMAL && integer(a) &&
        std::fabs(a.fval - b.fval) <= 1e-9*(1 + std::fabs(b.fval));
}

} // namespace

int main(){
    bool all_pass = true;
    const int rules[] = {DAQP_BRANCHING_FIRST, DAQP_BRANCHING_LAST, DAQP_BRANCHING_MOST_FRACTIONAL,
        DAQP_BRANCHING_LEAST_FRACTIONAL};
    const char* rule_names[] = {"first", "last", "most fractional", "least fractional"};
    const char* order_names[] = {"nearest", "farthest", "up", "down"};
    for(int eq_reduction : {DAQP_EQ_REDUCTION_OFF, DAQP_EQ_REDUCTION_ON}){
        int total[4][4] = {};
        bool pass = true;
        for(unsigned seed = 0; seed < 40; seed++){
            Problem p(seed);
            DAQPSettings settings;
            daqp_default_settings(&settings);
            settings.eq_reduction = eq_reduction;
            const Solution ref = solve(p, settings);
            pass = pass && ref.exitflag == DAQP_EXIT_OPTIMAL && integer(ref);
            for(int r = 0; r < 4; r++){
                for(int order = DAQP_CHILD_NEAREST; order <= DAQP_CHILD_DOWN; order++){
                    settings.branching = rules[r];
                    settings.child_order = order;
                    const Solution s = solve(p, settings);
                    pass = pass && same(s, ref);
                    total[r][order] += s.nodes;
                    // The default child order is NEAREST
                    if(r == 0 && order == DAQP_CHILD_NEAREST) pass = pass && s.nodes == ref.nodes;
                }
            }
            // Parallel portfolio of rules and child orders
            std::vector<DAQPSettings> portfolio(6, settings);
            const int mix[6][2] = {{DAQP_BRANCHING_FIRST, DAQP_CHILD_NEAREST},
                {DAQP_BRANCHING_LAST, DAQP_CHILD_FARTHEST}, {DAQP_BRANCHING_MOST_FRACTIONAL, DAQP_CHILD_UP},
                {DAQP_BRANCHING_LEAST_FRACTIONAL, DAQP_CHILD_DOWN},
                {DAQP_BRANCHING_LEAST_FRACTIONAL, DAQP_CHILD_NEAREST},
                {DAQP_BRANCHING_MOST_FRACTIONAL, DAQP_CHILD_FARTHEST}};
            for(int i = 0; i < 6; i++){
                portfolio[i].branching = mix[i][0];
                portfolio[i].child_order = mix[i][1];
            }
            const Solution par = solve(p, portfolio);
            pass = pass && same(par, ref);
        }
        std::printf("eq_reduction = %2d: nodes over 40 problems (rule x child order):\n", eq_reduction);
        for(int r = 0; r < 4; r++){
            std::printf("  %-16s", rule_names[r]);
            for(int o = 0; o < 4; o++) std::printf(" %s %5d", order_names[o], total[r][o]);
            std::printf("\n");
        }
        // The child order changes the explored tree
        for(int r = 0; r < 4; r++)
            pass = pass && (total[r][DAQP_CHILD_NEAREST] != total[r][DAQP_CHILD_FARTHEST] ||
                    total[r][DAQP_CHILD_UP] != total[r][DAQP_CHILD_DOWN]);
        // LEAST_FRACTIONAL differs from MOST_FRACTIONAL
        pass = pass && total[2][DAQP_CHILD_NEAREST] != total[3][DAQP_CHILD_NEAREST];
        std::printf("  %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass ? 0 : 1;
}
