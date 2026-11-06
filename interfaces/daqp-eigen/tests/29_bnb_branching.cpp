#include "api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// Branching rules (DAQPSettings.branching): every rule must find the optimal solution of mixed-integer
// problems (abs_subopt = 0), with equality reduction off and on. The priority list
// (daqp_set_branch_priority) skips indices of constraints that are not binary, and invalid lists are
// rejected.

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

Solution solve(Problem& p, DAQPSettings* settings, const std::vector<int>* priority = nullptr){
    Solution s{};
    DAQPWorkspace work{};
    work.settings = settings;
    setup_daqp(&p.qp, &work, nullptr);
    if(priority != nullptr && daqp_set_branch_priority(&work, priority->data(), (int)priority->size()) < 0)
        s.exitflag = -100;
    DAQPResult res{};
    res.x = s.x;
    res.lam = nullptr;
    if(s.exitflag != -100){
        daqp_solve(&res, &work);
        s.exitflag = res.exitflag;
        s.nodes = res.nodes;
        s.fval = res.fval;
    }
    free_work(&work);
    return s;
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
    const char* names[] = {"first", "last", "most fractional", "priority"};
    for(int eq_reduction : {DAQP_EQ_REDUCTION_OFF, DAQP_EQ_REDUCTION_ON}){
        int total[4] = {0, 0, 0, 0};
        bool pass = true;
        for(unsigned seed = 0; seed < 40; seed++){
            Problem p(seed);
            DAQPSettings settings;
            daqp_default_settings(&settings);
            settings.eq_reduction = eq_reduction;
            const Solution ref = solve(p, &settings);
            pass = pass && ref.exitflag == DAQP_EXIT_OPTIMAL && integer(ref);
            total[0] += ref.nodes;
            for(int rule : {DAQP_BRANCHING_LAST, DAQP_BRANCHING_MOST_FRACTIONAL}){
                settings.branching = rule;
                const Solution s = solve(p, &settings);
                pass = pass && same(s, ref);
                total[rule] += s.nodes;
            }
            // Priority: the binary variables in reverse order, with the index of a constraint that is not
            // binary (the general inequality) first, which is skipped
            settings.branching = DAQP_BRANCHING_PRIORITY;
            std::vector<int> prio = {N};
            for(int i = NB-1; i >= 0; i--) prio.push_back(i);
            Solution s = solve(p, &settings, &prio);
            pass = pass && same(s, ref);
            total[3] += s.nodes;
            // The reverse priority list branches as the rule last
            settings.branching = DAQP_BRANCHING_LAST;
            const Solution last = solve(p, &settings);
            pass = pass && s.nodes == last.nodes;
            // A partial list, followed by the binary constraints that are not in it in their order
            settings.branching = DAQP_BRANCHING_PRIORITY;
            std::vector<int> partial = {NB/2, 1};
            s = solve(p, &settings, &partial);
            pass = pass && same(s, ref);
            // An empty list branches as the rule first
            std::vector<int> empty;
            s = solve(p, &settings, &empty);
            pass = pass && same(s, ref) && s.nodes == ref.nodes;
        }
        std::printf("eq_reduction = %2d: nodes over 40 problems:", eq_reduction);
        for(int r = 0; r < 4; r++) std::printf(" %s %d,", names[r], total[r]);
        std::printf(" %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Invalid priority lists and a workspace without branch-and-bound data are rejected
    {
        Problem p(0);
        DAQPSettings settings;
        daqp_default_settings(&settings);
        DAQPWorkspace work{};
        work.settings = &settings;
        setup_daqp(&p.qp, &work, nullptr);
        int bad_high[] = {0, M}, bad_low[] = {-1};
        bool pass = daqp_set_branch_priority(&work, bad_high, 2) == -1 &&
            daqp_set_branch_priority(&work, bad_low, 1) == -1 &&
            daqp_set_branch_priority(&work, bad_high, -1) == -1;
        free_work(&work);
        for(int i = 0; i < NB; i++) p.sense[i] = 0;
        DAQPWorkspace qp_work{};
        qp_work.settings = &settings;
        setup_daqp(&p.qp, &qp_work, nullptr);
        int ok[] = {0};
        pass = pass && qp_work.bnb == nullptr && daqp_set_branch_priority(&qp_work, ok, 1) == -1;
        free_work(&qp_work);
        std::printf("Invalid priority lists rejected: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass ? 0 : 1;
}
