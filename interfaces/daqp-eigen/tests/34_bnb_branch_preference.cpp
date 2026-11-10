#include "api.h"

#include <cmath>
#include <cstdio>
#include <vector>

// Child order DAQP_CHILD_PREFERRED with a preference set by daqp_set_branch_preference:
// - every preference gives the optimal solution of the default (abs_subopt = 0), with equality reduction off and
//   on, so the lower bounds of the children are attached to the right children;
// - a preference that agrees with the optimal solution reduces the number of nodes, also with an equality
//   reduction that removes a constraint in front of the binary constraints (the preference is given in the
//   indices of the original problem);
// - a zero or cleared preference explores the same nodes as NEAREST;
// - invalid inputs are rejected;
// - a parallel portfolio with preferences finds the optimal solution.

namespace {

constexpr int N = 15;  // Variables
constexpr int NB = 10; // Variables 1, ..., NB are binary
constexpr int M = N + 3;

// H = I + c*ones + diagonal, the unconstrained minimizer t has entries in (0.2, 0.8) for the binary variables
// 1, ..., NB, with the general constraints sum(x[1:NB]) <= s, x[NB+1] + x[NB+2] = 0.1 and x[0] = 0.3. The
// equality reduction eliminates x[0], so its bound (constraint 0) is dropped from the reduced problem and the
// indices of the binary constraints in the reduced problem differ from the original ones.
struct Problem {
    c_float H[N*N], f[N], A[3*N], bu[M], bl[M];
    int sense[M];
    DAQPProblem qp{};
    explicit Problem(unsigned seed) {
        unsigned s = seed*2654435761u + 54321u;
        auto rnd = [&s]() { s = s*1664525u + 1013904223u; return (s >> 8)*(1.0/16777216.0); };
        const c_float c = 0.1 + 0.3*rnd();
        c_float t[N];
        for(int i = 0; i < N; i++) t[i] = (i >= 1 && i <= NB) ? 0.2 + 0.6*rnd() : 0.1;
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++) H[i*N+j] = (i == j)*(1 + rnd()) + c;
        for(int i = 0; i < N; i++){
            f[i] = 0;
            for(int j = 0; j < N; j++) f[i] -= H[i*N+j]*t[j];
        }
        for(int k = 0; k < 3*N; k++) A[k] = 0;
        for(int j = 1; j <= NB; j++) A[j] = 1;
        A[N+NB+1] = 1; A[N+NB+2] = 1; A[2*N] = 1;
        for(int i = 0; i < N; i++){
            const bool binary = i >= 1 && i <= NB;
            bu[i] = binary ? 1 : 2;
            bl[i] = binary ? 0 : -2;
            sense[i] = binary ? DAQP_BINARY : 0;
        }
        bu[N] = 2.5 + 3*rnd(); bl[N] = -DAQP_INF; sense[N] = 0;
        bu[N+1] = bl[N+1] = 0.1; sense[N+1] = DAQP_ACTIVE | DAQP_IMMUTABLE;
        bu[N+2] = bl[N+2] = 0.3; sense[N+2] = DAQP_ACTIVE | DAQP_IMMUTABLE;
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

// Solve with one workspace per entry of settings (in parallel if there is more than one). prefs[i] is the
// preference of workspace i (not set if empty)
Solution solve(Problem& p, std::vector<DAQPSettings>& settings, const std::vector<std::vector<int>>& prefs){
    Solution s{};
    const int n = (int)settings.size();
    std::vector<DAQPWorkspace> works(n);
    std::vector<DAQPWorkspace*> ptrs(n);
    for(int i = 0; i < n; i++){
        works[i] = DAQPWorkspace{};
        works[i].settings = &settings[i];
        setup_daqp(&p.qp, &works[i], nullptr);
        if(!prefs[i].empty() && daqp_set_branch_preference(&works[i], prefs[i].data(), M) != 1) s.exitflag = -100;
        ptrs[i] = &works[i];
    }
    DAQPResult res{};
    res.x = s.x;
    res.lam = nullptr;
    if(n == 1) daqp_solve(&res, ptrs[0]);
    else daqp_solve_parallel(&res, ptrs.data(), n);
    if(s.exitflag == 0) s.exitflag = res.exitflag;
    s.nodes = res.nodes;
    s.fval = res.fval;
    for(int i = 0; i < n; i++) free_work(&works[i]);
    return s;
}

Solution solve(Problem& p, const DAQPSettings& settings, const std::vector<int>& pref){
    std::vector<DAQPSettings> one(1, settings);
    return solve(p, one, {pref});
}

bool integer(const Solution& s){
    for(int i = 1; i <= NB; i++)
        if(std::fmin(std::fabs(s.x[i]), std::fabs(s.x[i] - 1)) > 1e-8) return false;
    return true;
}

bool same(const Solution& a, const Solution& b){
    return a.exitflag == DAQP_EXIT_OPTIMAL && b.exitflag == DAQP_EXIT_OPTIMAL && integer(a) &&
        std::fabs(a.fval - b.fval) <= 1e-9*(1 + std::fabs(b.fval));
}

} // namespace

int main(){
    bool all_pass = true;
    const int rules[] = {DAQP_BRANCHING_FIRST, DAQP_BRANCHING_MOST_FRACTIONAL};
    for(int eq_reduction : {DAQP_EQ_REDUCTION_OFF, DAQP_EQ_REDUCTION_ON}){
        bool pass = true;
        int nodes_nearest[2] = {}, nodes_agree[2] = {}, nodes_oppose[2] = {}, solves = 0;
        for(unsigned seed = 0; seed < 40; seed++){
            Problem p(seed);
            DAQPSettings settings;
            daqp_default_settings(&settings);
            settings.eq_reduction = eq_reduction;
            for(int r = 0; r < 2; r++){
                settings.branching = rules[r];
                settings.child_order = DAQP_CHILD_NEAREST;
                const Solution ref = solve(p, settings, {});
                pass = pass && ref.exitflag == DAQP_EXIT_OPTIMAL && integer(ref);
                nodes_nearest[r] += ref.nodes;

                // Preferences: agreeing with the optimum, opposing it, all up, all down, all zero, and random
                std::vector<std::vector<int>> prefs(6, std::vector<int>(M, 0));
                unsigned s = seed*747796405u + 2891336453u;
                for(int i = 1; i <= NB; i++){
                    prefs[0][i] = ref.x[i] > 0.5 ? 1 : -1;
                    prefs[1][i] = -prefs[0][i];
                    prefs[2][i] = 1;
                    prefs[3][i] = -1;
                    s = s*1664525u + 1013904223u;
                    prefs[5][i] = (int)((s >> 16) % 3) - 1;
                }
                // Entries of constraints that are not binary are not used
                for(int i : {0, NB+1, N, N+1, N+2}) prefs[5][i] = (int)((seed + i) % 3) - 1;
                settings.child_order = DAQP_CHILD_PREFERRED;
                for(int k = 0; k < 6; k++){
                    const Solution sol = solve(p, settings, prefs[k]);
                    pass = pass && same(sol, ref);
                    solves++;
                    if(k == 0) nodes_agree[r] += sol.nodes;
                    if(k == 1) nodes_oppose[r] += sol.nodes;
                    if(k == 4) pass = pass && sol.nodes == ref.nodes; // Zero preference: NEAREST
                }
                // Without a preference, PREFERRED explores the nodes of NEAREST
                const Solution none = solve(p, settings, {});
                pass = pass && same(none, ref) && none.nodes == ref.nodes;
            }
            // Parallel portfolio with a preference per workspace
            settings.branching = DAQP_BRANCHING_FIRST;
            settings.child_order = DAQP_CHILD_NEAREST;
            const Solution ref = solve(p, settings, {});
            std::vector<DAQPSettings> portfolio(3, settings);
            portfolio[0].child_order = DAQP_CHILD_PREFERRED;
            portfolio[1].branching = DAQP_BRANCHING_MOST_FRACTIONAL;
            portfolio[2].branching = DAQP_BRANCHING_MOST_FRACTIONAL;
            portfolio[2].child_order = DAQP_CHILD_PREFERRED;
            std::vector<int> oppose(M, 0), up(M, 0);
            for(int i = 1; i <= NB; i++){ oppose[i] = ref.x[i] > 0.5 ? -1 : 1; up[i] = 1; }
            const Solution par = solve(p, portfolio, {oppose, {}, up});
            pass = pass && same(par, ref);
        }
        std::printf("eq_reduction = %2d (%d solves with a preference): nodes over 40 problems\n",
                eq_reduction, solves);
        for(int r = 0; r < 2; r++){
            std::printf("  %-16s nearest %5d, agreeing preference %5d, opposing preference %5d\n",
                    r == 0 ? "first" : "most fractional", nodes_nearest[r], nodes_agree[r], nodes_oppose[r]);
            pass = pass && nodes_agree[r] < nodes_nearest[r] && nodes_agree[r] < nodes_oppose[r];
        }
        std::printf("  %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Invalid inputs
    {
        bool pass = true;
        Problem p(0);
        DAQPSettings settings;
        daqp_default_settings(&settings);
        DAQPWorkspace work{};
        work.settings = &settings;
        setup_daqp(&p.qp, &work, nullptr);
        std::vector<int> pref(M + 1, 0);
        pass = pass && daqp_set_branch_preference(&work, pref.data(), M) == 1;
        pass = pass && daqp_set_branch_preference(&work, pref.data(), M - 1) == -1;
        pass = pass && daqp_set_branch_preference(&work, pref.data(), M + 1) == -1;
        pass = pass && daqp_set_branch_preference(&work, pref.data(), -1) == -1;
        pref[3] = 2;
        pass = pass && daqp_set_branch_preference(&work, pref.data(), M) == -1;
        pref[3] = -2;
        pass = pass && daqp_set_branch_preference(&work, pref.data(), M) == -1;
        // A rejected call keeps the previous preference
        pass = pass && work.bnb->n_branch_pref == M;
        pass = pass && daqp_set_branch_preference(&work, nullptr, 0) == 1 && work.bnb->branch_pref == nullptr;
        free_work(&work);

        // A problem without binary constraints has no branch-and-bound data
        for(int i = 0; i < M; i++) p.sense[i] &= ~DAQP_BINARY;
        DAQPWorkspace cont{};
        cont.settings = &settings;
        setup_daqp(&p.qp, &cont, nullptr);
        pref[3] = 0;
        pass = pass && daqp_set_branch_preference(&cont, pref.data(), M) == -1;
        free_work(&cont);
        std::printf("invalid inputs: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass ? 0 : 1;
}
