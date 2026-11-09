#include "api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

// Parallel branch and bound with groups (daqp_solve_parallel_groups): a split of the problem on one or two
// binary variables, one group per combination of their values, gives the optimum of the single solve, also
// with several branching rules per group, with equality reduction, with an infeasible group, for an
// infeasible problem and for a singular Hessian. Groups that do not search restrictions of one problem are
// rejected, and a group whose threads all end at a limit makes the result not optimal.

namespace {

constexpr int N = 30;  // Variables
constexpr int NB = 22; // The first NB variables are binary
constexpr int M = N + 2;

// As in 30_bnb_parallel
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
    int exitflag;
    c_float fval;
    c_float x[N];
};

void free_work(DAQPWorkspace* work){
    work->settings = nullptr;
    free_daqp_workspace(work);
    free_daqp_ldp(work);
}

DAQPSettings make_settings(int eq_reduction, int rule){
    DAQPSettings s;
    daqp_default_settings(&s);
    s.eq_reduction = eq_reduction;
    s.abs_subopt = 0;
    s.rel_subopt = 0;
    s.branching = rule;
    return s;
}

// Each problem in problems is a group, solved by one workspace per entry of rules
Solution solve_groups(std::vector<std::unique_ptr<Problem>>& problems, const std::vector<int>& rules,
        int eq_reduction, int* flag = nullptr){
    Solution s{};
    const int n = (int)(problems.size()*rules.size());
    std::vector<DAQPSettings> settings(n);
    std::vector<DAQPWorkspace> works(n);
    std::vector<DAQPWorkspace*> ptrs(n);
    std::vector<int> groups(n);
    for(int i = 0; i < n; i++){
        const int g = i/(int)rules.size();
        settings[i] = make_settings(eq_reduction, rules[i%rules.size()]);
        works[i] = DAQPWorkspace{};
        works[i].settings = &settings[i];
        setup_daqp(&problems[g]->qp, &works[i], nullptr);
        ptrs[i] = &works[i];
        groups[i] = g;
    }
    DAQPResult res{};
    res.x = s.x;
    s.exitflag = daqp_solve_parallel_groups(&res, ptrs.data(), groups.data(), n);
    s.fval = res.fval;
    if(flag != nullptr) *flag = res.exitflag;
    for(int i = 0; i < n; i++) free_work(&works[i]);
    return s;
}

Solution solve_single(Problem& p, int eq_reduction){
    Solution s{};
    DAQPSettings settings = make_settings(eq_reduction, 0);
    DAQPWorkspace work{};
    work.settings = &settings;
    setup_daqp(&p.qp, &work, nullptr);
    DAQPResult res{};
    res.x = s.x;
    daqp_solve(&res, &work);
    s.exitflag = res.exitflag;
    s.fval = res.fval;
    free_work(&work);
    return s;
}

// One problem per combination of the values of the binary variables split (fixed by their bounds)
std::vector<std::unique_ptr<Problem>> split(unsigned seed, const std::vector<int>& vars,
        void (*modify)(Problem&) = nullptr, c_float diag = 1){
    std::vector<std::unique_ptr<Problem>> problems;
    for(int c = 0; c < (1 << vars.size()); c++){
        problems.push_back(std::make_unique<Problem>(seed, diag));
        if(modify != nullptr) modify(*problems.back());
        for(size_t k = 0; k < vars.size(); k++)
            problems.back()->bu[vars[k]] = problems.back()->bl[vars[k]] = (c >> k) & 1;
    }
    return problems;
}

bool integer(const Solution& s){
    for(int i = 0; i < NB; i++)
        if(std::min(std::fabs(s.x[i]), std::fabs(s.x[i] - 1)) > 1e-8) return false;
    return true;
}

bool same(const Solution& a, const Solution& b, c_float tol = 1e-9){
    return a.exitflag == b.exitflag && integer(a) && std::fabs(a.fval - b.fval) <= tol*(1 + std::fabs(b.fval));
}

} // namespace

int main(){
    bool all_pass = true;
    const std::vector<std::vector<int>> splits = {{0}, {NB-1}, {0, 1}, {3, NB-2}};
    for(int eq_reduction : {DAQP_EQ_REDUCTION_OFF, DAQP_EQ_REDUCTION_ON}){
        for(const std::vector<int>& rules : {std::vector<int>{0}, std::vector<int>{0, 1, 2}}){
            bool pass = true;
            int count = 0;
            for(unsigned seed = 0; seed < 20; seed++){
                Problem p(seed);
                const Solution ref = solve_single(p, eq_reduction);
                pass = pass && ref.exitflag == DAQP_EXIT_OPTIMAL;
                for(const std::vector<int>& vars : splits){
                    std::vector<std::unique_ptr<Problem>> problems = split(seed, vars);
                    const Solution s = solve_groups(problems, rules, eq_reduction);
                    pass = pass && same(s, ref);
                    count++;
                }
            }
            std::printf("eq_reduction = %2d, %zu rule(s) per group, %d splits: %s\n", eq_reduction,
                    rules.size(), count, pass ? "PASS" : "FAIL");
            all_pass &= pass;
        }
    }

    // An infeasible group: at most one binary variable is one, so the group with both split variables
    // fixed to one is infeasible
    {
        bool pass = true;
        auto at_most_one = [](Problem& p){ p.bu[N] = 1.5; };
        for(unsigned seed = 0; seed < 10; seed++){
            Problem p(seed);
            at_most_one(p);
            const Solution ref = solve_single(p, DAQP_EQ_REDUCTION_OFF);
            for(const std::vector<int>& rules : {std::vector<int>{0}, std::vector<int>{0, 2}}){
                std::vector<std::unique_ptr<Problem>> problems = split(seed, {0, 1}, at_most_one);
                pass = pass && ref.exitflag == DAQP_EXIT_OPTIMAL &&
                    same(solve_groups(problems, rules, DAQP_EQ_REDUCTION_OFF), ref);
            }
        }
        std::printf("Infeasible group: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Infeasible problem: the binary variables must sum to a value in (1,2)
    {
        bool pass = true;
        auto infeasible = [](Problem& p){ p.bu[N] = 1.5; p.bl[N] = 1.2; };
        for(unsigned seed = 0; seed < 5; seed++){
            for(const std::vector<int>& vars : splits){
                std::vector<std::unique_ptr<Problem>> problems = split(seed, vars, infeasible);
                pass = pass && solve_groups(problems, {0, 1}, DAQP_EQ_REDUCTION_OFF).exitflag == DAQP_EXIT_INFEASIBLE;
            }
        }
        std::printf("Infeasible problems: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Singular Hessian (daqp_bnb_prox): the bound is shared within each group only. The objective is that
    // of the single solve (within the bias of the regularization), the exit flag optimal (possibly inexact)
    {
        bool pass = true;
        for(unsigned seed = 0; seed < 5; seed++){
            Problem p(seed, 0);
            const Solution ref = solve_single(p, DAQP_EQ_REDUCTION_OFF);
            for(const std::vector<int>& vars : splits){
                std::vector<std::unique_ptr<Problem>> problems = split(seed, vars, nullptr, 0);
                const Solution s = solve_groups(problems, {0, 1}, DAQP_EQ_REDUCTION_OFF);
                pass = pass && ref.exitflag > 0 && (s.exitflag == DAQP_EXIT_OPTIMAL ||
                        s.exitflag == DAQP_EXIT_OPTIMAL_INEXACT) && integer(s) &&
                    std::fabs(s.fval - ref.fval) <= 1e-6*(1 + std::fabs(ref.fval));
            }
        }
        std::printf("Singular Hessian: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // A group whose threads all end at a limit: the result is not optimal. With another thread in the
    // group that is not limited, the group is completed and the result is optimal
    {
        bool pass = true;
        for(unsigned seed = 0; seed < 5; seed++){
            Problem p(seed);
            const Solution ref = solve_single(p, DAQP_EQ_REDUCTION_OFF);
            for(int limited_rules : {1, 2}){
                std::vector<std::unique_ptr<Problem>> problems = split(seed, {0});
                const int n = 1 + limited_rules;
                std::vector<DAQPSettings> settings(n);
                std::vector<DAQPWorkspace> works(n);
                std::vector<DAQPWorkspace*> ptrs(n);
                std::vector<int> groups(n);
                for(int i = 0; i < n; i++){
                    settings[i] = make_settings(DAQP_EQ_REDUCTION_OFF, i);
                    groups[i] = i == 0 ? 0 : 1;
                    if(i == 1) settings[i].iter_limit = 1;
                    works[i] = DAQPWorkspace{};
                    works[i].settings = &settings[i];
                    setup_daqp(&problems[groups[i]]->qp, &works[i], nullptr);
                    ptrs[i] = &works[i];
                }
                Solution s{};
                DAQPResult res{};
                res.x = s.x;
                s.exitflag = daqp_solve_parallel_groups(&res, ptrs.data(), groups.data(), n);
                s.fval = res.fval;
                if(limited_rules == 1) pass = pass && s.exitflag != DAQP_EXIT_OPTIMAL && s.exitflag != DAQP_EXIT_INFEASIBLE;
                else pass = pass && same(s, ref);
                for(int i = 0; i < n; i++) free_work(&works[i]);
            }
        }
        std::printf("Limits: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Rejected: groups that differ in the objective, in the binary constraints or in a setting that
    // changes the problem, invalid group numbers, and more than one group without binary constraints
    {
        auto run = [](void (*modify)(Problem&, DAQPSettings&, int), std::vector<int> groups){
            const int n = (int)groups.size();
            std::vector<std::unique_ptr<Problem>> problems;
            std::vector<DAQPSettings> settings(n);
            std::vector<DAQPWorkspace> works(n);
            std::vector<DAQPWorkspace*> ptrs(n);
            for(int i = 0; i < n; i++){
                problems.push_back(std::make_unique<Problem>(0));
                settings[i] = make_settings(DAQP_EQ_REDUCTION_OFF, i % 3);
                modify(*problems[i], settings[i], groups[i]);
                works[i] = DAQPWorkspace{};
                works[i].settings = &settings[i];
                setup_daqp(&problems[i]->qp, &works[i], nullptr);
                ptrs[i] = &works[i];
            }
            c_float x[N];
            DAQPResult res{};
            res.x = x;
            const int flag = daqp_solve_parallel_groups(&res, ptrs.data(), groups.data(), n);
            for(int i = 0; i < n; i++) free_work(&works[i]);
            return flag;
        };
        auto none = [](Problem&, DAQPSettings&, int){};
        bool pass =
            run([](Problem& p, DAQPSettings&, int g){ if(g == 1) p.f[2] += 0.1; }, {0, 1}) == DAQP_EXIT_UNSUPPORTED &&
            run([](Problem& p, DAQPSettings&, int g){ if(g == 1) p.H[1] += 0.1; }, {0, 0, 1}) == DAQP_EXIT_UNSUPPORTED &&
            run([](Problem& p, DAQPSettings&, int g){ if(g == 1) p.A[3] = 0.5; }, {0, 1}) == DAQP_EXIT_UNSUPPORTED &&
            run([](Problem& p, DAQPSettings&, int g){ if(g == 1) p.sense[3] = 0; }, {0, 1}) == DAQP_EXIT_UNSUPPORTED &&
            run([](Problem&, DAQPSettings& s, int g){ if(g == 1) s.fval_bound = 0; }, {0, 1}) == DAQP_EXIT_UNSUPPORTED &&
            run([](Problem&, DAQPSettings& s, int g){ if(g == 1) s.eq_reduction = DAQP_EQ_REDUCTION_ON; }, {0, 1}) == DAQP_EXIT_UNSUPPORTED &&
            run([](Problem&, DAQPSettings& s, int g){ if(g == 1) s.abs_subopt = 0.1; }, {0, 1}) == DAQP_EXIT_UNSUPPORTED &&
            run(none, {0, 2}) == DAQP_EXIT_UNSUPPORTED &&
            run(none, {0, -1}) == DAQP_EXIT_UNSUPPORTED &&
            run([](Problem& p, DAQPSettings&, int){ for(int i = 0; i < NB; i++) p.sense[i] = 0; }, {0, 1}) == DAQP_EXIT_UNSUPPORTED &&
            // Bounds and the branching rule may differ
            run([](Problem& p, DAQPSettings& s, int g){ p.bu[0] = p.bl[0] = g; }, {0, 1}) == DAQP_EXIT_OPTIMAL;
        std::printf("Incompatible groups rejected: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass ? 0 : 1;
}
