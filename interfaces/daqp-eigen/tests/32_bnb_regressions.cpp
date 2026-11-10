#include "api.h"
#include "utils.h"
extern "C" {
#include "codegen.h"
}

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

// Regressions of branch and bound:
// - a priority list holds indices of the original problem, which differ from those of the problem
//   that an equality elimination installs (and the eliminated constraints have no index there);
// - the generated storage of the working sets of the tree includes the soft constraints;
// - the default bound DAQP_INF on the objective is finite and applies to the objective J, also if
//   the internal objective ||u||^2 = 2(J + c) exceeds it;
// - the data of branch and bound that code generation writes is defined before the first solve (neq);
// - objective data that is not finite or of magnitude at least DAQP_INF is rejected at setup and update.

namespace {

void free_work(DAQPWorkspace* work){
    work->settings = nullptr;
    free_daqp_workspace(work);
    free_daqp_ldp(work);
}

} // namespace

int main(){
    bool all_pass = true;

    // Priority list with equality reduction: x0 = 0 is eliminated, x1 binary with f = -0.4
    {
        c_float H[] = {1, 0, 0, 1}, f[] = {0, -0.4}, A[] = {1, 0, 0, 1}, bu[] = {0, 1}, bl[] = {0, 0}, x[2];
        int sense[] = {DAQP_ACTIVE | DAQP_IMMUTABLE, DAQP_BINARY};
        DAQPProblem qp = {2, 2, 0, H, f, A, bu, bl, sense, nullptr, 0, 0};
        bool pass = true;
        for(int prio : {0, 1}){
            DAQPSettings settings;
            daqp_default_settings(&settings);
            settings.presolve = 0;
            settings.eq_reduction = DAQP_EQ_REDUCTION_ON;
            settings.branching = DAQP_BRANCHING_PRIORITY;
            DAQPWorkspace work{};
            work.settings = &settings;
            setup_daqp(&qp, &work, nullptr);
            pass = pass && daqp_set_branch_priority(&work, &prio, 1) == 1;
            DAQPResult res{};
            res.x = x;
            daqp_solve(&res, &work);
            pass = pass && res.exitflag == DAQP_EXIT_OPTIMAL && std::fabs(x[0]) < 1e-12 && std::fabs(x[1]) < 1e-12;
            free_work(&work);
        }
        std::printf("Priority list with equality reduction: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // Generated tree storage with soft constraints: room for (n+ns+1)*(nb+1) indices
    {
        constexpr int NS = 10;
        c_float H[] = {1, 0, 0, 1}, f[] = {-0.4, 0}, A[2*NS], bu[NS+1], bl[NS+1];
        int sense[NS+1];
        bu[0] = 1; bl[0] = 0; sense[0] = DAQP_BINARY;
        for(int i = 0; i < NS; i++){
            A[2*i] = 0; A[2*i+1] = 1;
            bu[i+1] = DAQP_INF; bl[i+1] = 2; sense[i+1] = DAQP_SOFT;
        }
        DAQPProblem qp = {2, NS+1, 1, H, f, A, bu, bl, sense, nullptr, 0, 0};
        DAQPSettings settings;
        daqp_default_settings(&settings);
        settings.presolve = 0;
        DAQPWorkspace work{};
        work.settings = &settings;
        setup_daqp(&qp, &work, nullptr);
        render_daqp_workspace(&work, "regression_workspace", "./", "reg_");
        std::ifstream in("regression_workspace.c");
        std::stringstream ss;
        ss << in.rdbuf();
        const std::string src = ss.str();
        const std::string decl = "int reg_tree_WS[";
        const size_t pos = src.find(decl);
        const int size = pos == std::string::npos ? -1 : std::stoi(src.substr(pos + decl.size()));
        const bool pass = size >= (2 + NS + 1)*(1 + 1);
        free_work(&work);
        std::remove("regression_workspace.c");
        std::remove("regression_workspace.h");
        std::printf("Generated tree storage with soft constraints (%d): %s\n", size, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // An objective with a large linear term: the internal objective at the optimum x = 0 exceeds
    // 2*DAQP_INF, the objective J is zero
    {
        bool pass = true;
        for(c_float g : {2e15, 1e18}){
            c_float H = 1, f = g, bu = 1, bl = 0, x = -1;
            int sense = DAQP_BINARY;
            DAQPProblem qp = {1, 1, 1, &H, &f, nullptr, &bu, &bl, &sense, nullptr, 0, 0};
            DAQPSettings settings;
            daqp_default_settings(&settings);
            settings.presolve = 0;
            DAQPResult res{};
            res.x = &x;
            daqp_quadprog(&res, &qp, &settings);
            pass = pass && res.exitflag == DAQP_EXIT_OPTIMAL && std::fabs(x) < 1e-12;
        }
        std::printf("Default bound with a large linear term: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    // Code generation before the first solve writes neq = 0. The test runs with MALLOC_PERTURB_ and without
    // the thread cache of glibc (see CMakeLists.txt), so that an allocation that does not initialize neq
    // returns nonzero content
    {
        c_float H[] = {1, 0, 0, 1}, f[] = {-0.4, -0.6}, bu[] = {1, 1}, bl[] = {0, 0};
        int sense[] = {DAQP_BINARY, DAQP_BINARY};
        DAQPProblem qp = {2, 2, 2, H, f, nullptr, bu, bl, sense, nullptr, 0, 0};
        DAQPSettings settings;
        daqp_default_settings(&settings);
        DAQPWorkspace work{};
        work.settings = &settings;
        setup_daqp(&qp, &work, nullptr);
        render_daqp_workspace(&work, "regression_bnb", "./", "rb_");
        std::ifstream in("regression_bnb.c");
        std::stringstream ss;
        ss << in.rdbuf();
        const std::string src = ss.str();
        // DAQPBnB rb_bnb_work= {rb_bin_ids, (int)nb, (int)neq, ...
        const std::string decl = "DAQPBnB rb_bnb_work= {rb_bin_ids, (int)2, (int)";
        const size_t pos = src.find(decl);
        const int neq = pos == std::string::npos ? -1 : std::stoi(src.substr(pos + decl.size()));
        const bool pass = neq == 0 && work.bnb->neq == 0;
        free_work(&work);
        std::remove("regression_bnb.c");
        std::remove("regression_bnb.h");
        std::printf("Generated branch-and-bound data before the first solve (neq %d): %s\n", neq, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    // Objective data beyond DAQP_INF or not finite is rejected (DAQP_EXIT_INVALID_DATA), at setup and at an
    // update, which leaves the workspace unchanged
    {
        bool pass = true;
        c_float H[] = {1, 0, 0, 1}, bu[] = {1, 1}, bl[] = {0, -1}, x[2];
        int sense[] = {DAQP_BINARY, 0};
        auto setup = [&](c_float* f, c_float* Hq, DAQPWorkspace& work, DAQPSettings& settings){
            DAQPProblem qp = {2, 2, 2, Hq, f, nullptr, bu, bl, sense, nullptr, 0, 0};
            daqp_default_settings(&settings);
            work = DAQPWorkspace{};
            work.settings = &settings;
            return setup_daqp(&qp, &work, nullptr);
        };
        const c_float bad[] = {2*DAQP_INF, -DAQP_INF, NAN, INFINITY};
        for(c_float v : bad){
            DAQPSettings settings;
            DAQPWorkspace work;
            c_float f[] = {v, 0};
            pass = pass && setup(f, H, work, settings) == DAQP_EXIT_INVALID_DATA;
            c_float Hb[] = {1, 0, 0, v};
            c_float fz[] = {0, 0};
            pass = pass && setup(fz, Hb, work, settings) == DAQP_EXIT_INVALID_DATA;
        }
        // A valid setup, a rejected update of f, and a solve with the previous data
        DAQPSettings settings;
        DAQPWorkspace work;
        c_float f[] = {-0.4, -0.2};
        pass = pass && setup(f, H, work, settings) == 1;
        DAQPProblem qp = {2, 2, 2, H, f, nullptr, bu, bl, sense, nullptr, 0, 0};
        c_float fbad[] = {1e31, 0};
        DAQPProblem qpb = qp;
        qpb.f = fbad;
        pass = pass && daqp_update_ldp(DAQP_UPDATE_v, &work, &qpb) == DAQP_EXIT_INVALID_DATA;
        DAQPResult res{};
        res.x = x;
        work.qp = &qp;
        daqp_solve(&res, &work);
        pass = pass && res.exitflag == DAQP_EXIT_OPTIMAL && std::fabs(x[0]) < 1e-9 && std::fabs(x[1]-0.2) < 1e-9;
        free_work(&work);
        std::printf("Objective data beyond DAQP_INF rejected: %s\n", pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass ? 0 : 1;
}
