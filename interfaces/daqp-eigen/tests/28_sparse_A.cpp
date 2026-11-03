#include "api.h"
#include "utils.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

// The general constraints in sparse form (DAQPSparseA): a problem with a dense
// Hessian, simple bounds on all variables, binary constraints, equality
// constraints and general constraints with few nonzeros each is solved with
// M*u formed from the dense M (DAQP_SPARSE_A=0) and from the sparse rows, with
// and without equality reduction (with it, also the products of the rows of M
// through G = W*W' and the lazy primal iterate). Both give the same exit flag,
// the same number of nodes and the same solution, also after an update of the
// bounds.

namespace {

constexpr int N = 60; // Variables
constexpr int NB = 6; // The first NB variables are binary
constexpr int NEQ = 10; // Equality constraints (the first general constraints)
constexpr int MA = 80; // General constraints
constexpr int M = N + MA;

// Deterministic values in [-1, 1]
c_float value(int k) { return std::sin(1.3*k + 0.7); }

// DAQP_SPARSE_A=0 (dense M) or unset
void set_sparse(bool sparse) {
#ifdef _WIN32
    _putenv_s("DAQP_SPARSE_A", sparse ? "" : "0");
#else
    if (sparse) unsetenv("DAQP_SPARSE_A");
    else setenv("DAQP_SPARSE_A", "0", 1);
#endif
}

struct Problem {
    std::vector<c_float> H, f, A, bu, bl;
    std::vector<int> sense;
    DAQPProblem qp{};
    explicit Problem(bool binary) : H(N*N), f(N), A(MA*N, 0.0), bu(M), bl(M), sense(M, 0) {
        // H = B'B + I with a dense B
        std::vector<c_float> B(N*N);
        for (int k = 0; k < N*N; ++k) B[k] = 0.1*value(k);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                c_float s = (i == j);
                for (int k = 0; k < N; ++k) s += B[k*N+i]*B[k*N+j];
                H[i*N+j] = s;
            }
        for (int i = 0; i < N; ++i) f[i] = value(3*i+1);
        for (int i = 0; i < N; ++i) {
            bu[i] = i < NB ? 1 : 2;
            bl[i] = i < NB ? 0 : -2;
            sense[i] = (binary && i < NB) ? DAQP_BINARY : 0;
        }
        // Three nonzeros per row; the equalities do not involve the binary variables
        for (int r = 0; r < MA; ++r) {
            for (int k = 0; k < 3; ++k) {
                const int col = r < NEQ ? NB + (r + 17*k) % (N - NB) : (r + 7*k) % N;
                A[r*N+col] += value(5*r+k);
            }
            if (r < NEQ) {
                bu[N+r] = bl[N+r] = 0.1*value(r);
                sense[N+r] = DAQP_ACTIVE | DAQP_IMMUTABLE;
            } else {
                bu[N+r] = 0.5;
                bl[N+r] = -0.5;
            }
        }
        qp = {N, M, N, H.data(), f.data(), A.data(), bu.data(), bl.data(),
              sense.data(), nullptr, 0, 0};
    }
};

struct Result {
    int exitflag, nodes;
    std::vector<c_float> x;
};

// Solves the problem, then again after an update of the bounds of the
// inequality constraints
std::vector<Result> solve(Problem& p, int eq_reduction, bool sparse) {
    set_sparse(sparse);
    DAQPWorkspace work{};
    allocate_daqp_settings(&work);
    work.settings->eq_reduction = eq_reduction;
    assert(setup_daqp_main(&p.qp, &work, nullptr, 0) >= 0);
    // The sparse rows are formed for the problem that is solved
    const bool reduced = DAQP_IS_REDUCED(&work);
    assert(reduced == (eq_reduction == DAQP_EQ_REDUCTION_ON));
    const DAQPSparseA* spA = reduced ? work.eq->other.spA : work.spA;
    assert((spA != nullptr) == sparse);
    if (sparse && reduced) assert(spA->W != nullptr && spA->G != nullptr);

    std::vector<Result> results;
    for (int update = 0; update < 2; ++update) {
        if (update) {
            for (int r = NEQ; r < MA; ++r) {
                p.bu[N+r] = 0.3;
                p.bl[N+r] = -0.3;
            }
            assert(daqp_update_ldp(DAQP_UPDATE_d, &work, &p.qp) >= 0);
        }
        Result res{0, 0, std::vector<c_float>(N)};
        std::vector<c_float> lam(M);
        DAQPResult r{};
        r.x = res.x.data();
        r.lam = lam.data();
        daqp_solve(&r, &work);
        res.exitflag = r.exitflag;
        res.nodes = r.nodes;
        results.push_back(res);
    }
    free_daqp_workspace(&work);
    free_daqp_ldp(&work);
    set_sparse(true);
    return results;
}

} // namespace

int main() {
    for (bool binary : {false, true}) {
        for (int eq_reduction : {DAQP_EQ_REDUCTION_OFF, DAQP_EQ_REDUCTION_ON}) {
            Problem pd(binary), ps(binary);
            const std::vector<Result> dense = solve(pd, eq_reduction, false);
            const std::vector<Result> sparse = solve(ps, eq_reduction, true);
            for (size_t k = 0; k < dense.size(); ++k) {
                assert(dense[k].exitflag > 0);
                assert(sparse[k].exitflag == dense[k].exitflag);
                assert(sparse[k].nodes == dense[k].nodes);
                for (int i = 0; i < N; ++i)
                    assert(std::abs(sparse[k].x[i] - dense[k].x[i]) < 1e-8);
            }
        }
    }
    return 0;
}
