// Bitwise mirror-symmetry test of the reconstruction and Riemann-solver kernels.
//
// For random inputs, the mirrored inputs (x -> -x: stencil reversed, normal velocity and
// normal field negated, L/R swapped) must give bitwise mirrored outputs. Non-associativity
// of floating-point sums/products is the usual culprit (Fleischmann, Adami & Adams 2019).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include "defs.hpp"
#include "recon.hpp"
#include "riemann.hpp"

static int nfail = 0, ntest = 0;
static void expect(bool ok, const char *what, int n) {
  ++ntest;
  if (!ok) { ++nfail; if (nfail <= 20) std::printf("  FAIL %s (sample %d)\n", what, n); }
}

template <Recon R>
static void TestRecon(const char *name, std::mt19937 &rng) {
  std::uniform_real_distribution<double> U(-2.0, 2.0);
  int bad = 0;
  for (int n = 0; n < 200000; ++n) {
    Real q[6], qm[6], qa[6];
    for (int s = 0; s < 6; ++s) q[s] = U(rng);
    if (n % 3 == 0) { for (int s = 0; s < 6; ++s) q[s] = std::fabs(q[s]); }           // positive data
    if (n % 5 == 0) { q[3] = q[2]; q[4] = q[1]; q[5] = q[0]; }                        // even symmetric stencil
    for (int s = 0; s < 6; ++s) { qm[s] = q[5 - s]; qa[s] = -q[5 - s]; }
    Real ql, qr, qlm, qrm, qla, qra;
    recon::Face<R>(q, ql, qr);
    recon::Face<R>(qm, qlm, qrm);    // mirrored scalar: ql' == qr, qr' == ql
    recon::Face<R>(qa, qla, qra);    // mirrored normal vector: ql' == -qr, qr' == -ql
    if (!(qlm == qr && qrm == ql)) ++bad;
    if (!(qla == -qr && qra == -ql)) ++bad;
  }
  std::printf("  %-6s : %d asymmetric samples of 200000\n", name, bad);
  nfail += (bad > 0);
}

template <bool HLLDSOLVER>
static void TestRiemann(const char *name, std::mt19937 &rng) {
  std::uniform_real_distribution<double> U(-1.0, 1.0), P(0.05, 2.0);
  EOS eos{5.0 / 3.0, 2.0 / 3.0, 1e-10, 1e-12};
  int bad = 0;
  for (int n = 0; n < 200000; ++n) {
    State L{P(rng), U(rng), U(rng), U(rng), P(rng), U(rng), U(rng)};
    State R{P(rng), U(rng), U(rng), U(rng), P(rng), U(rng), U(rng)};
    Real bn = U(rng);
    if (n % 4 == 0) bn = 0.0;                                   // degenerate branch
    if (n % 7 == 0) { R = L; R.u = -L.u; }                      // exactly mirror-symmetric pair
    State Lm{R.d, -R.u, R.v, R.w, R.p, R.by, R.bz};
    State Rm{L.d, -L.u, L.v, L.w, L.p, L.by, L.bz};
    Real f[NFLX], g[NFLX];
    if (HLLDSOLVER) { HLLD(L, R, bn, eos, f); HLLD(Lm, Rm, -bn, eos, g); }
    else { HLLE(L, R, bn, eos, f); HLLE(Lm, Rm, -bn, eos, g); }
    const bool ok = (g[FD] == -f[FD]) && (g[FMU] == f[FMU]) && (g[FMV] == -f[FMV]) && (g[FMW] == -f[FMW])
                    && (g[FEN] == -f[FEN]) && (g[FBY] == -f[FBY]) && (g[FBZ] == -f[FBZ]);
    if (!ok) {
      ++bad;
      if (bad <= 3) {
        std::printf("    sample %d bn=%g: ", n, bn);
        for (int k = 0; k < NFLX; ++k) std::printf("[%g vs %g] ", f[k], g[k]);
        std::printf("\n");
      }
    }
  }
  std::printf("  %-6s : %d asymmetric samples of 200000\n", name, bad);
  nfail += (bad > 0);
}

int main() {
  std::mt19937 rng(12345);
  std::printf("reconstruction (mirror of stencil):\n");
  TestRecon<Recon::plm>("plm", rng);
  TestRecon<Recon::ppm>("ppm", rng);
  TestRecon<Recon::wenoz>("wenoz", rng);
  std::printf("Riemann solvers (mirror of L/R states):\n");
  TestRiemann<false>("hlle", rng);
  TestRiemann<true>("hlld", rng);
  std::printf(nfail ? "SYMMETRY FAILURES\n" : "ALL SYMMETRIC\n");
  return nfail ? 1 : 0;
}
