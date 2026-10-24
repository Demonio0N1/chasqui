// Checks the analytic MHD eigensystem against the primitive-variable Jacobian:
//   L R = I   and   A R = R Lambda   for random states, including degenerate ones.
#include <cstdio>
#include <cmath>
#include <random>
#include "defs.hpp"
#include "eigen_mhd.hpp"

int main() {
  using namespace eigen_mhd;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1.5, 1.5), P(0.05, 3.0);
  const double gamma = 5.0 / 3.0;
  double worst_lr = 0, worst_ar = 0;
  int nbad = 0;
  for (int n = 0; n < 20000; ++n) {
    double d = P(rng), u = U(rng), v = U(rng), w = U(rng), p = P(rng), by = U(rng), bz = U(rng), bx = U(rng);
    if (n % 5 == 1) { by = 0; bz = 0; }                 // no transverse field
    if (n % 5 == 2) { bx = 0; }                          // no normal field
    if (n % 5 == 3) { bx = 0; by = 0; bz = 0; }          // hydro limit
    if (n % 5 == 4) { by = 0; bz = 0; bx = std::sqrt(gamma * p); }   // cf = cs = a degeneracy
    Real L[NW][NW], R[NW][NW];
    Eigensystem(d, u, v, w, p, by, bz, bx, gamma, L, R);
    // primitive Jacobian
    double A[NW][NW] = {{0}};
    A[0][0] = u; A[0][1] = d;
    A[1][1] = u; A[1][4] = 1.0 / d; A[1][5] = by / d; A[1][6] = bz / d;
    A[2][2] = u; A[2][5] = -bx / d;
    A[3][3] = u; A[3][6] = -bx / d;
    A[4][1] = gamma * p; A[4][4] = u;
    A[5][1] = by; A[5][2] = -bx; A[5][5] = u;
    A[6][1] = bz; A[6][3] = -bx; A[6][6] = u;
    // eigenvalues
    const double asq = gamma * p / d, casq = bx * bx / d, ctsq = (by * by + bz * bz) / d;
    const double tsum = asq + casq + ctsq, tdif = asq + ctsq - casq;
    const double disc = std::sqrt(std::max(tdif * tdif + 4 * ctsq * casq, 0.0));
    const double cf = std::sqrt(0.5 * (tsum + disc)), cs = std::sqrt(std::max(0.5 * (tsum - disc), 0.0)), ca = std::sqrt(casq);
    const double lam[NW] = {u - cf, u - ca, u - cs, u, u + cs, u + ca, u + cf};
    double elr = 0, ear = 0;
    for (int i = 0; i < NW; ++i)
      for (int j = 0; j < NW; ++j) {
        double s = 0, t = 0;
        for (int k = 0; k < NW; ++k) { s += L[i][k] * R[k][j]; t += A[i][k] * R[k][j]; }
        elr = std::max(elr, std::fabs(s - (i == j ? 1.0 : 0.0)));
        ear = std::max(ear, std::fabs(t - R[i][j] * lam[j]));
      }
    worst_lr = std::max(worst_lr, elr); worst_ar = std::max(worst_ar, ear);
    if (elr > 1e-9 || ear > 1e-9) {
      if (++nbad <= 5) std::printf("  sample %d (case %d): |LR-I|=%.2e |AR-RL|=%.2e  bx=%.3f by=%.3f bz=%.3f\n", n, n % 5, elr, ear, bx, by, bz);
    }
  }
  std::printf("worst |LR-I| = %.2e, worst |AR - R Lambda| = %.2e, bad samples = %d / 20000\n", worst_lr, worst_ar, nbad);
  return nbad ? 1 : 0;
}
