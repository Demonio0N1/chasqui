// Chasqui MHD — eigensystem of the 1D adiabatic ideal-MHD equations in primitive variables
// W = (rho, u, v, w, p, By, Bz) with Bx constant across the cell (Roe & Balsara 1996;
// Stone et al. 2008, Appendix A). Used for reconstruction in characteristic variables.
//
// Right eigenvectors are the COLUMNS of R, left eigenvectors the ROWS of L, with L R = I.
// Degeneracies (B_perp = 0, Bx = 0, cf = cs) are handled with the usual conventions.
#pragma once
#include "defs.hpp"

namespace eigen_mhd {

constexpr int NW = 7;   // rho, u, v, w, p, by, bz

KOKKOS_INLINE_FUNCTION
void Eigensystem(Real d, Real /*u*/, Real /*v*/, Real /*w*/, Real p, Real by, Real bz, Real bx,
                 Real gamma, Real L[NW][NW], Real R[NW][NW]) {
  const Real sqrtd = sqrt(d);
  const Real isqrtd = Real(1) / sqrtd;
  const Real asq = gamma * p / d;
  const Real a = sqrt(asq);
  const Real casq = bx * bx / d;             // Alfven speed^2 (normal component)
  const Real bperpsq = by * by + bz * bz;
  const Real ctsq = bperpsq / d;
  const Real tsum = asq + casq + ctsq;
  const Real tdif = asq + ctsq - casq;
  const Real disc = sqrt(fmax(tdif * tdif + Real(4) * ctsq * casq, Real(0)));
  const Real cfsq = Real(0.5) * (tsum + disc);
  const Real cssq = fmax(Real(0.5) * (tsum - disc), Real(0));   // may be ~0 by round-off
  const Real cf = sqrt(cfsq), cs = sqrt(cssq);

  // alpha_f, alpha_s (Roe & Balsara 1996, with the degenerate limits)
  Real alf, als;
  if (cfsq - cssq <= Real(1e-12) * cfsq) {
    alf = Real(1); als = Real(0);
  } else if (ctsq <= Real(1e-12) * (asq + casq)) {
    // no transverse field: the fast/slow waves are the acoustic and Alfvenic modes exactly
    if (asq >= casq) { alf = Real(1); als = Real(0); } else { alf = Real(0); als = Real(1); }
  } else if (asq - cssq <= Real(0)) {
    alf = Real(0); als = Real(1);
  } else if (cfsq - asq <= Real(0)) {
    alf = Real(1); als = Real(0);
  } else {
    alf = sqrt((asq - cssq) / (cfsq - cssq));
    als = sqrt((cfsq - asq) / (cfsq - cssq));
  }
  // beta_y, beta_z (direction of the transverse field) and the sign of Bx
  Real bety, betz;
  if (bperpsq > Real(0)) {
    const Real ib = Real(1) / sqrt(bperpsq);
    bety = by * ib; betz = bz * ib;
  } else {
    bety = Real(1) / sqrt(Real(2)); betz = bety;
  }
  const Real s = (bx >= Real(0)) ? Real(1) : Real(-1);

  const Real cff = cf * alf, css = cs * als;
  const Real qf = cff * s, qs = css * s;
  const Real af = a * alf * sqrtd, as = a * als * sqrtd;

  for (int i = 0; i < NW; ++i) for (int j = 0; j < NW; ++j) { L[i][j] = Real(0); R[i][j] = Real(0); }

  // ---- right eigenvectors (columns): 0: u-cf, 1: u-ca, 2: u-cs, 3: u, 4: u+cs, 5: u+ca, 6: u+cf
  R[0][0] = d * alf;  R[0][2] = d * als;  R[0][3] = Real(1);  R[0][4] = d * als;  R[0][6] = d * alf;
  R[1][0] = -cff;     R[1][2] = -css;                         R[1][4] = css;      R[1][6] = cff;
  R[2][0] = qs * bety; R[2][1] = -betz;  R[2][2] = -qf * bety; R[2][4] = qf * bety; R[2][5] = betz;  R[2][6] = -qs * bety;
  R[3][0] = qs * betz; R[3][1] = bety;   R[3][2] = -qf * betz; R[3][4] = qf * betz; R[3][5] = -bety; R[3][6] = -qs * betz;
  R[4][0] = d * alf * asq; R[4][2] = d * als * asq;            R[4][4] = d * als * asq; R[4][6] = d * alf * asq;
  R[5][0] = as * bety; R[5][1] = -betz * s * sqrtd; R[5][2] = -af * bety; R[5][4] = -af * bety; R[5][5] = -betz * s * sqrtd; R[5][6] = as * bety;
  R[6][0] = as * betz; R[6][1] = bety * s * sqrtd;  R[6][2] = -af * betz; R[6][4] = -af * betz; R[6][5] = bety * s * sqrtd;  R[6][6] = as * betz;

  // ---- left eigenvectors (rows)
  const Real norm = Real(0.5) / asq;
  const Real ncff = norm * cff, ncss = norm * css, nqf = norm * qf, nqs = norm * qs;
  const Real naf = norm * af / d, nas = norm * as / d;
  L[0][1] = -ncff; L[0][2] = nqs * bety; L[0][3] = nqs * betz; L[0][4] = norm * alf / d; L[0][5] = nas * bety; L[0][6] = nas * betz;
  L[1][2] = -Real(0.5) * betz; L[1][3] = Real(0.5) * bety; L[1][5] = -Real(0.5) * betz * s * isqrtd; L[1][6] = Real(0.5) * bety * s * isqrtd;
  L[2][1] = -ncss; L[2][2] = -nqf * bety; L[2][3] = -nqf * betz; L[2][4] = norm * als / d; L[2][5] = -naf * bety; L[2][6] = -naf * betz;
  L[3][0] = Real(1); L[3][4] = -Real(1) / asq;
  L[4][1] = ncss; L[4][2] = nqf * bety; L[4][3] = nqf * betz; L[4][4] = norm * als / d; L[4][5] = -naf * bety; L[4][6] = -naf * betz;
  L[5][2] = Real(0.5) * betz; L[5][3] = -Real(0.5) * bety; L[5][5] = -Real(0.5) * betz * s * isqrtd; L[5][6] = Real(0.5) * bety * s * isqrtd;
  L[6][1] = ncff; L[6][2] = -nqs * bety; L[6][3] = -nqs * betz; L[6][4] = norm * alf / d; L[6][5] = nas * bety; L[6][6] = nas * betz;
}

}  // namespace eigen_mhd
