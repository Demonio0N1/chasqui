// Chasqui MHD — the compute kernels of one Runge-Kutta stage.
//
// Design goal: minimum DRAM traffic per cell. Each stage runs five kernels:
//   1. Fluxes<dir>   : per face, reconstruct BOTH sides from primitives (stencil reads hit
//                      cache), solve the Riemann problem in registers, write flux + 2 EMFs.
//                      Left/right states never touch global memory.
//   2. CornerEMF     : Gardiner & Stone (2005) upwind average of face EMFs to cell edges.
//   3. Update        : conserved variables (RK combination + flux divergence) and
//                      face-centered B by constrained transport, one launch.
//   4. Boundaries    : periodic / outflow on conserved variables and face fields.
//   5. ConsToPrim    : primitives + cell-centered B + cell-centered EMF + (last stage) dt,
//                      one pass over the data with a single reduction.
#pragma once
#include "defs.hpp"
#include "mesh.hpp"
#include "recon.hpp"
#include "riemann.hpp"
#include "eigen_mhd.hpp"

enum class RSolver : int { hlle = 0, hlld = 1, lhlld = 2 };

struct Config {
  Recon recon = Recon::ppm;
  RSolver rsolver = RSolver::hlld;
  Integrator integ = Integrator::rk3;
  EOS eos{};
  Real cfl = Real(0.3);
  bool low_mach = false;   // Thornber et al. (2008) low-Mach correction of the face velocities
  bool flatten = true;     // Colella-Woodward shock flattening for ppm/ppmx/wenoz
  bool characteristic = false;   // reconstruct in characteristic variables
  Real lhlld_phi = Real(-1);     // debug: force the LHLLD low-Mach factor (-1 = computed)
  BCType bc1 = BCType::periodic, bc2 = BCType::periodic, bc3 = BCType::periodic;
};

namespace kernels {

// -----------------------------------------------------------------------------------------
// Load the 6-cell stencil of variable (array a, component n) along direction (di,dj,dk)
// around the face between cells (k,j,i)-1 and (k,j,i). DC/PLM read only q[1..4].
template <Recon R>
KOKKOS_INLINE_FUNCTION void LoadStencil(const Arr4 &a, int n, int k, int j, int i,
                                        int dk, int dj, int di, Real q[6]) {
  if constexpr (R == Recon::dc || R == Recon::plm) {
    q[1] = a(n, k - 2 * dk, j - 2 * dj, i - 2 * di);
    q[2] = a(n, k - dk, j - dj, i - di);
    q[3] = a(n, k, j, i);
    q[4] = a(n, k + dk, j + dj, i + di);
    q[0] = q[1]; q[5] = q[4];
  } else {
    q[0] = a(n, k - 3 * dk, j - 3 * dj, i - 3 * di);
    q[1] = a(n, k - 2 * dk, j - 2 * dj, i - 2 * di);
    q[2] = a(n, k - dk, j - dj, i - di);
    q[3] = a(n, k, j, i);
    q[4] = a(n, k + dk, j + dj, i + di);
    q[5] = a(n, k + 2 * dk, j + 2 * dj, i + 2 * di);
  }
}

// -----------------------------------------------------------------------------------------
// Fused reconstruction + Riemann solve for one direction.
//   DIR = 0,1,2  (x1,x2,x3).  In the rotated frame: u = v_DIR, v = v_{DIR+1}, w = v_{DIR+2}.
// Options: reconstruction in characteristic variables (local eigen-decomposition of the
// primitive MHD system at each side's cell), Colella-Woodward flattening, Thornber low-Mach
// blending, and the LHLLD factors of Minoshima & Miyoshi (2021).
template <int DIR, Recon R, RSolver S>
void Fluxes(const Mesh &m, const Config &cfg, Arr3 bn) {
  const Geom g(m);
  const EOS eos = cfg.eos;
  const bool low_mach = cfg.low_mach;
  const bool flatten = cfg.flatten && (R != Recon::dc && R != Recon::plm);
  const bool charac = cfg.characteristic && (R != Recon::dc);
  const Real force_phi = cfg.lhlld_phi;
  constexpr int di = (DIR == 0) ? 1 : 0, dj = (DIR == 1) ? 1 : 0, dk = (DIR == 2) ? 1 : 0;
  constexpr int IVU = IVX + DIR, IVV = IVX + (DIR + 1) % 3, IVW = IVX + (DIR + 2) % 3;
  constexpr int IBT1 = (DIR + 1) % 3, IBT2 = (DIR + 2) % 3;
  // unit offsets of the transverse directions (for the LHLLD shock detector)
  constexpr int t1 = (DIR + 1) % 3, t2 = (DIR + 2) % 3;
  constexpr int di1 = (t1 == 0), dj1 = (t1 == 1), dk1 = (t1 == 2);
  constexpr int di2 = (t2 == 0), dj2 = (t2 == 1), dk2 = (t2 == 2);
  const bool has_t1 = (t1 == 0) || (t1 == 1 && m.two_d) || (t1 == 2 && m.three_d);
  const bool has_t2 = (t2 == 0) || (t2 == 1 && m.two_d) || (t2 == 2 && m.three_d);

  // face ranges: normal index over faces of active cells, transverse ranges extended by
  // one cell on each side (needed by the corner-EMF stencil)
  int il = m.is, iu = m.ie, jl = m.js, ju = m.je, kl = m.ks, ku = m.ke;
  if (DIR == 0) iu = m.ie + 1; else if (m.nx1 > 1) { il = m.is - 1; iu = m.ie + 1; }
  if (DIR == 1) ju = m.je + 1; else if (m.two_d) { jl = m.js - 1; ju = m.je + 1; }
  if (DIR == 2) ku = m.ke + 1; else if (m.three_d) { kl = m.ks - 1; ku = m.ke + 1; }

  auto w = m.w; auto bcc = m.bcc;   // bn: the stage's face-normal field, passed by caller
  Arr4 flx = (DIR == 0) ? m.f1 : (DIR == 1) ? m.f2 : m.f3;
  // EMF on these faces: E_{t2} = -F(B_{t1}),  E_{t1} = +F(B_{t2})
  Arr3 et2 = (DIR == 0) ? m.e3x1 : (DIR == 1) ? m.e1x2 : m.e2x3;
  Arr3 et1 = (DIR == 0) ? m.e2x1 : (DIR == 1) ? m.e3x2 : m.e1x3;

  Kokkos::parallel_for("fluxes", Range3({kl, jl, il}, {ku + 1, ju + 1, iu + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      // stencils of the 7 primitive variables in the rotated frame: d, u, v, w, p, by, bz
      Real qs[7][6];
      LoadStencil<R>(w, IDN, k, j, i, dk, dj, di, qs[0]);
      LoadStencil<R>(w, IVU, k, j, i, dk, dj, di, qs[1]);
      LoadStencil<R>(w, IVV, k, j, i, dk, dj, di, qs[2]);
      LoadStencil<R>(w, IVW, k, j, i, dk, dj, di, qs[3]);
      LoadStencil<R>(w, IPR, k, j, i, dk, dj, di, qs[4]);
      LoadStencil<R>(bcc, IBT1, k, j, i, dk, dj, di, qs[5]);
      LoadStencil<R>(bcc, IBT2, k, j, i, dk, dj, di, qs[6]);
      Real wl[7], wr[7];
      if (!charac) {
        for (int n = 0; n < 7; ++n) recon::Face<R>(qs[n], wl[n], wr[n]);
      } else {
        // local characteristic decomposition at the cell on each side of the face
        Real Lm[eigen_mhd::NW][eigen_mhd::NW], Rm[eigen_mhd::NW][eigen_mhd::NW];
        Real c[6], cl, cr;
        // left side: cell i-1 (stencil index 2), normal field of that cell
        eigen_mhd::Eigensystem(qs[0][2], qs[1][2], qs[2][2], qs[3][2], qs[4][2], qs[5][2], qs[6][2],
                               bcc(DIR, k - dk, j - dj, i - di), eos.gamma, Lm, Rm);
        Real chl[7];
        for (int kk = 0; kk < 7; ++kk) {
          for (int st = 0; st < 6; ++st) {
            Real acc = Real(0);
            for (int n = 0; n < 7; ++n) acc += Lm[kk][n] * qs[n][st];
            c[st] = acc;
          }
          recon::Face<R>(c, cl, cr);
          chl[kk] = cl;
        }
        for (int n = 0; n < 7; ++n) {
          Real acc = Real(0);
          for (int kk = 0; kk < 7; ++kk) acc += Rm[n][kk] * chl[kk];
          wl[n] = acc;
        }
        // right side: cell i (stencil index 3)
        eigen_mhd::Eigensystem(qs[0][3], qs[1][3], qs[2][3], qs[3][3], qs[4][3], qs[5][3], qs[6][3],
                               bcc(DIR, k, j, i), eos.gamma, Lm, Rm);
        Real chr[7];
        for (int kk = 0; kk < 7; ++kk) {
          for (int st = 0; st < 6; ++st) {
            Real acc = Real(0);
            for (int n = 0; n < 7; ++n) acc += Lm[kk][n] * qs[n][st];
            c[st] = acc;
          }
          recon::Face<R>(c, cl, cr);
          chr[kk] = cr;
        }
        for (int n = 0; n < 7; ++n) {
          Real acc = Real(0);
          for (int kk = 0; kk < 7; ++kk) acc += Rm[n][kk] * chr[kk];
          wr[n] = acc;
        }
      }
      // entropy variable (dual energy): reconstructed as a scalar
      Real q[6], sl, sr;
      LoadStencil<R>(w, IENT, k, j, i, dk, dj, di, q); recon::Face<R>(q, sl, sr);
      const Real sl_dc = q[2], sr_dc = q[3];

      if (flatten) {
        // flattening coefficients of cell i-1 (stencil i-3..i+1) and cell i (i-2..i+2)
        const Real fL = recon::Flattening(&qs[4][0], qs[1][1], qs[1][3]);
        const Real fR = recon::Flattening(&qs[4][1], qs[1][2], qs[1][4]);
        if (fL > Real(0)) { for (int n = 0; n < 7; ++n) wl[n] += fL * (qs[n][2] - wl[n]); sl += fL * (sl_dc - sl); }
        if (fR > Real(0)) { for (int n = 0; n < 7; ++n) wr[n] += fR * (qs[n][3] - wr[n]); sr += fR * (sr_dc - sr); }
      }
      // positivity: fall back to donor cell on a side that reconstructed to rho<=0 or p<=0
      if (wl[0] <= eos.dfloor || wl[4] <= eos.pfloor) { for (int n = 0; n < 7; ++n) wl[n] = qs[n][2]; sl = sl_dc; }
      if (wr[0] <= eos.dfloor || wr[4] <= eos.pfloor) { for (int n = 0; n < 7; ++n) wr[n] = qs[n][3]; sr = sr_dc; }

      State L{wl[0], wl[1], wl[2], wl[3], wl[4], wl[5], wl[6]};
      State Rr{wr[0], wr[1], wr[2], wr[3], wr[4], wr[5], wr[6]};
      const Real b_n = bn(k, j, i);
      if (low_mach) {
        // z = min(1, max(M_L, M_R)) with the fast magnetosonic Mach number; blends the normal
        // velocities so that the pressure-dissipation of the Riemann solver scales with the
        // Mach number instead of staying O(1) at low speeds (Thornber et al. 2008).
        const Real cl = FastSpeed(L, b_n, eos.gamma), cr = FastSpeed(Rr, b_n, eos.gamma);
        const Real z = fmin(Real(1), fmax(fabs(L.u) / cl, fabs(Rr.u) / cr));
        const Real um = Real(0.5) * (L.u + Rr.u), ud = Real(0.5) * (z * (L.u - Rr.u));
        L.u = um + ud; Rr.u = um - ud;
      }
      Real f[NFLX];
      if constexpr (S == RSolver::hlld) {
        HLLD(L, Rr, b_n, eos, f);
      } else if constexpr (S == RSolver::lhlld) {
        // Minoshima & Miyoshi (2021): shock detector theta from the cell-centered velocity
        // jumps (normal and transverse) and low-Mach factor phi from the modified speed c_u
        const int kL = k - dk, jL = j - dj, iL = i - di;
        const Real cfmax = fmax(FastSpeed(L, b_n, eos.gamma), FastSpeed(Rr, b_n, eos.gamma));
        const Real du = w(IVU, k, j, i) - w(IVU, kL, jL, iL);
        Real dvw = Real(0);
        if (has_t1) {
          const Real d1 = fmin(fmin(w(IVV, kL, jL, iL) - w(IVV, kL - dk1, jL - dj1, iL - di1),
                                    w(IVV, kL + dk1, jL + dj1, iL + di1) - w(IVV, kL, jL, iL)),
                               fmin(w(IVV, k, j, i) - w(IVV, k - dk1, j - dj1, i - di1),
                                    w(IVV, k + dk1, j + dj1, i + di1) - w(IVV, k, j, i)));
          dvw = fmin(dvw, d1);
        }
        if (has_t2) {
          const Real d2 = fmin(fmin(w(IVW, kL, jL, iL) - w(IVW, kL - dk2, jL - dj2, iL - di2),
                                    w(IVW, kL + dk2, jL + dj2, iL + di2) - w(IVW, kL, jL, iL)),
                               fmin(w(IVW, k, j, i) - w(IVW, k - dk2, j - dj2, i - di2),
                                    w(IVW, k + dk2, j + dj2, i + di2) - w(IVW, k, j, i)));
          dvw = fmin(dvw, d2);
        }
        // theta = [min(1, (-min(du,0) + cf) / (-min(dv,dw,0) + cf))]^a with a = 4: it drops to
        // zero on the faces parallel to a strong grid-aligned shock, where the pressure term
        // of S_M feeds the carbuncle / odd-even instability.
        const Real ratio = fmin(Real(1), (-fmin(du, Real(0)) + cfmax) / (-dvw + cfmax));
        const Real r2 = ratio * ratio;
        const Real theta = r2 * r2;
        const Real chi = fmin(Real(1), fmax(FlowSpeedU(L, b_n), FlowSpeedU(Rr, b_n)) / cfmax);
        const Real phi = (force_phi >= Real(0)) ? force_phi : chi * (Real(2) - chi);
        HLLD(L, Rr, b_n, eos, f, theta, phi, true);
      } else {
        HLLE(L, Rr, b_n, eos, f);
      }
      flx(IDN, k, j, i) = f[FD];
      flx(IVU, k, j, i) = f[FMU];
      flx(IVV, k, j, i) = f[FMV];
      flx(IVW, k, j, i) = f[FMW];
      flx(IEN, k, j, i) = f[FEN];
      // entropy flux: upwinded with the mass flux (averaged at an exact zero for symmetry)
      flx(IENT, k, j, i) = (f[FD] > Real(0)) ? f[FD] * sl : (f[FD] < Real(0)) ? f[FD] * sr
                                             : Real(0.5) * (f[FD] * (sl + sr));
      et2(k, j, i) = -f[FBY];
      et1(k, j, i) = f[FBZ];
    });
}

// Upwind selection used by the corner-EMF algorithm: the derivative is taken from the
// upwind side of the mass flux; when the flux is exactly zero both sides are averaged, which
// keeps mirror symmetry (Athena++ convention).
// A mass flux below `tol` (round-off level in code units) counts as zero, so a mirror-
// symmetric flow does not pick a random side on its symmetry plane and seed an asymmetry.
KOKKOS_INLINE_FUNCTION Real Upwind(Real flux, Real left, Real right) {
  constexpr Real tol = Real(1e-14);
  return (flux > tol) ? left : (flux < -tol) ? right : Real(0.5) * (left + right);
}

// -----------------------------------------------------------------------------------------
// Upwind corner EMFs (Gardiner & Stone 2005, eqs. 41-51 as formulated in Stone & Gardiner
// 2009): E_edge = 1/4 (sum of 4 face EMFs) + upwind-selected derivative corrections built
// from the cell-centered EMF and the sign of the mass flux.
inline void CornerEMF(const Mesh &m) {
  auto e1 = m.e1, e2 = m.e2, e3 = m.e3;
  auto e2x1 = m.e2x1, e3x1 = m.e3x1, e1x2 = m.e1x2, e3x2 = m.e3x2, e1x3 = m.e1x3, e2x3 = m.e2x3;
  auto ecc = m.ecc;
  auto f1 = m.f1, f2 = m.f2, f3 = m.f3;
  const int is = m.is, ie = m.ie, js = m.js, je = m.je, ks = m.ks, ke = m.ke;
  const bool two_d = m.two_d, three_d = m.three_d;

  if (!three_d) {
    // E1 and E2 live on x3-edges that coincide (in 2D) with the x2- and x1-faces.
    Kokkos::parallel_for("corner_e12_2d", Range3({ks, js, is}, {ks + 1, je + 2, ie + 2}),
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        e1(k, j, i) = e1x2(k, j, i); e1(k + 1, j, i) = e1x2(k, j, i);
        e2(k, j, i) = e2x1(k, j, i); e2(k + 1, j, i) = e2x1(k, j, i);
      });
    if (!two_d) {
      Kokkos::parallel_for("corner_e3_1d", Range3({ks, js, is}, {ks + 1, js + 1, ie + 2}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          e3(k, j, i) = e3x1(k, j, i); e3(k, j + 1, i) = e3x1(k, j, i);
        });
      return;
    }
  } else {
    // E1 on x1-edges: (k-1/2, j-1/2, i)
    Kokkos::parallel_for("corner_e1", Range3({ks, js, is}, {ke + 2, je + 2, ie + 1}),
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real l3 = Upwind(f2(IDN, k - 1, j, i), e1x3(k, j - 1, i) - ecc(0, k - 1, j - 1, i), e1x3(k, j, i) - ecc(0, k - 1, j, i));
        const Real r3 = Upwind(f2(IDN, k, j, i), e1x3(k, j - 1, i) - ecc(0, k, j - 1, i), e1x3(k, j, i) - ecc(0, k, j, i));
        const Real l2 = Upwind(f3(IDN, k, j - 1, i), e1x2(k - 1, j, i) - ecc(0, k - 1, j - 1, i), e1x2(k, j, i) - ecc(0, k, j - 1, i));
        const Real r2 = Upwind(f3(IDN, k, j, i), e1x2(k - 1, j, i) - ecc(0, k - 1, j, i), e1x2(k, j, i) - ecc(0, k, j, i));
        // (pairwise sums keep the result bitwise mirror-symmetric)
        e1(k, j, i) = Real(0.25) * (((l3 + r3) + (l2 + r2))
                                    + ((e1x2(k - 1, j, i) + e1x2(k, j, i)) + (e1x3(k, j - 1, i) + e1x3(k, j, i))));
      });
    // E2 on x2-edges: (k-1/2, j, i-1/2)
    Kokkos::parallel_for("corner_e2", Range3({ks, js, is}, {ke + 2, je + 1, ie + 2}),
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real l3 = Upwind(f1(IDN, k - 1, j, i), e2x3(k, j, i - 1) - ecc(1, k - 1, j, i - 1), e2x3(k, j, i) - ecc(1, k - 1, j, i));
        const Real r3 = Upwind(f1(IDN, k, j, i), e2x3(k, j, i - 1) - ecc(1, k, j, i - 1), e2x3(k, j, i) - ecc(1, k, j, i));
        const Real l1 = Upwind(f3(IDN, k, j, i - 1), e2x1(k - 1, j, i) - ecc(1, k - 1, j, i - 1), e2x1(k, j, i) - ecc(1, k, j, i - 1));
        const Real r1 = Upwind(f3(IDN, k, j, i), e2x1(k - 1, j, i) - ecc(1, k - 1, j, i), e2x1(k, j, i) - ecc(1, k, j, i));
        e2(k, j, i) = Real(0.25) * (((l3 + r3) + (l1 + r1))
                                    + ((e2x3(k, j, i - 1) + e2x3(k, j, i)) + (e2x1(k - 1, j, i) + e2x1(k, j, i))));
      });
  }
  // E3 on x3-edges: (k, j-1/2, i-1/2)  (2D and 3D)
  Kokkos::parallel_for("corner_e3", Range3({ks, js, is}, {ke + 1, je + 2, ie + 2}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const Real l2 = Upwind(f1(IDN, k, j - 1, i), e3x2(k, j, i - 1) - ecc(2, k, j - 1, i - 1), e3x2(k, j, i) - ecc(2, k, j - 1, i));
      const Real r2 = Upwind(f1(IDN, k, j, i), e3x2(k, j, i - 1) - ecc(2, k, j, i - 1), e3x2(k, j, i) - ecc(2, k, j, i));
      const Real l1 = Upwind(f2(IDN, k, j, i - 1), e3x1(k, j - 1, i) - ecc(2, k, j - 1, i - 1), e3x1(k, j, i) - ecc(2, k, j, i - 1));
      const Real r1 = Upwind(f2(IDN, k, j, i), e3x1(k, j - 1, i) - ecc(2, k, j - 1, i), e3x1(k, j, i) - ecc(2, k, j, i));
      e3(k, j, i) = Real(0.25) * (((l2 + r2) + (l1 + r1))
                                  + ((e3x2(k, j, i - 1) + e3x2(k, j, i)) + (e3x1(k, j - 1, i) + e3x1(k, j, i))));
    });
}

// -----------------------------------------------------------------------------------------
// RK combination + flux divergence for U, and constrained transport for B:
//   out = g0*a + g1*b - beta_dt * (div F | curl E)
struct StageArrays {
  Arr4 ua, ub, uout;
  Arr3 b1a, b2a, b3a, b1b, b2b, b3b, b1out, b2out, b3out;
};

inline void Update(const Mesh &m, const StageArrays &s, Real g0, Real g1, Real beta_dt) {
  const int is = m.is, ie = m.ie, js = m.js, je = m.je, ks = m.ks, ke = m.ke;
  const Real idx1 = Real(1) / m.dx1, idx2 = Real(1) / m.dx2, idx3 = Real(1) / m.dx3;
  const bool two_d = m.two_d, three_d = m.three_d;
  auto f1 = m.f1, f2 = m.f2, f3 = m.f3;
  auto ua = s.ua, ub = s.ub, uout = s.uout;

  Kokkos::parallel_for("update_u", Range3({ks, js, is}, {ke + 1, je + 1, ie + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      for (int n = 0; n < NVAR; ++n) {
        Real divf = (f1(n, k, j, i + 1) - f1(n, k, j, i)) * idx1;
        if (two_d)   divf += (f2(n, k, j + 1, i) - f2(n, k, j, i)) * idx2;
        if (three_d) divf += (f3(n, k + 1, j, i) - f3(n, k, j, i)) * idx3;
        uout(n, k, j, i) = g0 * ua(n, k, j, i) + g1 * ub(n, k, j, i) - beta_dt * divf;
      }
    });

  auto e1 = m.e1, e2 = m.e2, e3 = m.e3;
  auto b1a = s.b1a, b2a = s.b2a, b3a = s.b3a, b1b = s.b1b, b2b = s.b2b, b3b = s.b3b;
  auto b1o = s.b1out, b2o = s.b2out, b3o = s.b3out;
  Kokkos::parallel_for("update_b", Range3({ks, js, is}, {ke + 2, je + 2, ie + 2}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      if (j <= je && k <= ke) {   // B1 on x1-faces
        Real curl = Real(0);
        if (two_d)   curl += (e3(k, j + 1, i) - e3(k, j, i)) * idx2;
        if (three_d) curl -= (e2(k + 1, j, i) - e2(k, j, i)) * idx3;
        b1o(k, j, i) = g0 * b1a(k, j, i) + g1 * b1b(k, j, i) - beta_dt * curl;
      }
      if (i <= ie && k <= ke && (two_d || j == js)) {   // B2 on x2-faces
        Real curl = -(e3(k, j, i + 1) - e3(k, j, i)) * idx1;
        if (three_d) curl += (e1(k + 1, j, i) - e1(k, j, i)) * idx3;
        b2o(k, j, i) = g0 * b2a(k, j, i) + g1 * b2b(k, j, i) - beta_dt * curl;
        if (!two_d) b2o(k, j + 1, i) = b2o(k, j, i);
      }
      if (i <= ie && j <= je && (three_d || k == ks)) {   // B3 on x3-faces
        Real curl = (e2(k, j, i + 1) - e2(k, j, i)) * idx1;
        if (two_d) curl -= (e1(k, j + 1, i) - e1(k, j, i)) * idx2;
        b3o(k, j, i) = g0 * b3a(k, j, i) + g1 * b3b(k, j, i) - beta_dt * curl;
        if (!three_d) b3o(k + 1, j, i) = b3o(k, j, i);
      }
    });
}

// -----------------------------------------------------------------------------------------
// Boundary conditions on conserved variables and face fields: periodic, outflow (zero
// gradient) and reflect (mirror; normal momentum and normal B change sign, so B_n = 0 and
// v_n = 0 on the wall).
inline void Boundaries(const Mesh &m, const Config &cfg, Arr4 u, Arr3 b1, Arr3 b2, Arr3 b3) {
  const int ng = m.ng, nx1 = m.nx1, nx2 = m.nx2, nx3 = m.nx3;
  const int is = m.is, ie = m.ie, js = m.js, je = m.je, ks = m.ks, ke = m.ke;
  const int ni = m.ni, nj = m.nj, nk = m.nk;

  // ---- x1 ----
  {
    const int bc = static_cast<int>(cfg.bc1);   // 0 periodic, 1 outflow, 2 reflect
    Kokkos::parallel_for("bc_x1", Range3({0, 0, 0}, {nk + 1, nj + 1, ng}),
      KOKKOS_LAMBDA(const int k, const int j, const int g) {
        const int il = g, ir = ie + 1 + g;             // ghost cells left / right
        const int srcl = (bc == 0) ? il + nx1 : (bc == 1) ? is : 2 * is - 1 - il;
        const int srcr = (bc == 0) ? ir - nx1 : (bc == 1) ? ie : 2 * ie + 1 - ir;
        const Real sgn = (bc == 2) ? Real(-1) : Real(1);
        if (k < nk && j < nj) {
          for (int n = 0; n < NVAR; ++n) {
            const Real f = (n == IM1) ? sgn : Real(1);
            u(n, k, j, il) = f * u(n, k, j, srcl); u(n, k, j, ir) = f * u(n, k, j, srcr);
          }
          b2(k, j, il) = b2(k, j, srcl); b2(k, j, ir) = b2(k, j, srcr);
          b3(k, j, il) = b3(k, j, srcl); b3(k, j, ir) = b3(k, j, srcr);
        }
        // face-normal field: ghost faces fl (left of face is) and fr (right of face ie+1)
        const int fl = g, fr = ie + 2 + g;
        if (bc == 0)      { b1(k, j, fl) = b1(k, j, fl + nx1); b1(k, j, fr) = b1(k, j, fr - nx1); }
        else if (bc == 1) { b1(k, j, fl) = b1(k, j, is);       b1(k, j, fr) = b1(k, j, ie + 1); }
        else              { b1(k, j, fl) = -b1(k, j, 2 * is - fl); b1(k, j, fr) = -b1(k, j, 2 * (ie + 1) - fr); }
        if (bc == 0 && g == 0) b1(k, j, ie + 1) = b1(k, j, is);   // shared periodic face
      });
  }
  // ---- x2 ----
  if (m.two_d) {
    const int bc = static_cast<int>(cfg.bc2);
    Kokkos::parallel_for("bc_x2", Range3({0, 0, 0}, {nk + 1, ng, ni + 1}),
      KOKKOS_LAMBDA(const int k, const int g, const int i) {
        const int jl = g, jr = je + 1 + g;
        const int srcl = (bc == 0) ? jl + nx2 : (bc == 1) ? js : 2 * js - 1 - jl;
        const int srcr = (bc == 0) ? jr - nx2 : (bc == 1) ? je : 2 * je + 1 - jr;
        const Real sgn = (bc == 2) ? Real(-1) : Real(1);
        if (k < nk && i < ni) {
          for (int n = 0; n < NVAR; ++n) {
            const Real f = (n == IM2) ? sgn : Real(1);
            u(n, k, jl, i) = f * u(n, k, srcl, i); u(n, k, jr, i) = f * u(n, k, srcr, i);
          }
          b3(k, jl, i) = b3(k, srcl, i); b3(k, jr, i) = b3(k, srcr, i);
        }
        if (k < nk) { b1(k, jl, i) = b1(k, srcl, i); b1(k, jr, i) = b1(k, srcr, i); }   // b1 has i = ni too
        const int fl = g, fr = je + 2 + g;
        if (bc == 0)      { b2(k, fl, i) = b2(k, fl + nx2, i); b2(k, fr, i) = b2(k, fr - nx2, i); }
        else if (bc == 1) { b2(k, fl, i) = b2(k, js, i);       b2(k, fr, i) = b2(k, je + 1, i); }
        else              { b2(k, fl, i) = -b2(k, 2 * js - fl, i); b2(k, fr, i) = -b2(k, 2 * (je + 1) - fr, i); }
        if (bc == 0 && g == 0) b2(k, je + 1, i) = b2(k, js, i);
      });
  }
  // ---- x3 ----
  if (m.three_d) {
    const int bc = static_cast<int>(cfg.bc3);
    Kokkos::parallel_for("bc_x3", Range3({0, 0, 0}, {ng, nj + 1, ni + 1}),
      KOKKOS_LAMBDA(const int g, const int j, const int i) {
        const int kl = g, kr = ke + 1 + g;
        const int srcl = (bc == 0) ? kl + nx3 : (bc == 1) ? ks : 2 * ks - 1 - kl;
        const int srcr = (bc == 0) ? kr - nx3 : (bc == 1) ? ke : 2 * ke + 1 - kr;
        const Real sgn = (bc == 2) ? Real(-1) : Real(1);
        if (j < nj && i < ni) {
          for (int n = 0; n < NVAR; ++n) {
            const Real f = (n == IM3) ? sgn : Real(1);
            u(n, kl, j, i) = f * u(n, srcl, j, i); u(n, kr, j, i) = f * u(n, srcr, j, i);
          }
        }
        b1(kl, j, i) = b1(srcl, j, i); b1(kr, j, i) = b1(srcr, j, i);
        b2(kl, j, i) = b2(srcl, j, i); b2(kr, j, i) = b2(srcr, j, i);
        const int fl = g, fr = ke + 2 + g;
        if (bc == 0)      { b3(fl, j, i) = b3(fl + nx3, j, i); b3(fr, j, i) = b3(fr - nx3, j, i); }
        else if (bc == 1) { b3(fl, j, i) = b3(ks, j, i);       b3(fr, j, i) = b3(ke + 1, j, i); }
        else              { b3(fl, j, i) = -b3(2 * ks - fl, j, i); b3(fr, j, i) = -b3(2 * (ke + 1) - fr, j, i); }
        if (bc == 0 && g == 0) b3(ke + 1, j, i) = b3(ks, j, i);
      });
  }
}

// -----------------------------------------------------------------------------------------
// Conserved -> primitive, cell-centered B and EMF, and (optionally) the CFL time step.
// Returns min_i dx_i/(|v_i| + c_f,i) over the active cells (or a huge number).
inline Real ConsToPrim(const Mesh &m, const Config &cfg, Arr4 u, Arr3 b1, Arr3 b2, Arr3 b3,
                       bool compute_dt) {
  const EOS eos = cfg.eos;
  const int is = m.is, ie = m.ie, js = m.js, je = m.je, ks = m.ks, ke = m.ke;
  const bool two_d = m.two_d, three_d = m.three_d;
  const Real dx1 = m.dx1, dx2 = m.dx2, dx3 = m.dx3;
  auto w = m.w, bcc = m.bcc, ecc = m.ecc;
  Real min_dt = Real(1e300);
  Kokkos::parallel_reduce("c2p", Range3({0, 0, 0}, {m.nk, m.nj, m.ni}),
    KOKKOS_LAMBDA(const int k, const int j, const int i, Real &mdt) {
      const Real bx = Real(0.5) * (b1(k, j, i) + b1(k, j, i + 1));
      const Real by = two_d ? Real(0.5) * (b2(k, j, i) + b2(k, j + 1, i)) : b2(k, j, i);
      const Real bz = three_d ? Real(0.5) * (b3(k, j, i) + b3(k + 1, j, i)) : b3(k, j, i);
      const Real d = fmax(u(IDN, k, j, i), eos.dfloor);
      const Real id = Real(1) / d;
      const Real vx = u(IM1, k, j, i) * id, vy = u(IM2, k, j, i) * id, vz = u(IM3, k, j, i) * id;
      const Real bsq = bx * bx + by * by + bz * bz;
      const Real ke_ = Real(0.5) * d * (vx * vx + vy * vy + vz * vz);
      const Real etot = u(IEN, k, j, i);
      const Real eint = etot - ke_ - Real(0.5) * bsq;
      Real p = eos.gm1 * eint;
      // dual energy (entropy switch): where the internal energy is a tiny fraction of the
      // total, the subtraction above has lost its digits; use p = s rho^gamma from the
      // separately advected entropy instead, and re-synchronize the total energy to it.
      // Elsewhere the entropy is re-synchronized to the (reliable) total-energy pressure,
      // so that it tracks shock heating.
      Real sent = u(IENT, k, j, i) * id;
      if (eos.dual_energy && eint < eos.de_eta * fabs(etot)) {
        p = sent * pow(d, eos.gamma);
        p = fmax(p, eos.pfloor);
        u(IEN, k, j, i) = ke_ + Real(0.5) * bsq + p / eos.gm1;
      } else {
        p = fmax(p, eos.pfloor);
        sent = p / pow(d, eos.gamma);
        u(IENT, k, j, i) = d * sent;
      }
      w(IDN, k, j, i) = d; w(IVX, k, j, i) = vx; w(IVY, k, j, i) = vy; w(IVZ, k, j, i) = vz; w(IPR, k, j, i) = p;
      w(IENT, k, j, i) = sent;
      bcc(IBX, k, j, i) = bx; bcc(IBY, k, j, i) = by; bcc(IBZ, k, j, i) = bz;
      // E = -(v x B)
      ecc(0, k, j, i) = vz * by - vy * bz;
      ecc(1, k, j, i) = vx * bz - vz * bx;
      ecc(2, k, j, i) = vy * bx - vx * by;
      if (compute_dt && i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke) {
        const Real gp = eos.gamma * p;
        const Real a = (gp + bsq) * id;
        auto cf = [&](Real bn) {
          const Real disc = a * a - Real(4) * gp * bn * bn * id * id;
          return sqrt(Real(0.5) * (a + sqrt(fmax(disc, Real(0)))));
        };
        Real dt = dx1 / (fabs(vx) + cf(bx));
        if (two_d)   dt = fmin(dt, dx2 / (fabs(vy) + cf(by)));
        if (three_d) dt = fmin(dt, dx3 / (fabs(vz) + cf(bz)));
        mdt = fmin(mdt, dt);
      }
    }, Kokkos::Min<Real>(min_dt));
  return min_dt;
}

// -----------------------------------------------------------------------------------------
// Runtime dispatch of the compile-time templates
template <int DIR, Recon R>
void FluxesSolver(const Mesh &m, const Config &cfg, Arr3 bn) {
  if (cfg.rsolver == RSolver::hlld) Fluxes<DIR, R, RSolver::hlld>(m, cfg, bn);
  else if (cfg.rsolver == RSolver::lhlld) Fluxes<DIR, R, RSolver::lhlld>(m, cfg, bn);
  else Fluxes<DIR, R, RSolver::hlle>(m, cfg, bn);
}

template <int DIR>
void FluxesDispatch(const Mesh &m, const Config &cfg, Arr3 bn) {
  switch (cfg.recon) {
    case Recon::dc:    FluxesSolver<DIR, Recon::dc>(m, cfg, bn); break;
    case Recon::plm:   FluxesSolver<DIR, Recon::plm>(m, cfg, bn); break;
    case Recon::ppm:   FluxesSolver<DIR, Recon::ppm>(m, cfg, bn); break;
    case Recon::ppmx:  FluxesSolver<DIR, Recon::ppmx>(m, cfg, bn); break;
    default:           FluxesSolver<DIR, Recon::wenoz>(m, cfg, bn);
  }
}

}  // namespace kernels
