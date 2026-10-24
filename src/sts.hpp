// Chasqui MHD — RKL2 super-time-stepping for the diffusive terms (Meyer, Balsara & Aslam
// 2014, J. Comput. Phys. 257, 594). Operator-split after the hyperbolic update: the
// parabolic operator M(U) = -div(F_diff), M(B) = -curl(eta J) is advanced over the full
// hyperbolic time step tau with s Runge-Kutta-Legendre stages, stable for
//   tau <= dt_par * (s^2 + s - 2) / 4,
// so the cost grows like sqrt(tau / dt_par) instead of tau / dt_par for sub-cycling.
#pragma once
#include <cmath>
#include "defs.hpp"
#include "mesh.hpp"
#include "physics.hpp"
#include "diffusion.hpp"

namespace sts {

struct Work {   // scratch arrays, allocated on first use
  bool allocated = false;
  Arr4 y0, y1, y2, m0, mj;
  Arr3 b0[3], b1[3], b2[3], mb0[3], mbj[3];
  void Allocate(const Mesh &m) {
    if (allocated) return;
    y0 = Arr4("sts_y0", NVAR, m.nk, m.nj, m.ni); y1 = Arr4("sts_y1", NVAR, m.nk, m.nj, m.ni);
    y2 = Arr4("sts_y2", NVAR, m.nk, m.nj, m.ni); m0 = Arr4("sts_m0", NVAR, m.nk, m.nj, m.ni);
    mj = Arr4("sts_mj", NVAR, m.nk, m.nj, m.ni);
    for (int c = 0; c < 3; ++c) {
      b0[c] = Arr3("sts_b0", m.nk + 1, m.nj + 1, m.ni + 1); b1[c] = Arr3("sts_b1", m.nk + 1, m.nj + 1, m.ni + 1);
      b2[c] = Arr3("sts_b2", m.nk + 1, m.nj + 1, m.ni + 1); mb0[c] = Arr3("sts_mb0", m.nk + 1, m.nj + 1, m.ni + 1);
      mbj[c] = Arr3("sts_mbj", m.nk + 1, m.nj + 1, m.ni + 1);
    }
    allocated = true;
  }
};

// Number of RKL2 stages needed to cover tau with parabolic limit dt_par.
inline int Stages(Real tau, Real dt_par) {
  const double r = tau / dt_par;
  int s = static_cast<int>(std::ceil(0.5 * (-1.0 + std::sqrt(9.0 + 16.0 * r))));
  return (s < 2) ? 2 : s;
}

// M(U,B): the parabolic operator evaluated on (u, b) with primitives already computed in m.w
// (the caller runs ConsToPrim first). Writes -div F_diff into mu and -curl(eta J) into mb.
inline void Operator(const Mesh &m, const Config & /*cfg*/, const diffusion::Coeffs &dc,
                     Arr3 b1c, Arr3 b2c, Arr3 b3c, Arr4 mu, Arr3 mb1, Arr3 mb2, Arr3 mb3) {
  const int is = m.is, ie = m.ie, js = m.js, je = m.je, ks = m.ks, ke = m.ke;
  const bool two_d = m.two_d, three_d = m.three_d;
  auto f1 = m.f1, f2 = m.f2, f3 = m.f3;
  auto e1 = m.e1, e2 = m.e2, e3 = m.e3;
  // zero the flux / EMF arrays, then let the diffusion kernels add their contributions
  Kokkos::parallel_for("sts_zero", Range3({0, 0, 0}, {m.nk + 1, m.nj + 1, m.ni + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      for (int n = 0; n < NVAR; ++n) { f1(n, k, j, i) = 0; f2(n, k, j, i) = 0; f3(n, k, j, i) = 0; }
      e1(k, j, i) = 0; e2(k, j, i) = 0; e3(k, j, i) = 0;
    });
  diffusion::Fluxes<0>(m, dc, b1c, b2c, b3c);
  if (two_d) diffusion::Fluxes<1>(m, dc, b1c, b2c, b3c);
  if (three_d) diffusion::Fluxes<2>(m, dc, b1c, b2c, b3c);
  diffusion::ResistiveEMF(m, dc, b1c, b2c, b3c);
  const Real idx1 = Real(1) / m.dx1, idx2 = Real(1) / m.dx2, idx3 = Real(1) / m.dx3;
  Kokkos::parallel_for("sts_div", Range3({ks, js, is}, {ke + 2, je + 2, ie + 2}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      if (i <= ie && j <= je && k <= ke) {
        for (int n = 0; n < NVAR; ++n) {
          Real d = (f1(n, k, j, i + 1) - f1(n, k, j, i)) * idx1;
          if (two_d)   d += (f2(n, k, j + 1, i) - f2(n, k, j, i)) * idx2;
          if (three_d) d += (f3(n, k + 1, j, i) - f3(n, k, j, i)) * idx3;
          mu(n, k, j, i) = -d;
        }
      }
      // curl of the resistive EMF on the faces (same stencil as the CT update)
      if (j <= je && k <= ke) {
        Real c = Real(0);
        if (two_d)   c += (e3(k, j + 1, i) - e3(k, j, i)) * idx2;
        if (three_d) c -= (e2(k + 1, j, i) - e2(k, j, i)) * idx3;
        mb1(k, j, i) = -c;
      }
      if (i <= ie && k <= ke && (two_d || j == js)) {
        Real c = -(e3(k, j, i + 1) - e3(k, j, i)) * idx1;
        if (three_d) c += (e1(k + 1, j, i) - e1(k, j, i)) * idx3;
        mb2(k, j, i) = -c;
        if (!two_d) mb2(k, j + 1, i) = -c;
      }
      if (i <= ie && j <= je && (three_d || k == ks)) {
        Real c = (e2(k, j, i + 1) - e2(k, j, i)) * idx1;
        if (two_d) c -= (e1(k, j + 1, i) - e1(k, j, i)) * idx2;
        mb3(k, j, i) = -c;
        if (!three_d) mb3(k + 1, j, i) = -c;
      }
    });
}

// out = a*x + b*y + c*z + d*p + e*q  on cells (NVAR) and faces, over the full arrays
inline void Combine(const Mesh &m, Arr4 out, Real a, Arr4 x, Real b, Arr4 y, Real c, Arr4 z,
                    Real d, Arr4 p, Real e, Arr4 q) {
  Kokkos::parallel_for("sts_comb_u", Range3({m.ks, m.js, m.is}, {m.ke + 1, m.je + 1, m.ie + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      for (int n = 0; n < NVAR; ++n)
        out(n, k, j, i) = ((a * x(n, k, j, i) + b * y(n, k, j, i)) + c * z(n, k, j, i)) + (d * p(n, k, j, i) + e * q(n, k, j, i));
    });
}
inline void CombineB(const Mesh &m, Arr3 out, Real a, Arr3 x, Real b, Arr3 y, Real c, Arr3 z,
                     Real d, Arr3 p, Real e, Arr3 q) {
  Kokkos::parallel_for("sts_comb_b", Range3({m.ks, m.js, m.is}, {m.ke + 2, m.je + 2, m.ie + 2}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      out(k, j, i) = ((a * x(k, j, i) + b * y(k, j, i)) + c * z(k, j, i)) + (d * p(k, j, i) + e * q(k, j, i));
    });
}

// Advance the diffusive terms over tau starting from (m.u0, m.b?f0); result left in them.
// Returns the number of stages used.
inline int Advance(Mesh &m, const Config &cfg, const diffusion::Coeffs &dc, Work &wk, Real tau, Real dt_par) {
  wk.Allocate(m);
  const int s = Stages(tau, dt_par);
  const double w1 = 4.0 / (double(s) * s + s - 2.0);
  auto bj = [](int j) { return (j <= 2) ? 1.0 / 3.0 : (double(j) * j + j - 2.0) / (2.0 * j * (j + 1.0)); };

  // Y0 = current state; M0 = M(Y0)
  Kokkos::deep_copy(wk.y0, m.u0);
  Kokkos::deep_copy(wk.b0[0], m.b1f0); Kokkos::deep_copy(wk.b0[1], m.b2f0); Kokkos::deep_copy(wk.b0[2], m.b3f0);
  kernels::ConsToPrim(m, cfg, wk.y0, wk.b0[0], wk.b0[1], wk.b0[2], false);
  Operator(m, cfg, dc, wk.b0[0], wk.b0[1], wk.b0[2], wk.m0, wk.mb0[0], wk.mb0[1], wk.mb0[2]);

  // Y1 = Y0 + mu1~ tau M0,  mu1~ = b1 w1
  const Real mt1 = Real(bj(1) * w1) * tau;
  Combine(m, wk.y1, 1, wk.y0, 0, wk.y0, 0, wk.y0, mt1, wk.m0, 0, wk.m0);
  for (int c = 0; c < 3; ++c) CombineB(m, wk.b1[c], 1, wk.b0[c], 0, wk.b0[c], 0, wk.b0[c], mt1, wk.mb0[c], 0, wk.mb0[c]);
  kernels::Boundaries(m, cfg, wk.y1, wk.b1[0], wk.b1[1], wk.b1[2]);
  // Y_{j-2} = Y0 (shares storage with y0 until j = 3), Y_{j-1} = Y1
  Arr4 yjm2 = wk.y0, yjm1 = wk.y1, yj = wk.y2;
  Arr3 bjm2[3] = {wk.b0[0], wk.b0[1], wk.b0[2]}, bjm1[3] = {wk.b1[0], wk.b1[1], wk.b1[2]}, bjn[3] = {wk.b2[0], wk.b2[1], wk.b2[2]};
  Arr4 spare = m.u1;                       // one more cell array to rotate through
  Arr3 spareb[3] = {m.b1f1, m.b2f1, m.b3f1};

  for (int j = 2; j <= s; ++j) {
    const double b_j = bj(j), b_jm1 = bj(j - 1), b_jm2 = bj(j - 2);
    const double mu = (2.0 * j - 1.0) / j * b_j / b_jm1;
    const double nu = -(j - 1.0) / j * b_j / b_jm2;
    const double mut = mu * w1;
    const double gt = -(1.0 - b_jm1) * mut;
    kernels::ConsToPrim(m, cfg, yjm1, bjm1[0], bjm1[1], bjm1[2], false);
    Operator(m, cfg, dc, bjm1[0], bjm1[1], bjm1[2], wk.mj, wk.mbj[0], wk.mbj[1], wk.mbj[2]);
    // Y_j = mu Y_{j-1} + nu Y_{j-2} + (1 - mu - nu) Y_0 + mut tau M(Y_{j-1}) + gt tau M0
    Combine(m, yj, Real(mu), yjm1, Real(nu), yjm2, Real(1.0 - mu - nu), wk.y0, Real(mut) * tau, wk.mj, Real(gt) * tau, wk.m0);
    for (int c = 0; c < 3; ++c)
      CombineB(m, bjn[c], Real(mu), bjm1[c], Real(nu), bjm2[c], Real(1.0 - mu - nu), wk.b0[c], Real(mut) * tau, wk.mbj[c], Real(gt) * tau, wk.mb0[c]);
    kernels::Boundaries(m, cfg, yj, bjn[0], bjn[1], bjn[2]);
    // rotate: the array that held Y_{j-2} becomes free, unless it is y0 (kept for the recursion)
    Arr4 freed = (j == 2) ? spare : yjm2;
    Arr3 freedb[3] = {(j == 2) ? spareb[0] : bjm2[0], (j == 2) ? spareb[1] : bjm2[1], (j == 2) ? spareb[2] : bjm2[2]};
    yjm2 = yjm1; yjm1 = yj; yj = freed;
    for (int c = 0; c < 3; ++c) { bjm2[c] = bjm1[c]; bjm1[c] = bjn[c]; bjn[c] = freedb[c]; }
  }
  // result: Y_s is in yjm1
  Kokkos::deep_copy(m.u0, yjm1);
  Kokkos::deep_copy(m.b1f0, bjm1[0]); Kokkos::deep_copy(m.b2f0, bjm1[1]); Kokkos::deep_copy(m.b3f0, bjm1[2]);
  return s;
}

}  // namespace sts
