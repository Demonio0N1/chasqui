// Chasqui MHD — flux-limiting positivity preservation (Hu, Adams & Shu 2013, JCP 242, 169),
// adapted to the MHD face fluxes: every high-order flux is blended toward the first-order
// Lax-Friedrichs flux, F = F_LF + theta (F_HO - F_LF), with the largest theta in [0,1]
// for which the two one-sided half-updates
//   U_L^- = U_L - 2 ndim (dt/dx) (F - F(U_L)),   U_R^+ = U_R + 2 ndim (dt/dx) (F - F(U_R))
// keep rho >= rho_floor and p >= p_floor. The LF flux gives positive half-updates for
// 2 ndim (dt/dx) alpha <= 1 (Euler), so theta always exists; for MHD the magnetic field is
// held fixed in the pressure check (the CT update is not limited).
#pragma once
#include "defs.hpp"
#include "mesh.hpp"
#include "physics.hpp"

namespace positivity {

// physical flux of a cell state (rotated frame) for the 5 hydro variables + entropy
KOKKOS_INLINE_FUNCTION void CellFlux(const State &s, Real bn, Real sent, Real gm1, Real fc[6], Real uc[6], Real &cf, Real gamma) {
  Real f[NFLX], e, pt;
  PhysFlux(s, bn, gm1, e, pt, f);
  fc[0] = f[FD]; fc[1] = f[FMU]; fc[2] = f[FMV]; fc[3] = f[FMW]; fc[4] = f[FEN]; fc[5] = f[FD] * sent;
  uc[0] = s.d; uc[1] = s.d * s.u; uc[2] = s.d * s.v; uc[3] = s.d * s.w; uc[4] = e; uc[5] = s.d * sent;
  cf = FastSpeed(s, bn, gamma);
}

template <int DIR>
void LimitFluxes(const Mesh &m, const Config &cfg, Real beta_dt, Arr3 /*bn*/, long *count) {
  const EOS eos = cfg.eos;
  constexpr int di = (DIR == 0) ? 1 : 0, dj = (DIR == 1) ? 1 : 0, dk = (DIR == 2) ? 1 : 0;
  constexpr int IVU = IVX + DIR, IVV = IVX + (DIR + 1) % 3, IVW = IVX + (DIR + 2) % 3;
  constexpr int IBT1 = (DIR + 1) % 3, IBT2 = (DIR + 2) % 3;
  const int is = m.is, ie = m.ie, js = m.js, je = m.je, ks = m.ks, ke = m.ke;
  const int iu = (DIR == 0) ? ie + 1 : ie, ju = (DIR == 1) ? je + 1 : je, ku = (DIR == 2) ? ke + 1 : ke;
  const int ndim = 1 + (m.two_d ? 1 : 0) + (m.three_d ? 1 : 0);
  const Real dx = (DIR == 0) ? m.dx1 : (DIR == 1) ? m.dx2 : m.dx3;
  const Real lam2 = Real(2) * ndim * beta_dt / dx;
  auto w = m.w, bcc = m.bcc;
  Arr4 flx = (DIR == 0) ? m.f1 : (DIR == 1) ? m.f2 : m.f3;
  const int nv[6] = {IDN, IVU, IVV, IVW, IEN, IENT};   // conserved slots in the rotated order
  long nlim = 0;
  Kokkos::parallel_reduce("pos_limit", Range3({ks, js, is}, {ku + 1, ju + 1, iu + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i, long &cnt) {
      const int kL = k - dk, jL = j - dj, iL = i - di;
      State L{w(IDN, kL, jL, iL), w(IVU, kL, jL, iL), w(IVV, kL, jL, iL), w(IVW, kL, jL, iL), w(IPR, kL, jL, iL),
              bcc(IBT1, kL, jL, iL), bcc(IBT2, kL, jL, iL)};
      State R{w(IDN, k, j, i), w(IVU, k, j, i), w(IVV, k, j, i), w(IVW, k, j, i), w(IPR, k, j, i),
              bcc(IBT1, k, j, i), bcc(IBT2, k, j, i)};
      const Real bnL = bcc(DIR, kL, jL, iL), bnR = bcc(DIR, k, j, i);
      Real fL[6], fR[6], uL[6], uR[6], cfL, cfR;
      CellFlux(L, bnL, w(IENT, kL, jL, iL), eos.gm1, fL, uL, cfL, eos.gamma);
      CellFlux(R, bnR, w(IENT, k, j, i), eos.gm1, fR, uR, cfR, eos.gamma);
      const Real alpha = fmax(fabs(L.u) + cfL, fabs(R.u) + cfR);
      Real fho[6], flf[6];
      for (int n = 0; n < 6; ++n) {
        fho[n] = flx(nv[n], k, j, i);
        flf[n] = Real(0.5) * (fL[n] + fR[n]) - Real(0.5) * (alpha * (uR[n] - uL[n]));
      }
      // magnetic energy of each cell (fixed during the check)
      const Real emL = Real(0.5) * (bnL * bnL + L.by * L.by + L.bz * L.bz);
      const Real emR = Real(0.5) * (bnR * bnR + R.by * R.by + R.bz * R.bz);
      auto ok = [&](Real th) {
        Real f[6];
        for (int n = 0; n < 6; ++n) f[n] = flf[n] + th * (fho[n] - flf[n]);
        Real a[5], b[5];
        for (int n = 0; n < 5; ++n) { a[n] = uL[n] - lam2 * (f[n] - fL[n]); b[n] = uR[n] + lam2 * (f[n] - fR[n]); }
        if (a[0] < eos.dfloor || b[0] < eos.dfloor) return false;
        const Real ea = a[4] - Real(0.5) * (a[1] * a[1] + a[2] * a[2] + a[3] * a[3]) / a[0] - emL;
        const Real eb = b[4] - Real(0.5) * (b[1] * b[1] + b[2] * b[2] + b[3] * b[3]) / b[0] - emR;
        Real pa = eos.gm1 * ea, pb = eos.gm1 * eb;
        if (eos.dual_energy) {   // same switch as ConsToPrim: entropy pressure where E is unreliable
          const Real sa = (uL[5] - lam2 * (f[5] - fL[5])) / a[0], sb = (uR[5] + lam2 * (f[5] - fR[5])) / b[0];
          if (ea < eos.de_eta * fabs(a[4])) pa = sa * pow(a[0], eos.gamma);
          if (eb < eos.de_eta * fabs(b[4])) pb = sb * pow(b[0], eos.gamma);
        }
        return (pa >= eos.pfloor && pb >= eos.pfloor);
      };
      if (ok(Real(1))) return;      // the high-order flux is already safe (the usual case)
      // bisection for the largest admissible theta in [0, 1]
      Real lo = Real(0), hi = Real(1);
      for (int it = 0; it < 12; ++it) {
        const Real mid = Real(0.5) * (lo + hi);
        if (ok(mid)) lo = mid; else hi = mid;
      }
      const Real th = lo;
      for (int n = 0; n < 6; ++n) flx(nv[n], k, j, i) = flf[n] + th * (fho[n] - flf[n]);
      ++cnt;
    }, nlim);
  *count += nlim;
}

}  // namespace positivity
