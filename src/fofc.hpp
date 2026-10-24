// Chasqui MHD — first-order flux correction (safety net for strong shocks / low beta).
//
// After a stage update, cells whose new density or pressure fell below the floors are
// flagged; the fluxes (and face EMFs) on all faces touching a flagged cell are recomputed
// with donor-cell reconstruction + HLLE, the diffusive fluxes are re-added on those faces,
// the corner EMFs are rebuilt and the stage update is repeated. A first-order flux is
// positivity preserving, so this removes the failure without an ad-hoc floor on the state.
#pragma once
#include "defs.hpp"
#include "mesh.hpp"
#include "physics.hpp"
#include "diffusion.hpp"

namespace fofc {

// Flag cells with rho <= dfloor or p <= pfloor in the tentative state. Returns the count.
inline int FlagBadCells(const Mesh &m, const Config &cfg, Arr4 u, Arr3 b1, Arr3 b2, Arr3 b3) {
  const EOS eos = cfg.eos;
  const bool two_d = m.two_d, three_d = m.three_d;
  auto flag = m.flag;
  int nbad = 0;
  Kokkos::parallel_reduce("fofc_flag", Range3({m.ks, m.js, m.is}, {m.ke + 1, m.je + 1, m.ie + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i, int &n) {
      const Real bx = Real(0.5) * (b1(k, j, i) + b1(k, j, i + 1));
      const Real by = two_d ? Real(0.5) * (b2(k, j, i) + b2(k, j + 1, i)) : b2(k, j, i);
      const Real bz = three_d ? Real(0.5) * (b3(k, j, i) + b3(k + 1, j, i)) : b3(k, j, i);
      const Real d = u(IDN, k, j, i);
      bool bad = (d <= eos.dfloor);
      if (!bad) {
        const Real ke = Real(0.5) * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) / d;
        const Real eint = u(IEN, k, j, i) - ke - Real(0.5) * (bx * bx + by * by + bz * bz);
        Real p = eos.gm1 * eint;
        // with dual energy the total-energy pressure is not the one that will be used where
        // the internal energy is a tiny fraction of the total: check the entropy pressure there
        if (eos.dual_energy && eint < eos.de_eta * fabs(u(IEN, k, j, i)))
          p = (u(IENT, k, j, i) / d) * pow(d, eos.gamma);
        bad = (p <= eos.pfloor);
      }
      flag(k, j, i) = bad ? 1 : 0;
      if (bad) ++n;
    }, nbad);
  return nbad;
}

// Recompute first-order fluxes on faces adjacent to flagged cells (direction DIR).
template <int DIR>
void RepairFluxes(const Mesh &m, const Config &cfg, const diffusion::Coeffs &dc,
                  Arr3 b1c, Arr3 b2c, Arr3 b3c) {
  const EOS eos = cfg.eos;
  constexpr int di = (DIR == 0) ? 1 : 0, dj = (DIR == 1) ? 1 : 0, dk = (DIR == 2) ? 1 : 0;
  constexpr int IVU = IVX + DIR, IVV = IVX + (DIR + 1) % 3, IVW = IVX + (DIR + 2) % 3;
  constexpr int IBT1 = (DIR + 1) % 3, IBT2 = (DIR + 2) % 3;
  int il = m.is, iu = m.ie, jl = m.js, ju = m.je, kl = m.ks, ku = m.ke;
  if (DIR == 0) iu = m.ie + 1; else if (m.nx1 > 1) { il = m.is - 1; iu = m.ie + 1; }
  if (DIR == 1) ju = m.je + 1; else if (m.two_d) { jl = m.js - 1; ju = m.je + 1; }
  if (DIR == 2) ku = m.ke + 1; else if (m.three_d) { kl = m.ks - 1; ku = m.ke + 1; }
  const int is = m.is, ie = m.ie, js = m.js, je = m.je, ks = m.ks, ke = m.ke;
  auto w = m.w, bcc = m.bcc;
  auto flag = m.flag;
  Arr3 bn = (DIR == 0) ? b1c : (DIR == 1) ? b2c : b3c;
  Arr4 flx = (DIR == 0) ? m.f1 : (DIR == 1) ? m.f2 : m.f3;
  Arr3 et2 = (DIR == 0) ? m.e3x1 : (DIR == 1) ? m.e1x2 : m.e2x3;
  Arr3 et1 = (DIR == 0) ? m.e2x1 : (DIR == 1) ? m.e3x2 : m.e1x3;
  const bool diff = dc.any();
  const diffusion::FaceCtx<DIR> ctx(m, dc, b1c, b2c, b3c);

  Kokkos::parallel_for("fofc_repair", Range3({kl, jl, il}, {ku + 1, ju + 1, iu + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const int kL = k - dk, jL = j - dj, iL = i - di;
      auto inside = [&](int kk, int jj, int ii) {
        return (ii >= is && ii <= ie && jj >= js && jj <= je && kk >= ks && kk <= ke);
      };
      const bool fl = inside(k, j, i) && flag(k, j, i);
      const bool fr = inside(kL, jL, iL) && flag(kL, jL, iL);
      if (!(fl || fr)) return;
      State L, R;
      L.d = w(IDN, kL, jL, iL); L.u = w(IVU, kL, jL, iL); L.v = w(IVV, kL, jL, iL); L.w = w(IVW, kL, jL, iL);
      L.p = w(IPR, kL, jL, iL); L.by = bcc(IBT1, kL, jL, iL); L.bz = bcc(IBT2, kL, jL, iL);
      R.d = w(IDN, k, j, i); R.u = w(IVU, k, j, i); R.v = w(IVV, k, j, i); R.w = w(IVW, k, j, i);
      R.p = w(IPR, k, j, i); R.by = bcc(IBT1, k, j, i); R.bz = bcc(IBT2, k, j, i);
      Real f[NFLX];
      HLLE(L, R, bn(k, j, i), eos, f);
      flx(IDN, k, j, i) = f[FD]; flx(IVU, k, j, i) = f[FMU]; flx(IVV, k, j, i) = f[FMV];
      flx(IVW, k, j, i) = f[FMW]; flx(IEN, k, j, i) = f[FEN];
      flx(IENT, k, j, i) = (f[FD] > Real(0)) ? f[FD] * w(IENT, kL, jL, iL) : (f[FD] < Real(0)) ? f[FD] * w(IENT, k, j, i)
                                             : Real(0.5) * (f[FD] * (w(IENT, kL, jL, iL) + w(IENT, k, j, i)));
      et2(k, j, i) = -f[FBY];
      et1(k, j, i) = f[FBZ];
      // diffusive fluxes exist only on the faces of active cells (same range as diffusion::Fluxes)
      const bool af = (DIR == 0) ? (i >= is && i <= ie + 1 && j >= js && j <= je && k >= ks && k <= ke)
                    : (DIR == 1) ? (j >= js && j <= je + 1 && i >= is && i <= ie && k >= ks && k <= ke)
                                 : (k >= ks && k <= ke + 1 && i >= is && i <= ie && j >= js && j <= je);
      if (diff && af) ctx.Add(k, j, i);
    });
}

}  // namespace fofc
