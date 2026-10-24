// Chasqui MHD — Riemann solvers for ideal MHD on a single face.
//
// States are given in a frame where u is the face-normal velocity, (v, w) the transverse
// velocities, bn the (common) face-normal field and (by, bz) the transverse fields.
// Units: magnetic pressure = B^2/2 (mu_0 = 1, Lorentz-Heaviside), same as Athena and PLUTO.
//
// HLLD: Miyoshi & Kusano, JCP 208, 315 (2005). Written from the published equations.
#pragma once
#include "defs.hpp"

struct State {
  Real d, u, v, w, p, by, bz;
};

// Flux ordering: [0]=mass [1]=normal momentum [2],[3]=transverse momenta [4]=energy
//                [5]=by  [6]=bz
enum : int { FD = 0, FMU = 1, FMV = 2, FMW = 3, FEN = 4, FBY = 5, FBZ = 6, NFLX = 7 };

KOKKOS_INLINE_FUNCTION Real FastSpeed(const State &s, Real bn, Real gamma) {
  const Real bsq = bn * bn + s.by * s.by + s.bz * s.bz;
  const Real gp = gamma * s.p;
  const Real a = (gp + bsq) / s.d;
  const Real disc = a * a - Real(4) * gp * bn * bn / (s.d * s.d);
  return sqrt(Real(0.5) * (a + sqrt(fmax(disc, Real(0)))));
}

KOKKOS_INLINE_FUNCTION void PhysFlux(const State &s, Real bn, Real gm1, Real &e, Real &pt, Real f[NFLX]) {
  const Real vsq = s.u * s.u + s.v * s.v + s.w * s.w;
  const Real bsq = bn * bn + s.by * s.by + s.bz * s.bz;
  e = s.p / gm1 + Real(0.5) * (s.d * vsq + bsq);
  pt = s.p + Real(0.5) * bsq;
  const Real vdotb = s.u * bn + s.v * s.by + s.w * s.bz;
  f[FD] = s.d * s.u;
  f[FMU] = s.d * s.u * s.u + pt - bn * bn;
  f[FMV] = s.d * s.u * s.v - bn * s.by;
  f[FMW] = s.d * s.u * s.w - bn * s.bz;
  f[FEN] = (e + pt) * s.u - bn * vdotb;
  f[FBY] = s.by * s.u - bn * s.v;
  f[FBZ] = s.bz * s.u - bn * s.w;
}

// ---------------------------------------------------------------------------------------
// HLLE (Einfeldt 1988) — robust fallback
KOKKOS_INLINE_FUNCTION void HLLE(const State &L, const State &R, Real bn, const EOS &eos, Real f[NFLX]) {
  Real fl[NFLX], fr[NFLX], el, er, ptl, ptr;
  PhysFlux(L, bn, eos.gm1, el, ptl, fl);
  PhysFlux(R, bn, eos.gm1, er, ptr, fr);
  const Real cl = FastSpeed(L, bn, eos.gamma), cr = FastSpeed(R, bn, eos.gamma);
  const Real sl = fmin(L.u - cl, R.u - cr);
  const Real sr = fmax(L.u + cl, R.u + cr);
  if (sl >= Real(0)) { for (int n = 0; n < NFLX; ++n) f[n] = fl[n]; return; }
  if (sr <= Real(0)) { for (int n = 0; n < NFLX; ++n) f[n] = fr[n]; return; }
  const Real ul[NFLX] = {L.d, L.d * L.u, L.d * L.v, L.d * L.w, el, L.by, L.bz};
  const Real ur[NFLX] = {R.d, R.d * R.u, R.d * R.v, R.d * R.w, er, R.by, R.bz};
  const Real inv = Real(1) / (sr - sl);
  for (int n = 0; n < NFLX; ++n) f[n] = (sr * fl[n] - sl * fr[n] + sl * sr * (ur[n] - ul[n])) * inv;
}

// ---------------------------------------------------------------------------------------
// HLLD (Miyoshi & Kusano 2005)
KOKKOS_INLINE_FUNCTION void HLLD(const State &L, const State &R, Real bn, const EOS &eos, Real f[NFLX]) {
  Real fl[NFLX], fr[NFLX], el, er, ptl, ptr;
  PhysFlux(L, bn, eos.gm1, el, ptl, fl);
  PhysFlux(R, bn, eos.gm1, er, ptr, fr);

  const Real cl = FastSpeed(L, bn, eos.gamma), cr = FastSpeed(R, bn, eos.gamma);
  const Real sl = fmin(L.u - cl, R.u - cr);
  const Real sr = fmax(L.u + cl, R.u + cr);

  if (sl >= Real(0)) { for (int n = 0; n < NFLX; ++n) f[n] = fl[n]; return; }
  if (sr <= Real(0)) { for (int n = 0; n < NFLX; ++n) f[n] = fr[n]; return; }

  const Real bnsq = bn * bn;
  const Real slml = sl - L.u;   // (S_L - u_L) < 0
  const Real srmr = sr - R.u;   // (S_R - u_R) > 0
  const Real dsl = L.d * slml;  // rho_L (S_L - u_L)
  const Real dsr = R.d * srmr;  // rho_R (S_R - u_R)

  // contact speed and total pressure in the star regions (MK05 eqs. 38, 41)
  const Real sm = (dsr * R.u - dsl * L.u - ptr + ptl) / (dsr - dsl);
  const Real pts = (dsr * ptl - dsl * ptr + L.d * R.d * srmr * slml * (R.u - L.u)) / (dsr - dsl);

  // ---- left star state (MK05 eqs. 43-47) ----
  const Real slms = sl - sm;
  const Real dls = dsl / slms;                       // rho*_L
  Real vls, wls, byls, bzls;
  {
    const Real den = L.d * slml * slms - bnsq;
    const Real scale = fabs(L.d * slml * slms) + bnsq;
    if (fabs(den) <= Real(1e-12) * scale + Real(1e-300)) {
      vls = L.v; wls = L.w; byls = L.by; bzls = L.bz;  // degenerate: no jump across S_L
    } else {
      const Real num_v = bn * (sm - L.u) / den;
      vls = L.v - L.by * num_v;
      wls = L.w - L.bz * num_v;
      const Real num_b = (L.d * slml * slml - bnsq) / den;
      byls = L.by * num_b;
      bzls = L.bz * num_b;
    }
  }
  const Real vdotb_l = L.u * bn + L.v * L.by + L.w * L.bz;
  const Real vdotb_ls = sm * bn + vls * byls + wls * bzls;
  const Real els = (slml * el - ptl * L.u + pts * sm + bn * (vdotb_l - vdotb_ls)) / slms;

  // ---- right star state ----
  const Real srms = sr - sm;
  const Real drs = dsr / srms;
  Real vrs, wrs, byrs, bzrs;
  {
    const Real den = R.d * srmr * srms - bnsq;
    const Real scale = fabs(R.d * srmr * srms) + bnsq;
    if (fabs(den) <= Real(1e-12) * scale + Real(1e-300)) {
      vrs = R.v; wrs = R.w; byrs = R.by; bzrs = R.bz;
    } else {
      const Real num_v = bn * (sm - R.u) / den;
      vrs = R.v - R.by * num_v;
      wrs = R.w - R.bz * num_v;
      const Real num_b = (R.d * srmr * srmr - bnsq) / den;
      byrs = R.by * num_b;
      bzrs = R.bz * num_b;
    }
  }
  const Real vdotb_r = R.u * bn + R.v * R.by + R.w * R.bz;
  const Real vdotb_rs = sm * bn + vrs * byrs + wrs * bzrs;
  const Real ers = (srmr * er - ptr * R.u + pts * sm + bn * (vdotb_r - vdotb_rs)) / srms;

  // Alfven speeds bounding the double-star region (MK05 eq. 51)
  const Real sqdl = sqrt(dls), sqdr = sqrt(drs);
  const Real abn = fabs(bn);
  const Real sls = sm - abn / sqdl;
  const Real srs = sm + abn / sqdr;

  // conserved star states
  const Real uls[NFLX] = {dls, dls * sm, dls * vls, dls * wls, els, byls, bzls};
  const Real urs[NFLX] = {drs, drs * sm, drs * vrs, drs * wrs, ers, byrs, bzrs};
  const Real ul[NFLX] = {L.d, L.d * L.u, L.d * L.v, L.d * L.w, el, L.by, L.bz};
  const Real ur[NFLX] = {R.d, R.d * R.u, R.d * R.v, R.d * R.w, er, R.by, R.bz};

  if (sls >= Real(0)) {
    for (int n = 0; n < NFLX; ++n) f[n] = fl[n] + sl * (uls[n] - ul[n]);
    return;
  }
  if (srs <= Real(0)) {
    for (int n = 0; n < NFLX; ++n) f[n] = fr[n] + sr * (urs[n] - ur[n]);
    return;
  }

  // ---- double-star states (MK05 eqs. 59-63) ----
  // If bn is (numerically) zero the double-star region collapses onto the star states.
  const Real bnsmall = Real(1e-12) * (ptl + ptr);
  Real vss, wss, byss, bzss;
  if (bnsq <= bnsmall) {
    // treat as a single contact: pick the side by S_M
    if (sm >= Real(0)) {
      for (int n = 0; n < NFLX; ++n) f[n] = fl[n] + sl * (uls[n] - ul[n]);
    } else {
      for (int n = 0; n < NFLX; ++n) f[n] = fr[n] + sr * (urs[n] - ur[n]);
    }
    return;
  }
  const Real sgn = SIGN(bn);
  const Real inv = Real(1) / (sqdl + sqdr);
  vss = (sqdl * vls + sqdr * vrs + (byrs - byls) * sgn) * inv;
  wss = (sqdl * wls + sqdr * wrs + (bzrs - bzls) * sgn) * inv;
  byss = (sqdl * byrs + sqdr * byls + sqdl * sqdr * (vrs - vls) * sgn) * inv;
  bzss = (sqdl * bzrs + sqdr * bzls + sqdl * sqdr * (wrs - wls) * sgn) * inv;
  const Real vdotb_ss = sm * bn + vss * byss + wss * bzss;

  if (sm >= Real(0)) {
    const Real elss = els - sqdl * (vdotb_ls - vdotb_ss) * sgn;
    const Real ulss[NFLX] = {dls, dls * sm, dls * vss, dls * wss, elss, byss, bzss};
    for (int n = 0; n < NFLX; ++n)
      f[n] = fl[n] + sl * (uls[n] - ul[n]) + sls * (ulss[n] - uls[n]);
  } else {
    const Real erss = ers + sqdr * (vdotb_rs - vdotb_ss) * sgn;
    const Real urss[NFLX] = {drs, drs * sm, drs * vss, drs * wss, erss, byss, bzss};
    for (int n = 0; n < NFLX; ++n)
      f[n] = fr[n] + sr * (urs[n] - ur[n]) + srs * (urss[n] - urs[n]);
  }
}
