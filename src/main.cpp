// Chasqui MHD — driver.
//
//   chasqui -i input.in [-r restart.rst] [block.key=value ...]
#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "defs.hpp"
#include "params.hpp"
#include "mesh.hpp"
#include "physics.hpp"
#include "diffusion.hpp"
#include "srcterms.hpp"
#include "fofc.hpp"
#include "pgen.hpp"
#include "output.hpp"
#include "restart.hpp"
#include "sts.hpp"
#include "positivity.hpp"

namespace {

BCType ParseBC(const std::string &s) {
  if (s == "periodic") return BCType::periodic;
  if (s == "outflow") return BCType::outflow;
  if (s == "reflect") return BCType::reflect;
  throw std::runtime_error("unknown boundary condition: " + s);
}

struct Physics {
  Config cfg;
  diffusion::Coeffs diff;
  srcterms::Gravity grav;
  srcterms::Cooling cool;
  bool fofc = false;
  bool flux_limiter = false;   // Hu-Adams-Shu positivity flux limiter
  bool sts = false;            // RKL2 super-time-stepping for the diffusive terms
  sts::Work sts_work;
  long fofc_count = 0, limiter_count = 0, sts_stages = 0;
};

// One Runge-Kutta stage: out = g0*u0 + g1*u1 - beta_dt * L(cur), where `cur` is the state
// the primitives were computed from (u0/b0 on stage 1, u1/b1 afterwards).
void Stage(Mesh &m, Physics &ph, bool cur_is_u1, bool out_is_u1, Real g0, Real g1, Real beta_dt) {
  const Config &cfg = ph.cfg;
  Arr3 b1c = cur_is_u1 ? m.b1f1 : m.b1f0;
  Arr3 b2c = cur_is_u1 ? m.b2f1 : m.b2f0;
  Arr3 b3c = cur_is_u1 ? m.b3f1 : m.b3f0;

  kernels::FluxesDispatch<0>(m, cfg, b1c);
  if (m.two_d) kernels::FluxesDispatch<1>(m, cfg, b2c);
  if (m.three_d) kernels::FluxesDispatch<2>(m, cfg, b3c);
  if (!ph.sts) {
    diffusion::Fluxes<0>(m, ph.diff, b1c, b2c, b3c);
    if (m.two_d) diffusion::Fluxes<1>(m, ph.diff, b1c, b2c, b3c);
    if (m.three_d) diffusion::Fluxes<2>(m, ph.diff, b1c, b2c, b3c);
  }
  if (ph.flux_limiter) {
    positivity::LimitFluxes<0>(m, cfg, beta_dt, b1c, &ph.limiter_count);
    if (m.two_d) positivity::LimitFluxes<1>(m, cfg, beta_dt, b2c, &ph.limiter_count);
    if (m.three_d) positivity::LimitFluxes<2>(m, cfg, beta_dt, b3c, &ph.limiter_count);
  }
  kernels::CornerEMF(m);
  if (!ph.sts) diffusion::ResistiveEMF(m, ph.diff, b1c, b2c, b3c);

  kernels::StageArrays s;
  s.ua = m.u0; s.ub = m.u1; s.uout = out_is_u1 ? m.u1 : m.u0;
  s.b1a = m.b1f0; s.b2a = m.b2f0; s.b3a = m.b3f0;
  s.b1b = m.b1f1; s.b2b = m.b2f1; s.b3b = m.b3f1;
  s.b1out = out_is_u1 ? m.b1f1 : m.b1f0;
  s.b2out = out_is_u1 ? m.b2f1 : m.b2f0;
  s.b3out = out_is_u1 ? m.b3f1 : m.b3f0;
  kernels::Update(m, s, g0, g1, beta_dt);
  srcterms::ApplyGravity(m, ph.grav, s.uout, beta_dt);

  if (ph.fofc) {
    const int nbad = fofc::FlagBadCells(m, cfg, s.uout, s.b1out, s.b2out, s.b3out);
    if (nbad > 0) {
      ph.fofc_count += nbad;
      const diffusion::Coeffs dcr = ph.sts ? diffusion::Coeffs{} : ph.diff;
      fofc::RepairFluxes<0>(m, cfg, dcr, b1c, b2c, b3c);
      if (m.two_d) fofc::RepairFluxes<1>(m, cfg, dcr, b1c, b2c, b3c);
      if (m.three_d) fofc::RepairFluxes<2>(m, cfg, dcr, b1c, b2c, b3c);
      kernels::CornerEMF(m);
      if (!ph.sts) diffusion::ResistiveEMF(m, ph.diff, b1c, b2c, b3c);
      kernels::Update(m, s, g0, g1, beta_dt);
      srcterms::ApplyGravity(m, ph.grav, s.uout, beta_dt);
    }
  }
  kernels::Boundaries(m, cfg, s.uout, s.b1out, s.b2out, s.b3out);
}

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  Params pin;
  std::string input, rstfile;
  for (int a = 1; a < argc; ++a) {
    if (std::strcmp(argv[a], "-i") == 0 && a + 1 < argc) { input = argv[++a]; continue; }
    if (std::strcmp(argv[a], "-r") == 0 && a + 1 < argc) { rstfile = argv[++a]; continue; }
  }
  if (input.empty()) { std::cerr << "usage: chasqui -i input.in [-r restart.rst] [block.key=value ...]\n"; return 1; }
  pin.ReadFile(input);
  for (int a = 1; a < argc; ++a) {
    if (std::strcmp(argv[a], "-i") == 0 || std::strcmp(argv[a], "-r") == 0) { ++a; continue; }
    if (std::strchr(argv[a], '=') != nullptr) pin.Override(argv[a]);
  }

  // ---------------- configuration ----------------
  Physics ph;
  Config &cfg = ph.cfg;
  {
    const std::string r = pin.GetOrAddString("mhd", "reconstruct", "ppm");
    cfg.recon = (r == "dc") ? Recon::dc : (r == "plm") ? Recon::plm : (r == "ppm") ? Recon::ppm
              : (r == "ppmx") ? Recon::ppmx : (r == "wenoz") ? Recon::wenoz
              : throw std::runtime_error("unknown reconstruct: " + r);
    const std::string s = pin.GetOrAddString("mhd", "rsolver", "hlld");
    cfg.rsolver = (s == "hlld") ? RSolver::hlld : (s == "lhlld") ? RSolver::lhlld : (s == "hlle") ? RSolver::hlle
                : throw std::runtime_error("unknown rsolver: " + s);
    cfg.characteristic = pin.GetOrAddBool("mhd", "characteristic", false);
    cfg.lhlld_phi = pin.GetOrAddReal("mhd", "lhlld_phi", -1.0);
    cfg.eos.dual_energy = pin.GetOrAddBool("mhd", "dual_energy", false);
    cfg.eos.de_eta = pin.GetOrAddReal("mhd", "de_eta", 1.0e-3);
    const std::string t = pin.GetOrAddString("time", "integrator", "rk3");
    cfg.integ = (t == "rk1") ? Integrator::rk1 : (t == "rk2") ? Integrator::rk2 : (t == "rk3") ? Integrator::rk3
              : (t == "rk4" || t == "ssprk43") ? Integrator::rk4 : throw std::runtime_error("unknown integrator: " + t);
    cfg.low_mach = pin.GetOrAddBool("mhd", "low_mach", false);
    cfg.flatten = pin.GetOrAddBool("mhd", "flatten", true);
    cfg.cfl = pin.GetOrAddReal("time", "cfl", 0.3);
    cfg.eos.gamma = pin.GetOrAddReal("mhd", "gamma", 5.0 / 3.0);
    cfg.eos.gm1 = cfg.eos.gamma - 1.0;
    cfg.eos.dfloor = pin.GetOrAddReal("mhd", "dfloor", 1.0e-10);
    cfg.eos.pfloor = pin.GetOrAddReal("mhd", "pfloor", 1.0e-12);
    cfg.bc1 = ParseBC(pin.GetOrAddString("mesh", "bc_x1", "periodic"));
    cfg.bc2 = ParseBC(pin.GetOrAddString("mesh", "bc_x2", "periodic"));
    cfg.bc3 = ParseBC(pin.GetOrAddString("mesh", "bc_x3", "periodic"));
    ph.fofc = pin.GetOrAddBool("mhd", "fofc", true);
    ph.flux_limiter = pin.GetOrAddBool("mhd", "flux_limiter", false);
    ph.sts = pin.GetOrAddBool("diffusion", "sts", false);
    // non-ideal terms
    ph.diff.nu = pin.GetOrAddReal("diffusion", "nu", 0.0);
    ph.diff.kappa = pin.GetOrAddReal("diffusion", "kappa", 0.0);
    ph.diff.eta = pin.GetOrAddReal("diffusion", "eta", 0.0);
    // gravity
    ph.grav.uniform = pin.GetOrAddBool("gravity", "uniform", false);
    ph.grav.g1 = pin.GetOrAddReal("gravity", "g1", 0.0);
    ph.grav.g2 = pin.GetOrAddReal("gravity", "g2", 0.0);
    ph.grav.g3 = pin.GetOrAddReal("gravity", "g3", 0.0);
    ph.grav.point = pin.GetOrAddBool("gravity", "point_mass", false);
    ph.grav.gm = pin.GetOrAddReal("gravity", "gm", 0.0);
    ph.grav.x0 = pin.GetOrAddReal("gravity", "x0", 0.0);
    ph.grav.y0 = pin.GetOrAddReal("gravity", "y0", 0.0);
    ph.grav.z0 = pin.GetOrAddReal("gravity", "z0", 0.0);
    ph.grav.rsoft = pin.GetOrAddReal("gravity", "rsoft", 0.0);
    // cooling
    ph.cool.enabled = pin.GetOrAddBool("cooling", "enabled", false);
    ph.cool.lambda0 = pin.GetOrAddReal("cooling", "lambda0", 0.0);
    ph.cool.alpha = pin.GetOrAddReal("cooling", "alpha", 1.0);
    ph.cool.tfloor = pin.GetOrAddReal("cooling", "tfloor", 0.0);
  }

  Mesh m;
  m.nx1 = pin.GetInt("mesh", "nx1");
  m.nx2 = pin.GetOrAddInt("mesh", "nx2", 1);
  m.nx3 = pin.GetOrAddInt("mesh", "nx3", 1);
  const int ng_min = (cfg.recon == Recon::ppm || cfg.recon == Recon::ppmx || cfg.recon == Recon::wenoz) ? 3 : 2;
  m.ng = pin.GetOrAddInt("mesh", "nghost", ng_min);
  if (m.ng < ng_min) throw std::runtime_error("nghost too small for the reconstruction method");
  m.x1min = pin.GetOrAddReal("mesh", "x1min", 0.0); m.x1max = pin.GetOrAddReal("mesh", "x1max", 1.0);
  m.x2min = pin.GetOrAddReal("mesh", "x2min", 0.0); m.x2max = pin.GetOrAddReal("mesh", "x2max", 1.0);
  m.x3min = pin.GetOrAddReal("mesh", "x3min", 0.0); m.x3max = pin.GetOrAddReal("mesh", "x3max", 1.0);
  m.Allocate();

  const Real tlim = pin.GetOrAddReal("time", "tlim", 1.0);
  const int nlim = pin.GetOrAddInt("time", "nlim", -1);
  const int ndiag = pin.GetOrAddInt("time", "ndiag", 100);
  const std::string problem = pin.GetString("problem", "name");
  const std::string basename = pin.GetOrAddString("output", "basename", problem);
  const Real dt_vtk = pin.GetOrAddReal("output", "vtk_dt", -1.0);
  const Real dt_hst = pin.GetOrAddReal("output", "hst_dt", -1.0);
  const Real dt_rst = pin.GetOrAddReal("output", "rst_dt", -1.0);
  const bool compute_error = pin.GetOrAddBool("output", "compute_error", false);

  std::cout << "# Chasqui MHD\n";
  pin.Print(std::cout);
  std::cout << "# execution space: " << ExecSpace::name() << "\n";

  // ---------------- initial conditions or restart ----------------
  Real t = 0.0;
  int cycle = 0;
  if (rstfile.empty()) {
    pgen::Initialize(problem, pin, m, cfg);
  } else {
    restart::Read(m, rstfile, t, cycle);
    std::cout << "# restarted from " << rstfile << " at t = " << t << ", cycle = " << cycle << "\n";
  }
  kernels::Boundaries(m, cfg, m.u0, m.b1f0, m.b2f0, m.b3f0);
  Real min_dt = kernels::ConsToPrim(m, cfg, m.u0, m.b1f0, m.b2f0, m.b3f0, true);
  if (!ph.sts) min_dt = fmin(min_dt, diffusion::TimeStep(m, cfg, ph.diff));

  // keep the initial state on the host for error measurements
  auto u_init = Kokkos::create_mirror(Kokkos::HostSpace(), m.u0);
  auto bcc_init = Kokkos::create_mirror(Kokkos::HostSpace(), m.bcc);
  Kokkos::deep_copy(u_init, m.u0);
  Kokkos::deep_copy(bcc_init, m.bcc);

  // ---------------- outputs ----------------
  int nvtk = 0, nrst = 0;
  Real next_vtk = t, next_hst = t, next_rst = t + dt_rst;
  std::ofstream hst;
  if (dt_hst > 0) {
    hst.open(basename + ".hst", rstfile.empty() ? std::ios::out : std::ios::app);
    if (rstfile.empty()) hst << "# time dt mass mom1 mom2 mom3 etot ekin emag eint divB_max divB_L1\n";
  }
  auto do_outputs = [&](Real dt, bool force, bool final = false) {
    const Real eps = 1e-10 * (tlim > 0 ? tlim : 1.0);
    if (dt_vtk > 0 && (t >= next_vtk - eps || force)) {
      char fn[256]; std::snprintf(fn, sizeof(fn), "%s.%04d.vtk", basename.c_str(), nvtk++);
      output::WriteVTK(m, fn, t, cycle);
      next_vtk += dt_vtk;
    }
    if (dt_hst > 0 && (t >= next_hst - eps || force)) {
      auto h = output::History(m, m.u0, m.b1f0, m.b2f0, m.b3f0);
      hst << std::setprecision(12) << t << " " << dt << " " << h.mass << " " << h.mom1 << " " << h.mom2 << " "
          << h.mom3 << " " << h.etot << " " << h.ekin << " " << h.emag << " " << h.eint << " " << h.divb_max
          << " " << h.divb_l1 << "\n";
      hst.flush();
      next_hst += dt_hst;
    }
    if (dt_rst > 0 && (t >= next_rst - eps || final)) {
      char fn[256]; std::snprintf(fn, sizeof(fn), "%s.%04d.rst", basename.c_str(), nrst++);
      restart::Write(m, fn, t, cycle);
      next_rst += dt_rst;
    }
  };
  if (rstfile.empty()) do_outputs(0.0, true);

  // ---------------- time loop ----------------
  const int nstage = static_cast<int>(cfg.integ);
  const int cycle0 = cycle;
  Kokkos::fence();
  auto t0 = std::chrono::steady_clock::now();
  while (t < tlim && (nlim < 0 || cycle - cycle0 < nlim)) {
    Real dt = cfg.cfl * min_dt;
    if (t + dt > tlim) dt = tlim - t;

    for (int s = 1; s <= nstage; ++s) {
      Real g0, g1, beta;
      bool cur_u1, out_u1;
      if (nstage == 1) { g0 = 1; g1 = 0; beta = 1; cur_u1 = false; out_u1 = false; }
      else if (nstage == 2) {
        if (s == 1) { g0 = 1; g1 = 0; beta = 1; cur_u1 = false; out_u1 = true; }
        else        { g0 = 0.5; g1 = 0.5; beta = 0.5; cur_u1 = true; out_u1 = false; }
      } else if (nstage == 3) {
        if (s == 1)      { g0 = 1; g1 = 0; beta = 1; cur_u1 = false; out_u1 = true; }
        else if (s == 2) { g0 = 0.75; g1 = 0.25; beta = 0.25; cur_u1 = true; out_u1 = true; }
        else             { g0 = 1.0 / 3.0; g1 = 2.0 / 3.0; beta = 2.0 / 3.0; cur_u1 = true; out_u1 = false; }
      } else {
        // SSPRK(4,3) of Spiteri & Ruuth (2002): third order, SSP coefficient 2, i.e. stable
        // up to twice the CFL number of SSPRK(3,3) at four stages per step (1.5x cheaper).
        if (s == 1)      { g0 = 1; g1 = 0; beta = 0.5; cur_u1 = false; out_u1 = true; }
        else if (s == 2) { g0 = 0; g1 = 1; beta = 0.5; cur_u1 = true; out_u1 = true; }
        else if (s == 3) { g0 = 2.0 / 3.0; g1 = 1.0 / 3.0; beta = 1.0 / 6.0; cur_u1 = true; out_u1 = true; }
        else             { g0 = 0; g1 = 1; beta = 0.5; cur_u1 = true; out_u1 = false; }
      }
      const bool last = (s == nstage);
      Stage(m, ph, cur_u1, out_u1, g0, g1, beta * dt);
      Arr4 uo = out_u1 ? m.u1 : m.u0;
      Arr3 b1o = out_u1 ? m.b1f1 : m.b1f0, b2o = out_u1 ? m.b2f1 : m.b2f0, b3o = out_u1 ? m.b3f1 : m.b3f0;
      const bool split_ops = last && (ph.sts || ph.cool.enabled);
      const Real mdt = kernels::ConsToPrim(m, cfg, uo, b1o, b2o, b3o, last && !split_ops);
      if (last && !split_ops) min_dt = fmin(mdt, diffusion::TimeStep(m, cfg, ph.diff));
    }
    // operator-split pieces on the end-of-step state (in u0 / b?f0)
    if (ph.sts) {
      ph.sts_stages += sts::Advance(m, cfg, ph.diff, ph.sts_work, dt, diffusion::TimeStep(m, cfg, ph.diff));
      kernels::Boundaries(m, cfg, m.u0, m.b1f0, m.b2f0, m.b3f0);
    }
    if (ph.cool.enabled) {
      kernels::ConsToPrim(m, cfg, m.u0, m.b1f0, m.b2f0, m.b3f0, false);
      srcterms::ApplyCooling(m, cfg.eos, ph.cool, m.u0, dt);
      kernels::Boundaries(m, cfg, m.u0, m.b1f0, m.b2f0, m.b3f0);   // ghosts must see the cooled state
    }
    if (ph.sts || ph.cool.enabled) {
      min_dt = kernels::ConsToPrim(m, cfg, m.u0, m.b1f0, m.b2f0, m.b3f0, true);
      if (!ph.sts) min_dt = fmin(min_dt, diffusion::TimeStep(m, cfg, ph.diff));
    }
    t += dt;
    ++cycle;
    if (ndiag > 0 && cycle % ndiag == 0) {
      std::cout << "cycle=" << cycle << " time=" << std::setprecision(8) << t << " dt=" << dt << "\n";
    }
    do_outputs(dt, false);
  }
  Kokkos::fence();
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (dt_vtk > 0 || dt_hst > 0 || dt_rst > 0) do_outputs(0.0, true, true);

  const double zones = static_cast<double>(m.nx1) * m.nx2 * m.nx3;
  const int ncyc = cycle - cycle0;
  std::cout << "# cycles = " << ncyc << ", time = " << t << ", wall = " << wall << " s\n";
  std::cout << "# zone-cycles/second = " << std::scientific << zones * ncyc / wall << "\n";
  if (ph.fofc) std::cout << "# first-order flux corrections applied: " << ph.fofc_count << "\n";
  if (ph.flux_limiter) std::cout << "# positivity flux limiter activations: " << ph.limiter_count << "\n";
  if (ph.sts) std::cout << "# super-time-stepping stages (total): " << ph.sts_stages << "\n";

  if (compute_error) {
    auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.u0);
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.bcc);
    double err[NVAR + 3] = {0}, tot = 0;
    for (int k = m.ks; k <= m.ke; ++k)
      for (int j = m.js; j <= m.je; ++j)
        for (int i = m.is; i <= m.ie; ++i) {
          for (int n = 0; n < NVAR; ++n) err[n] += std::fabs(hu(n, k, j, i) - u_init(n, k, j, i));
          for (int n = 0; n < 3; ++n) err[NVAR + n] += std::fabs(hb(n, k, j, i) - bcc_init(n, k, j, i));
        }
    std::cout << "# L1 errors vs initial state (rho m1 m2 m3 E rho*s B1 B2 B3 | total):\n# ";
    for (int n = 0; n < NVAR + 3; ++n) { err[n] /= zones; tot += err[n]; std::cout << err[n] << " "; }
    std::cout << "| " << tot << "\n";
    std::ofstream ef(basename + "_errors.dat", std::ios::app);
    ef << m.nx1 << " " << m.nx2 << " " << m.nx3 << " " << std::scientific << std::setprecision(8);
    for (int n = 0; n < NVAR + 3; ++n) ef << err[n] << " ";
    ef << tot << "\n";
  }
  return 0;
}
