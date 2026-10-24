// Chasqui MHD — driver.
//
//   chasqui -i input.in [block.key=value ...]
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
#include "pgen.hpp"
#include "output.hpp"

namespace {

BCType ParseBC(const std::string &s) {
  if (s == "periodic") return BCType::periodic;
  if (s == "outflow") return BCType::outflow;
  throw std::runtime_error("unknown boundary condition: " + s);
}

// One Runge-Kutta stage. `cur` is the state the primitives were computed from (u0/b0 on
// stage 1, u1/b1 after), `out` the array receiving  g0*u0 + g1*u1 - beta*dt*L(cur).
void Stage(Mesh &m, const Config &cfg, bool cur_is_u1, bool out_is_u1,
           Real g0, Real g1, Real beta_dt, bool last) {
  Arr3 b1c = cur_is_u1 ? m.b1f1 : m.b1f0;
  Arr3 b2c = cur_is_u1 ? m.b2f1 : m.b2f0;
  Arr3 b3c = cur_is_u1 ? m.b3f1 : m.b3f0;

  kernels::FluxesDispatch<0>(m, cfg, b1c);
  if (m.two_d) kernels::FluxesDispatch<1>(m, cfg, b2c);
  if (m.three_d) kernels::FluxesDispatch<2>(m, cfg, b3c);
  kernels::CornerEMF(m);

  kernels::StageArrays s;
  s.ua = m.u0; s.ub = m.u1; s.uout = out_is_u1 ? m.u1 : m.u0;
  s.b1a = m.b1f0; s.b2a = m.b2f0; s.b3a = m.b3f0;
  s.b1b = m.b1f1; s.b2b = m.b2f1; s.b3b = m.b3f1;
  s.b1out = out_is_u1 ? m.b1f1 : m.b1f0;
  s.b2out = out_is_u1 ? m.b2f1 : m.b2f0;
  s.b3out = out_is_u1 ? m.b3f1 : m.b3f0;
  kernels::Update(m, s, g0, g1, beta_dt);
  kernels::Boundaries(m, cfg, s.uout, s.b1out, s.b2out, s.b3out);
  (void)last;
}

}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  Params pin;
  std::string input;
  for (int a = 1; a < argc; ++a) {
    if (std::strcmp(argv[a], "-i") == 0 && a + 1 < argc) { input = argv[++a]; continue; }
  }
  if (input.empty()) { std::cerr << "usage: chasqui -i input.in [block.key=value ...]\n"; return 1; }
  pin.ReadFile(input);
  for (int a = 1; a < argc; ++a) {
    if (std::strcmp(argv[a], "-i") == 0) { ++a; continue; }
    if (std::strchr(argv[a], '=') != nullptr) pin.Override(argv[a]);
  }

  // ---------------- configuration ----------------
  Config cfg;
  {
    const std::string r = pin.GetOrAddString("mhd", "reconstruct", "ppm");
    cfg.recon = (r == "dc") ? Recon::dc : (r == "plm") ? Recon::plm : (r == "ppm") ? Recon::ppm
              : (r == "wenoz") ? Recon::wenoz : throw std::runtime_error("unknown reconstruct: " + r);
    const std::string s = pin.GetOrAddString("mhd", "rsolver", "hlld");
    cfg.rsolver = (s == "hlld") ? RSolver::hlld : (s == "hlle") ? RSolver::hlle
                : throw std::runtime_error("unknown rsolver: " + s);
    const std::string t = pin.GetOrAddString("time", "integrator", "rk3");
    cfg.integ = (t == "rk1") ? Integrator::rk1 : (t == "rk2") ? Integrator::rk2 : (t == "rk3") ? Integrator::rk3
              : throw std::runtime_error("unknown integrator: " + t);
    cfg.cfl = pin.GetOrAddReal("time", "cfl", 0.3);
    cfg.eos.gamma = pin.GetOrAddReal("mhd", "gamma", 5.0 / 3.0);
    cfg.eos.gm1 = cfg.eos.gamma - 1.0;
    cfg.eos.dfloor = pin.GetOrAddReal("mhd", "dfloor", 1.0e-10);
    cfg.eos.pfloor = pin.GetOrAddReal("mhd", "pfloor", 1.0e-12);
    cfg.bc1 = ParseBC(pin.GetOrAddString("mesh", "bc_x1", "periodic"));
    cfg.bc2 = ParseBC(pin.GetOrAddString("mesh", "bc_x2", "periodic"));
    cfg.bc3 = ParseBC(pin.GetOrAddString("mesh", "bc_x3", "periodic"));
  }

  Mesh m;
  m.nx1 = pin.GetInt("mesh", "nx1");
  m.nx2 = pin.GetOrAddInt("mesh", "nx2", 1);
  m.nx3 = pin.GetOrAddInt("mesh", "nx3", 1);
  const int ng_min = (cfg.recon == Recon::ppm || cfg.recon == Recon::wenoz) ? 3 : 2;
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
  const bool compute_error = pin.GetOrAddBool("output", "compute_error", false);

  std::cout << "# Chasqui MHD\n";
  pin.Print(std::cout);
  std::cout << "# execution space: " << ExecSpace::name() << "\n";

  // ---------------- initial conditions ----------------
  pgen::Initialize(problem, pin, m, cfg);
  kernels::Boundaries(m, cfg, m.u0, m.b1f0, m.b2f0, m.b3f0);
  Real min_dt = kernels::ConsToPrim(m, cfg, m.u0, m.b1f0, m.b2f0, m.b3f0, true);

  // keep the initial state on the host for error measurements
  // (create_mirror always allocates, so this is a true copy even on host-only builds)
  auto u_init = Kokkos::create_mirror(Kokkos::HostSpace(), m.u0);
  auto bcc_init = Kokkos::create_mirror(Kokkos::HostSpace(), m.bcc);
  Kokkos::deep_copy(u_init, m.u0);
  Kokkos::deep_copy(bcc_init, m.bcc);

  // ---------------- outputs at t = 0 ----------------
  int nvtk = 0, nhst = 0;
  Real t = 0.0, next_vtk = 0.0, next_hst = 0.0;
  std::ofstream hst;
  if (dt_hst > 0) {
    hst.open(basename + ".hst");
    hst << "# time dt mass mom1 mom2 mom3 etot ekin emag eint divB_max divB_L1\n";
  }
  auto do_outputs = [&](Real dt, bool force) {
    if (dt_vtk > 0 && (t >= next_vtk - 1e-12 * tlim || force)) {
      char fn[256]; std::snprintf(fn, sizeof(fn), "%s.%04d.vtk", basename.c_str(), nvtk++);
      output::WriteVTK(m, fn, t, 0);
      next_vtk += dt_vtk;
    }
    if (dt_hst > 0 && (t >= next_hst - 1e-12 * tlim || force)) {
      auto h = output::History(m, m.u0, m.b1f0, m.b2f0, m.b3f0);
      hst << std::setprecision(12) << t << " " << dt << " " << h.mass << " " << h.mom1 << " " << h.mom2 << " "
          << h.mom3 << " " << h.etot << " " << h.ekin << " " << h.emag << " " << h.eint << " " << h.divb_max
          << " " << h.divb_l1 << "\n";
      hst.flush();
      next_hst += dt_hst; ++nhst;
    }
  };
  do_outputs(0.0, true);

  // ---------------- time loop ----------------
  const int nstage = static_cast<int>(cfg.integ);
  int cycle = 0;
  Kokkos::fence();
  auto t0 = std::chrono::steady_clock::now();
  while (t < tlim && (nlim < 0 || cycle < nlim)) {
    Real dt = cfg.cfl * min_dt;
    if (t + dt > tlim) dt = tlim - t;

    for (int s = 1; s <= nstage; ++s) {
      Real g0, g1, beta;
      bool cur_u1, out_u1;
      if (nstage == 1) { g0 = 1; g1 = 0; beta = 1; cur_u1 = false; out_u1 = false; }
      else if (nstage == 2) {
        if (s == 1) { g0 = 1; g1 = 0; beta = 1; cur_u1 = false; out_u1 = true; }
        else        { g0 = 0.5; g1 = 0.5; beta = 0.5; cur_u1 = true; out_u1 = false; }
      } else {
        if (s == 1)      { g0 = 1; g1 = 0; beta = 1; cur_u1 = false; out_u1 = true; }
        else if (s == 2) { g0 = 0.75; g1 = 0.25; beta = 0.25; cur_u1 = true; out_u1 = true; }
        else             { g0 = 1.0 / 3.0; g1 = 2.0 / 3.0; beta = 2.0 / 3.0; cur_u1 = true; out_u1 = false; }
      }
      const bool last = (s == nstage);
      Stage(m, cfg, cur_u1, out_u1, g0, g1, beta * dt, last);
      Arr4 uo = out_u1 ? m.u1 : m.u0;
      Arr3 b1o = out_u1 ? m.b1f1 : m.b1f0, b2o = out_u1 ? m.b2f1 : m.b2f0, b3o = out_u1 ? m.b3f1 : m.b3f0;
      const Real mdt = kernels::ConsToPrim(m, cfg, uo, b1o, b2o, b3o, last);
      if (last) min_dt = mdt;
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
  if (dt_vtk > 0 || dt_hst > 0) do_outputs(0.0, true);

  const double zones = static_cast<double>(m.nx1) * m.nx2 * m.nx3;
  std::cout << "# cycles = " << cycle << ", time = " << t << ", wall = " << wall << " s\n";
  std::cout << "# zone-cycles/second = " << std::scientific << zones * cycle / wall << "\n";

  if (compute_error) {
    auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.u0);
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.bcc);
    double err[NVAR + 3] = {0, 0, 0, 0, 0, 0, 0, 0}, tot = 0;
    for (int k = m.ks; k <= m.ke; ++k)
      for (int j = m.js; j <= m.je; ++j)
        for (int i = m.is; i <= m.ie; ++i) {
          for (int n = 0; n < NVAR; ++n) err[n] += std::fabs(hu(n, k, j, i) - u_init(n, k, j, i));
          for (int n = 0; n < 3; ++n) err[NVAR + n] += std::fabs(hb(n, k, j, i) - bcc_init(n, k, j, i));
        }
    std::cout << "# L1 errors vs initial state (rho m1 m2 m3 E B1 B2 B3 | total):\n# ";
    for (int n = 0; n < NVAR + 3; ++n) { err[n] /= zones; tot += err[n]; std::cout << err[n] << " "; }
    std::cout << "| " << tot << "\n";
    std::ofstream ef(basename + "_errors.dat", std::ios::app);
    ef << m.nx1 << " " << m.nx2 << " " << m.nx3 << " " << std::scientific << std::setprecision(8);
    for (int n = 0; n < NVAR + 3; ++n) ef << err[n] << " ";
    ef << tot << "\n";
  }
  return 0;
}
