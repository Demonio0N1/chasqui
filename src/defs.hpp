// Chasqui MHD — basic definitions
// Copyright (c) 2026 Gary Leonel Guamán Toasa and contributors. BSD 3-Clause License.
#pragma once
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdint>

#ifdef CHASQUI_SINGLE_PRECISION
using Real = float;
#else
using Real = double;
#endif

// Conserved variable indices
// IENT: conserved rho*s with s = p / rho^gamma (dual-energy entropy variable)
enum : int { IDN = 0, IM1 = 1, IM2 = 2, IM3 = 3, IEN = 4, IENT = 5, NVAR = 6 };
// Primitive variable indices (same slots; velocity and pressure replace momentum/energy)
enum : int { IVX = 1, IVY = 2, IVZ = 3, IPR = 4 };
// Magnetic field components
enum : int { IBX = 0, IBY = 1, IBZ = 2 };

// All arrays are LayoutRight so the last index (i) is the fastest everywhere,
// which gives contiguous vector loops on CPU and coalesced access on GPU.
using Layout = Kokkos::LayoutRight;
using Arr3 = Kokkos::View<Real***,  Layout>;
using Arr4 = Kokkos::View<Real****, Layout>;
using Arr1 = Kokkos::View<Real*,    Layout>;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using Range3 = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;

KOKKOS_INLINE_FUNCTION Real SQR(Real x) { return x * x; }
KOKKOS_INLINE_FUNCTION Real SIGN(Real x) { return (x > Real(0)) ? Real(1) : ((x < Real(0)) ? Real(-1) : Real(0)); }

enum class Recon : int { dc = 0, plm = 1, ppm = 2, wenoz = 3, ppmx = 4 };
enum class Integrator : int { rk1 = 1, rk2 = 2, rk3 = 3, rk4 = 4 };   // rk4 = SSPRK(4,3), CFL coefficient 2
enum class BCType : int { periodic = 0, outflow = 1, reflect = 2 };

struct EOS {
  Real gamma;
  Real gm1;
  Real dfloor;
  Real pfloor;
  bool dual_energy = false;   // use the entropy variable where the total-energy pressure is unreliable
  Real de_eta = Real(1e-3);   // threshold: e_int / E_total below which entropy pressure is used
};
