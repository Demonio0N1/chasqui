#!/usr/bin/env bash
# Convergence test: circularly polarized Alfven wave, 1D and 2D oblique, three reconstructions.
# Usage: tests/convergence.sh <path/to/chasqui binary>
set -euo pipefail
BIN=${1:-./build/chasqui}
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mktemp -d)
cd "$OUT"
order () { awk 'BEGIN{p=0}{o=(p>0)?log(p/$NF)/log(2):0; printf "  %5d  %.3e  %5.2f\n",$1,$NF,o; p=$NF}' "$1"; }
for r in plm ppm wenoz; do
  rm -f cp1_errors.dat
  for n in 32 64 128 256; do "$BIN" -i "$HERE/inputs/cpaw_1d.in" mesh.nx1=$n mhd.reconstruct=$r time.ndiag=0 output.basename=cp1 >/dev/null; done
  echo "== CPAW 1D, $r + HLLD + RK3   (N, L1 total, orden)"; order cp1_errors.dat
done
for r in plm wenoz; do
  rm -f cp2_errors.dat
  for n in 32 64 128; do "$BIN" -i "$HERE/inputs/cpaw_2d.in" mesh.nx1=$n mesh.nx2=$n mhd.reconstruct=$r time.ndiag=0 output.basename=cp2 >/dev/null; done
  echo "== CPAW 2D oblicua 45 grados, $r + HLLD + RK3"; order cp2_errors.dat
done
rm -rf "$OUT"
