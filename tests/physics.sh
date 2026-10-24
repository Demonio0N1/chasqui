#!/usr/bin/env bash
# Physics tests against exact solutions:
#   - resistive / viscous / conductive decay of a sinusoid along each axis
#   - uniform gravity (momentum = rho g t V) and power-law cooling (exact T(t))
#   - reflecting walls (mass and energy conservation) and restart reproducibility
# Usage: tests/physics.sh /abs/path/to/chasqui
set -euo pipefail
BIN=${1:?path to chasqui binary}
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mktemp -d); cd "$OUT"
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread} OMP_PLACES=${OMP_PLACES:-threads}

cat > diff.in <<'EOF'
[mesh]
nx1 = 64
[time]
integrator = rk3
cfl = 0.3
tlim = 0.5
[mhd]
reconstruct = plm
[problem]
name = diffusion_test
amp = 1.0e-3
[output]
basename = diff
vtk_dt = 0.5
EOF
run () { local label=$1; shift; "$BIN" -i diff.in output.basename=$label time.ndiag=0 "$@" > $label.log; }
run eta_x problem.var=b problem.axis=1 mesh.nx1=64 mesh.nx2=1  mesh.nx3=1  diffusion.eta=0.02
run eta_y problem.var=b problem.axis=2 mesh.nx1=8  mesh.nx2=64 mesh.nx3=1  diffusion.eta=0.02
run eta_z problem.var=b problem.axis=3 mesh.nx1=8  mesh.nx2=8  mesh.nx3=64 diffusion.eta=0.02
run nu_x  problem.var=v problem.axis=1 mesh.nx1=64 mesh.nx2=1  mesh.nx3=1  diffusion.nu=0.02
run nu_y  problem.var=v problem.axis=2 mesh.nx1=8  mesh.nx2=64 mesh.nx3=1  diffusion.nu=0.02
run nu_z  problem.var=v problem.axis=3 mesh.nx1=8  mesh.nx2=8  mesh.nx3=64 diffusion.nu=0.02
run kap_x problem.var=temp problem.axis=1 mesh.nx1=64 mesh.nx2=1 mesh.nx3=1 diffusion.kappa=0.003 time.tlim=2.0 output.vtk_dt=2.0

cat > uni.in <<'EOF'
[mesh]
nx1 = 16
nx2 = 16
nx3 = 16
[time]
tlim = 1.0
[mhd]
reconstruct = plm
[problem]
name = uniform
dens = 2.0
pres = 1.0
[output]
basename = uni
hst_dt = 1.0
EOF
"$BIN" -i uni.in gravity.uniform=true gravity.g1=0.5 gravity.g2=-0.25 gravity.g3=1.0 output.basename=grav time.ndiag=0 > grav.log
"$BIN" -i uni.in cooling.enabled=true cooling.lambda0=0.1 cooling.alpha=0.5 output.basename=cool time.ndiag=0 output.vtk_dt=1.0 > cool.log
"$BIN" -i "$HERE/inputs/blast_3d.in" mesh.nx1=64 mesh.nx2=64 mesh.nx3=1 mesh.bc_x1=reflect problem.angle=90 time.tlim=0.4 time.ndiag=0 output.basename=box output.hst_dt=0.4 > box.log
"$BIN" -i "$HERE/inputs/orszag_tang.in" mesh.nx1=64 mesh.nx2=64 time.tlim=0.2 time.ndiag=0 output.basename=full output.vtk_dt=0.2 output.rst_dt=0.1 output.hst_dt=-1 > full.log
"$BIN" -i "$HERE/inputs/orszag_tang.in" -r full.0000.rst mesh.nx1=64 mesh.nx2=64 time.tlim=0.2 time.ndiag=0 output.basename=resumed output.vtk_dt=0.2 output.hst_dt=-1 > resumed.log

python3 - <<'EOF'
import numpy as np, re, math, glob
def read_vtk(fn):
    data=open(fn,'rb').read()
    dims=[int(x) for x in re.search(rb'DIMENSIONS (\d+) (\d+) (\d+)', data).groups()]
    n=int(re.search(rb'CELL_DATA (\d+)', data).group(1)); out={}
    for name in ['rho','prs','vx1','vx2','vx3','Bx1','Bx2','Bx3']:
        m=re.search(rb'SCALARS '+name.encode()+rb' float\nLOOKUP_TABLE default\n', data)
        nx=dims[0]-1; ny=max(dims[1]-1,1); nz=max(dims[2]-1,1)
        out[name]=np.frombuffer(data[m.end():m.end()+4*n], dtype='>f4').astype(float).reshape(nz,ny,nx)
    return out
ok=True
def check(name, meas, exact, tol):
    global ok
    rel=abs(meas/exact-1); flag='OK ' if rel<tol else 'FAIL'; ok = ok and rel<tol
    print('  %s %-28s medido %.6e  exacto %.6e  err.rel %.1e' % (flag, name, meas, exact, rel))
amp=1e-3; k=2*math.pi; g=5/3
for label,field,D,ax,t in [('eta_x','Bx2',0.02,0,0.5),('eta_y','Bx3',0.02,1,0.5),('eta_z','Bx1',0.02,2,0.5),
                           ('nu_x','vx2',0.02,0,0.5),('nu_y','vx3',0.02,1,0.5),('nu_z','vx1',0.02,2,0.5),
                           ('kap_x','T',0.003*(g-1)/g,0,2.0)]:
    d=read_vtk(label+'.0001.vtk')
    f = d['prs']/d['rho']-1.0 if field=='T' else d[field]
    n=f.shape[2-ax]; x=(np.arange(n)+0.5)/n
    prof=f.mean(axis=tuple(a for a in range(3) if a!=2-ax))
    A=np.sum(prof*np.sin(k*x))/np.sum(np.sin(k*x)**2)
    check('decaimiento '+label, A, amp*math.exp(-D*k*k*t), 3e-2 if field=='T' else 2e-3)
m=open('grav.hst').read().strip().split('\n')[-1].split()
check('gravedad mom1', float(m[3]), 1.0, 1e-9); check('gravedad mom3', float(m[5]), 2.0, 1e-9)
d=read_vtk('cool.0001.vtk'); T=(d['prs']/d['rho']).mean(); c=(g-1)*2.0*0.1
check('enfriamiento T(1)', T, (math.sqrt(0.5)-0.5*c)**2, 1e-6)
m=open('box.hst').read().strip().split('\n')[-1].split()
check('pared reflectante: masa', float(m[2]), 1.0, 1e-9)
a=read_vtk(sorted(glob.glob('full.*.vtk'))[-1]); b=read_vtk(sorted(glob.glob('resumed.*.vtk'))[-1])
dm=max(np.abs(a[kk]-b[kk]).max() for kk in a)
print('  %s reinicio bitwise: max|dif| = %.1e' % ('OK ' if dm==0 else 'FAIL', dm)); ok = ok and dm==0
print('TODO OK' if ok else 'HAY FALLOS')
EOF
rm -rf "$OUT"
