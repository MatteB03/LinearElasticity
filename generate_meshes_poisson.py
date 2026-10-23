from firedrake import *
import math
import numpy as np
import argparse
from pathlib import Path
from fem_source.problem_poisson import (create_box_mesh, solve_forward, exact_solution_sines)

csv_dir = Path("mesh_data_csv")
csv_dir.mkdir(exist_ok=True)
vtu_dir = Path("mesh_data_vtu")
vtu_dir.mkdir(exist_ok=True)

###### FOR FUTURE COMPARISONS WITH FEM RESULTS ######
def generate_and_save(name, param_expr, mm, V,Q, bcs, lambda_, mu, p_load):
    param_true = Function(Q, name="alpha_true")
    param_true.interpolate(param_expr)
    u_true = solve_forward(param_true, bcs, lambda_, mu, p_load, V=V, name="u_true")


    # CSV alpha column always stores param * mu for consistent scaling
    csv_param_vals = param_true.dat.data_ro * float(mu)
    # CSV for PINNs
    coords = mm.coordinates.dat.data_ro
    u_vals = u_true.dat.data_ro


    # VTK for visualization 
    vtk_path = str(vtu_dir / f"{name}.pvd")
    VTKFile(vtk_path).write(u_true, param_true)

    eps = sym(grad(u_true))
    strain_data = np.zeros((coords.shape[0], 9))
    for idx, (i, j) in enumerate([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]):
        f = Function(Q)
        f.project(eps[i, j])
        strain_data[:, idx] = f.dat.data_ro

    data = np.column_stack([coords, u_vals, csv_param_vals, strain_data])
    header = "x,y,z,ux,uy,uz,alpha,e_xx,e_xy,e_xz,e_yx,e_yy,e_yz,e_zx,e_zy,e_zz"
    csv_path = str(csv_dir / f"{name}.csv")
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    
    print(f"[{name}]")
    print(f"  CSV:  {csv_path}  ({coords.shape[0]} points)")
    print(f"  VTK:  {vtk_path}")
    return u_true

pts_dir = Path("mesh_data_pts")
pts_dir.mkdir(exist_ok=True)

def save_pts(name, mm, pts_dir):
    """Write mesh node coordinates to a .pts file (one 'x y z' per line)."""
    pts_dir = Path(pts_dir)
    pts_dir.mkdir(parents=True, exist_ok=True)

    coords = mm.coordinates.dat.data_ro.copy()   # shape (n_points, 3)
    pts_path = str(pts_dir / f"{name}.pts")
    np.savetxt(pts_path, coords, fmt="%.10e", delimiter=" ",
               header=str(coords.shape[0]), comments="")

    print(f"[{name}]")
    print(f"  PTS:  {pts_path}  ({coords.shape[0]} points)")
    return coords

dynpt_dir = Path("mesh_data_dynpt")
dynpt_dir.mkdir(exist_ok=True)

def write_dynpt(path, vals):
    """IGB con header vec3f (richiesto dal reader) ma dati scalari: n float, 1 per nodo."""
    vals = np.asarray(vals, dtype="<f4").ravel()
    n = vals.size
    hdr = (f"x:{n} y:1 z:1 t:1 type:vec3f systeme:little_endian org_t:0 \r\n"
           f"dim_t:0 unites_x:um unites_y:um unites_z:um unites_t:ms unites:mm \r\n"
           f"facteur:1 zero:0 \r\n").encode("ascii")
    assert len(hdr) < 1021
    hdr = hdr.ljust(1021, b" ") + b"\r\n\f"      # 1024 byte totali
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(vals.tobytes())                  # solo n float, NON 3n


def save_dynpt(name, mm, L1, L2, L3, dynpt_dir):
    dynpt_dir = Path(dynpt_dir)
    dynpt_dir.mkdir(parents=True, exist_ok=True)

    Q = FunctionSpace(mm, "CG", 1)
    u_ex = Function(Q, name="u_exact")
    u_ex.interpolate(exact_solution_sines(mm, L1, L2, L3))
    vals = u_ex.dat.data_ro.copy()

    dynpt_path = dynpt_dir / f"{name}.dynpt"
    write_dynpt(dynpt_path, vals)

    print(f"  DYNPT: {dynpt_path}  ({vals.shape[0]} values)")
    return vals


N = 15
h = np.log(6)/N

L1,L2,L3 = [],[],[]
for i in range(N+1):
    for j in range(N+1-i):
        a, b = i*h, j*h
        l1 = np.exp(-(2*a+b)/3)
        l2 = l1 * np.exp(a)
        l3 = l1 * np.exp(a+b)
        L1.append(l1); L2.append(l2); L3.append(l3)

L1 = np.array(L1)
L2 = np.array(L2)
L3 = np.array(L3)
AR = L3/L1
pts = np.column_stack([L1, L2, L3])   # shape (55, 3)

np.savetxt(pts_dir / "L_values.pts", pts,
           header=str(pts.shape[0]),   # first line = number of points
           comments="",
           fmt="%.10f")

for k in range(len(L1)):
    h = 30
    Lx,Ly,Lz = L1[k],L2[k],L3[k]
    Nx,Ny,Nz = math.floor(Lx*h),math.floor(Ly*h),math.floor(Lz*h)
    mm = create_box_mesh(Nx,Ny,Nz,Lx,Ly,Lz)
    save_pts(f'mesh_{k+1}', mm, pts_dir)
    save_dynpt(f'mesh_{k+1}',mm,Lx,Ly,Lz, dynpt_dir)
    