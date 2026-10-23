from firedrake import *
import math
import numpy as np
from pathlib import Path
from fem_source.problem_le import (create_box_mesh, solve_forward, exact_disp,)

csv_dir = Path("le_mesh_data_csv")
csv_dir.mkdir(exist_ok=True)
vtu_dir = Path("le_mesh_data_vtu")
vtu_dir.mkdir(exist_ok=True)

N = 9
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

np.savetxt(csv_dir / "L_values.pts", pts,
           header=str(pts.shape[0]),   # first line = number of points
           comments="",
           fmt="%.10f")


mu = 1.0
mi = 0.3
lambda_ = 2*mu*mi/(1-2*mi)


def generate_and_save(name, mm, L, lambda_, mu):
    lambda_ = Constant(lambda_)
    mu = Constant(mu)
 
    V = VectorFunctionSpace(mm, "P", 1)
    Q = FunctionSpace(mm, "P", 1)
 
    u_ex, f = exact_disp(mm, L, lambda_, mu)
    u_true = solve_forward(mm, f, V, lambda_, mu)

    # VTK for visualization
    vtk_path = str(vtu_dir / f"{name}.pvd")
    VTKFile(vtk_path).write(u_true)
 
    # CSV (same layout as forward.py); alpha column = mu (constant here)
    coords = mm.coordinates.dat.data_ro
    u_vals = u_true.dat.data_ro
    alpha_vals = np.full(coords.shape[0], float(mu))
 
    eps = sym(grad(u_true))
    strain_data = np.zeros((coords.shape[0], 9))
    for idx, (i, j) in enumerate([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]):
        g = Function(Q)
        g.project(eps[i, j])
        strain_data[:, idx] = g.dat.data_ro
 
    data = np.column_stack([coords, u_vals, alpha_vals, strain_data])
    header = "x,y,z,ux,uy,uz,alpha,e_xx,e_xy,e_xz,e_yx,e_yy,e_yz,e_zx,e_zy,e_zz"
    csv_path = str(csv_dir / f"{name}.csv")
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
 
    print(f"[{name}]  L=({L[0]:.4f}, {L[1]:.4f}, {L[2]:.4f})")
    print(f"  CSV:  {csv_path}  ({coords.shape[0]} points)")
    print(f"  VTK:  {vtk_path}")
    return u_true


for k in range(len(L1)):
    h = 30
    Lx,Ly,Lz = L1[k],L2[k],L3[k]
    Nx,Ny,Nz = math.floor(Lx*h),math.floor(Ly*h),math.floor(Lz*h)
    mm = create_box_mesh(Nx,Ny,Nz,Lx,Ly,Lz)
    generate_and_save(f"mesh_{k+1}", mm, (Lx, Ly, Lz), lambda_, mu)
