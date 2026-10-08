from firedrake import *
import numpy as np
import argparse
from pathlib import Path
from itertools import product

from fem_source import (create_box_mesh, create_spaces, symmetry_bcs,
                        solve_forward)

# Output directories
csv_dir = Path("cp_data_csv")
csv_dir.mkdir(exist_ok=True)
vtu_dir = Path("cp_data_vtu")
vtu_dir.mkdir(exist_ok=True)


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

# ── Common setup ──────────────────────────────────────────────

lambda_ = Constant(650.0)
mu = Constant(8.0)
p_load = Constant(-10.0)

Lx_list = [2.5, 3.5]#, 4.5, 5.5, 6.5]
Ly_list = [2.0, 3.0]#, 4.0, 5.0, 6.0]
Lz_list = [1.6, 3.2]#, 4.8, 6.4]


# ── Number of geometries ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate forward elasticity data.")
parser.add_argument("n", type=int, default=1, nargs="?", help="Numero di geometrie da generare (default: 1)")
args = parser.parse_args()

for Lx, Ly, Lz in product(Lx_list, Ly_list, Lz_list):
    Nx, Ny, Nz = round(Lx*10), round(Ly*10), round(Lz*10)
    print("────────────────────────────────────────────────────────────────────────────────────────────")
    print(f"Starting case Nx: {Nx}, Ny: {Ny}, Nz: {Nz}")

    mm_true = create_box_mesh(Nx, Ny, Nz, Lx, Ly, Lz) ### va bene o dobbiamo fare uno stesso numero di punti anche se le mesh sono più grandi?
    V_t, Q_t = create_spaces(mm_true)
    bcs_t = symmetry_bcs(V_t)
    x_t = SpatialCoordinate(mm_true)

    generate_and_save(f"hom_symcube_{Nx}_{Ny}_{Nz}", Constant(1.0),
                      mm_true, V_t, Q_t, bcs_t, lambda_, mu, p_load)
    