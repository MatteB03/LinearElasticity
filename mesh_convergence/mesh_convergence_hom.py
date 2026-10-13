from firedrake import *
import numpy as np
import argparse
from pathlib import Path
from fem_source import (create_box_mesh, create_spaces, symmetry_bcs,
                        solve_forward)

csv_dir = Path("mesh_conv_data_csv")
csv_dir.mkdir(exist_ok=True)
vtu_dir = Path("mesh_conv_data_vtu")
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
    return u_true


def make_inclusion_field(inclusions, bg=1.0):
    """Build a piecewise-constant field from a list of elliptical inclusions.

    Each entry must have keys cx, cy, a, b, val.
    """
    expr = Constant(bg)
    for inc in inclusions:
        inside = ((x_t[0] - inc["cx"]) / inc["a"])**2 \
               + ((x_t[1] - inc["cy"]) / inc["b"])**2 < 1.0
        expr = conditional(inside, inc["val"], expr)
    return expr



#############################################################################################################
##### IMPORTANT STUFF #####
#############################################################################################################
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import qmc

Lx_t, Ly_t, Lz_t = 25,20,16
refinements = 6
sobol_exp = 7

sobol_sampler = qmc.Sobol(d=3, scramble = True, seed = 0)
random_pts_cube = sobol_sampler.random_base2(sobol_exp)
test_points = qmc.scale(random_pts_cube,l_bounds=[0,0,0], u_bounds=[25,20,16])
# print(test_points)

interp_values = np.zeros((refinements, 2**sobol_exp, 3))

for i in range(refinements):
    Nx_t, Ny_t, Nz_t = 2**(i+1), 2**(i+1), 2**(i)
    mm_true = create_box_mesh(Nx_t, Ny_t, Nz_t, Lx_t, Ly_t, Lz_t)
    V_t, Q_t = create_spaces(mm_true)
    bcs_t = symmetry_bcs(V_t)

    lambda_ = Constant(650.0)
    mu = Constant(8.0)
    p_load = Constant(-10.0)

    x_t = SpatialCoordinate(mm_true)
    u_true = generate_and_save(f"mesh_{i+1}",
                      conditional(x_t[0] < x_t[1], 1.0, 2.0), 
                      mm_true, V_t, Q_t, bcs_t, lambda_, mu, p_load)
    pt_eval = PointEvaluator(mm_true, test_points)
    interp_values[i, :, :] = pt_eval.evaluate(u_true)
    if i > 0:
        print()
        rel_diff = np.linalg.norm(interp_values[i,:,:]-interp_values[i-1,:,:])/np.linalg.norm(interp_values[i,:,:])
        print(f"relative difference between interpolated values is {rel_diff}")

