from firedrake import *
import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.insert(0, "..")
from fem_source.problem_poisson import (create_box_mesh, create_spaces, dirichlet_bcs,
                        solve_forward, source_term_cosines)

csv_dir = Path("mc_cosines_csv")
csv_dir.mkdir(exist_ok=True)
vtu_dir = Path("mc_cosines_vtu")
vtu_dir.mkdir(exist_ok=True)

dim = 3 # source_term_cosines is implemented for 2 or 3 
k = 10

def generate_and_save(name, mm, V,Q, bcs):
    f = source_term_cosines(mm, k, dim = dim)
    u_true = solve_forward(bcs, f, V=V, name="u_true")

    # CSV for PINNs
    coords = mm.coordinates.dat.data_ro
    u_vals = u_true.dat.data_ro


    # VTK for visualization 
    vtk_path = str(vtu_dir / f"{name}_{dim}.pvd")
    VTKFile(vtk_path).write(u_true)

    data = np.column_stack([coords, u_vals])
    header = ",".join(["x", "y", "z"][:dim]) + ",u_val"
    csv_path = str(csv_dir / f"{name}_{dim}.csv")
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    
    print(f"[{name}]")
    print(f"  CSV:  {csv_path}  ({coords.shape[0]} points)")
    print(f"  VTK:  {vtk_path}")
    return u_true



#############################################################################################################
##### IMPORTANT STUFF #####
#############################################################################################################
import numpy as np
from scipy.stats import qmc

refinements = 7
sobol_exp = 7

sobol_sampler = qmc.Sobol(d=dim, scramble = True, seed = 0)
test_points = sobol_sampler.random_base2(sobol_exp)
# print(test_points)
u_exact = np.prod((1-test_points)**3 * test_points**3 * np.cos(k*test_points), axis=1) # for the sake of comparison

interp_values = np.zeros((refinements, 2**sobol_exp))

for i in range(refinements):
    N_vert = 2**i
    mm_true = create_box_mesh(N_vert, N_vert, N_vert, dim = dim)
    V_t, Q_t = create_spaces(mm_true)
    bcs_t = dirichlet_bcs(V_t)

    x_t = SpatialCoordinate(mm_true)
    u_true = generate_and_save(f"mesh_{i+1}", mm_true, V_t, Q_t, bcs_t)
    pt_eval = PointEvaluator(mm_true, test_points)
    interp_values[i, :] = pt_eval.evaluate(u_true)
    mse = np.mean((interp_values[i,:]-u_exact)**2)
    print(f"MSE between interpolated value and analytic solution is {mse}")

