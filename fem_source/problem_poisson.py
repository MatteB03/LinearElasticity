from firedrake import *

def create_box_mesh(Nx, Ny, Nz, Lx=1.0, Ly=1.0, Lz=1.0): #the problem is in [0,1]**3
    """Create a tetrahedral box mesh."""
    return BoxMesh(Nx, Ny, Nz, Lx, Ly, Lz, hexahedral=False)

# the difference with the disp code is that u is a scalar, so we need only a single space and not VectorF...
def create_spaces(mesh, u_degree=1):
    """Create function spaces. Returns (V, Q) where V is vector, Q is scalar."""
    V = FunctionSpace(mesh, "P", u_degree)
    Q = FunctionSpace(mesh, "P", 1)
    return V, Q

# this replaces symmetry_bcs from the original file
# if I saw correctly firedrake has the default "on all boundary" option
def dirichlet_bcs(V):                         
    return [DirichletBC(V, Constant(0.0), "on_boundary")]

def source_term_sines(mesh):
    xyz = SpatialCoordinate(mesh)
    return 3*pi**2 * sin(pi*xyz[0]) * sin(pi*xyz[1]) * sin(pi*xyz[2])

# 2[y(1-y)z(1-z)+x(1-x)z(1-z)+x(1-x)y(1-y)].
def source_term_bubble(mesh):
    xyz = SpatialCoordinate(mesh)
    return 2*(xyz[0]*(1-xyz[0])*xyz[1]*(1-xyz[1]) + xyz[0]*(1-xyz[0])*xyz[2]*(1-xyz[2]) + xyz[2]*(1-xyz[2])*xyz[1]*(1-xyz[1]))

# atm this is useless
# def exact_solution_sines(mesh):
#     xyz = SpatialCoordinate(mesh)
#     return sin(pi*xyz[0]) * sin(pi*xyz[1]) * sin(pi*xyz[2])

# here there used to be strain_energy, I simply replaced it with the energy for the poison problem
def poisson_energy(u, f):             
    return 0.5*inner(grad(u), grad(u))*dx - f*u*dx


def make_forward_solver(u, bcs, f):
    """Create forward solver. Returns (solver, W_form, G_form)."""
    W_form = poisson_energy(u, f)
    G_form = derivative(W_form, u)
    prob = NonlinearVariationalProblem(G_form, u, bcs)
    solver = NonlinearVariationalSolver(prob)
    return solver, W_form, G_form


def solve_forward(bcs, f, V=None, u=None, name="u_fwd"):
    """One-shot forward solve. Returns the solution Function."""
    if u is None:
        u = Function(V, name=name)
    solver, _, _ = make_forward_solver(u, bcs, f)
    solver.solve()
    return u



