from firedrake import *
def create_box_mesh(Nx, Ny, Nz, Lx=1.0, Ly=1.0, Lz=1.0, dim = 3): #the problem is in [0,1]**3
    """Create a tetrahedral box mesh."""
    if dim == 3:
        return BoxMesh(Nx, Ny, Nz, Lx, Ly, Lz, hexahedral=False)
    elif dim == 2:
        return RectangleMesh(Nx, Ny, Lx, Ly)

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

def source_term_sines_pts(mesh, L1, L2, L3):
    xyz = SpatialCoordinate(mesh)
    k2 = pi**2 * (1/L1**2 + 1/L2**2 + 1/L3**2)
    return k2 * sin(pi*xyz[0]/L1) * sin(pi*xyz[1]/L2) * sin(pi*xyz[2]/L3)

def exact_solution_sines(mesh,L1,L2,L3):
    xyz = SpatialCoordinate(mesh)
    return sin(pi*xyz[0]/L1) * sin(pi*xyz[1]/L2) * sin(pi*xyz[2]/L3)

# 2[y(1-y)z(1-z)+x(1-x)z(1-z)+x(1-x)y(1-y)].
def source_term_bubble(mesh):
    xyz = SpatialCoordinate(mesh)
    return 2*(xyz[0]*(1-xyz[0])*xyz[1]*(1-xyz[1]) + xyz[0]*(1-xyz[0])*xyz[2]*(1-xyz[2]) + xyz[2]*(1-xyz[2])*xyz[1]*(1-xyz[1]))

def source_term_cosines(mesh, k, dim=2):
    xyz = SpatialCoordinate(mesh)

    def term_i(x):
        return x**3 * (1-x)**3 * cos(k*x)
    
    def u_ii(x): # very uninspired name for d2u/di2
        polynomial = x**3 * (1-x)**3
        first_der_pol = 3*x**2 * (1-x)**2 * (1-2*x)
        sec_der_pol = 6*x * (1-x) * (1-5*x+5*x**2)
        return (sec_der_pol - k**2*polynomial)*cos(k*x) - 2*k*first_der_pol*sin(k*x)

    if dim == 2:
        return -(u_ii(xyz[0])*term_i(xyz[1]) + u_ii(xyz[1])*term_i(xyz[0]))
    elif dim == 3:
        return -(u_ii(xyz[0])*term_i(xyz[1])*term_i(xyz[2]) + u_ii(xyz[1])*term_i(xyz[0])*term_i(xyz[2]) + u_ii(xyz[2])*term_i(xyz[0])*term_i(xyz[1]))

# here there used to be strain_energy, I simply replaced it with the energy for the generalized poisson problem
def generalized_poisson_energy(u, f, rho):
    return (0.5*inner(grad(u), grad(u)) + 0.5*rho*u**2 - f*u)*dx


def make_forward_solver(u, bcs, f, rho):
    """Create forward solver. Returns (solver, W_form, G_form)."""
    W_form = generalized_poisson_energy(u, f, rho)
    G_form = derivative(W_form, u)
    prob = NonlinearVariationalProblem(G_form, u, bcs)
    solver = NonlinearVariationalSolver(prob)
    return solver, W_form, G_form


def solve_forward(bcs, f, rho = 0.0, V=None, u=None, name="u_fwd"):
    """One-shot forward solve. Returns the solution Function."""
    if u is None:
        u = Function(V, name=name)
    solver, _, _ = make_forward_solver(u, bcs, f, rho)
    solver.solve()
    return u



