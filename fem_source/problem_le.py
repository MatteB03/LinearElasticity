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
    V = VectorFunctionSpace(mesh, "P", u_degree)
    Q = FunctionSpace(mesh, "P", 1)
    return V, Q

def dirichlet_bcs(V):                         
    return [DirichletBC(V, 0, "on_boundary")]

def exact_disp(mm, L, lambda_, mu, a=(1.0,1.0,1.0)):
    x = SpatialCoordinate(mm)
    k = [pi / L[i] for i in range(3)]
    s = [sin(k[i]*x[i]) for i in range(3)]
    c = [cos(k[i]*x[i]) for i in range(3)]
    phi = s[0]*s[1]*s[2]
    k2 = sum(ki**2 for ki in k)
 
    comps = []
    for i in range(3):
        acc = a[i]*k[i]**2*phi
        for j in range(3):
            if j == i:
                continue
            m = 3-i-j
            acc -= a[j]*k[i]*k[j]*c[i]*c[j]*s[m]
        comps.append(mu*k2*a[i]*phi + (lambda_+mu)*acc)
 
    u_ex = as_vector([phi*a[0], phi*a[1], phi*a[2]])
    f = as_vector(comps)
    return u_ex, f
 
 
def solve_forward(mm, f, V, lambda_, mu, name="u_true"):
    
    u = TrialFunction(V)
    v = TestFunction(V)
    eps = lambda w: sym(grad(w))
    a_form = (2*mu*inner(eps(u), eps(v)) + lambda_*div(u)*div(v))*dx
    L_form = inner(f, v)*dx(degree=4)
    bc = dirichlet_bcs(V)
    u_h = Function(V, name=name)
    solve(a_form == L_form, u_h, bcs=bc,
          solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"})
    return u_h
