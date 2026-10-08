import pinn_source.pinn_lib as ns
from pinn_source.pinn_lib import physics_simp
import tensorflow as tf

#############################################################################
# PDEs: 
# Poisson : - nabla(u) = f
# Heat : dt(u) - nabla(u) = f
#############################################################################

def PDE_Poisson(x, model, dim, alpha=1.0, source=0.0):
    alpha_tensor = tf.constant(alpha, dtype=ns.config.get_dtype())
    source_tensor = tf.convert_to_tensor(source, dtype=ns.config.get_dtype())
    n_pts = tf.shape(x)[0]
    f = tf.repeat([source_tensor], n_pts, axis=0)
    with ns.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)
        u = model(x)
        lap = physics_simp.laplacian(tape, u, x, dim)
    return tf.add(-alpha_tensor * lap, -f)

def PDE_Heat(x, model, dim, alpha=1.0, source=0.0):
    alpha_tensor = tf.constant(alpha, dtype=ns.config.get_dtype())
    source_tensor = tf.convert_to_tensor(source, dtype=ns.config.get_dtype())
    n_pts = tf.shape(x)[0]
    f = tf.repeat([source_tensor], n_pts, axis=0)

    with ns.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)
        u = model(x)
        u_t = physics_simp.time_derivative(tape, u, x, dim)
        lap = physics_simp.laplacian(tape, u, x, dim)

    return tf.add(u_t, tf.add(-alpha_tensor * lap, -f))

#############################################################################
# Boundary conditions
#############################################################################

def Dirichlet(x, model, vector, component = None):
    d = model(x)
    if callable(vector):
        vec = vector(x)
    else:
        vec = vector
    if component is not None:
        d = d[:, component]
        vec = vec[component]
    return d - vec

def Neumann(x, model, dim, normal_axis, vector, component = None):
    with ns.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)
        u = model(x)
        du_dn = physics_simp.normal_derivative(tape, u, x, dim, normal_axis)
    if callable(vector):
        vec = vector(x)
    else:
        vec = vector
    if component is not None:
        du_dn = du_dn[:, component]
    return du_dn - vec