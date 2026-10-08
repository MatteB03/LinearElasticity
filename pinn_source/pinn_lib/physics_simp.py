import tensorflow as tf



def laplacian(tape, u, x, dim):
    """
    ----------------------------------------------------------------------
    Comp-wise Laplacian
    ----------------------------------------------------------------------
    Inputs: 
    tape : tf.GradientTape (persistent)
        Must already watch *x* and have *u = model(x)* recorded;
    u : tf.Tensor, shape (n, dim)
        Displacement field evaluated at *x*;
    x : tf.Tensor, shape (n, dim)
        Spatial coordinates (watched by *tape*);
    dim : int
        Spatial dimension (typically 3).
    ----------------------------------------------------------------------
    Output:
    tf.Tensor, shape (n, dim)
    Entry [:, i] is sum_j d^2 u_i / dx_j^2
    ----------------------------------------------------------------------
    """
    comps = []
    for i in range(dim):
        g = tape.gradient(u[:, i], x)[:, :dim]
        # create a list with a tensor for each disp comp
        comps.append(tf.add_n([tape.gradient(g[:, j], x)[:, j] for j in range(dim)]))
    return tf.stack(comps, axis=-1) # stack all dim lists in a single tensor of shape (n,dim)

def normal_derivative(tape, u, x, dim, normal_axis):
    """
    ----------------------------------------------------------------------
    Computes the normal derivative along prescribed axis
    ----------------------------------------------------------------------
    Inputs: 
    tape : tf.GradientTape (persistent)
        Must already watch *x* and have *u = model(x)* recorded;
    u : tf.Tensor, shape (n, dim)
        Displacement field evaluated at *x*;
    x : tf.Tensor, shape (n, dim)
        Spatial coordinates (watched by *tape*);
    dim : int
        Spatial dimension (typically 3);
    normal_axis : int
        Index of coordinate axis along which the 
        derivative is taken (0 for x, 1 for y, 2 for z).
    ----------------------------------------------------------------------
    Output:
    tf.Tensor, shape (n, dim)
        Entry [:, i] is d u_i / d x_normal_axis,
    ----------------------------------------------------------------------
    """
    return tf.stack(
        [tape.gradient(u[:, i], x)[:, normal_axis] for i in range(dim)], axis=-1
    )

def time_derivative(tape, u, x, dim):
    """
    ----------------------------------------------------------------------
    Computes the time derivative of each component of u
    ----------------------------------------------------------------------
    Inputs: 
    tape : tf.GradientTape (persistent)
        Must already watch *x* and have *u = model(x)* recorded;
    u : tf.Tensor, shape (n, dim)
        Solution field evaluated at *x*;
    x : tf.Tensor, shape (n, dim + 1)
        Space-time coordinates (watched by *tape*),
        space in columns 0..dim-1, time in the last column (index dim);
    dim : int
        Spatial dimension (typically 3).
    ----------------------------------------------------------------------
    Output:
    tf.Tensor, shape (n, dim)
    Entry [:, i] is d u_i / dt
    ----------------------------------------------------------------------
    """
    return tf.stack(
        [tape.gradient(u[:, i], x)[:, dim] for i in range(dim)], axis=-1
    )