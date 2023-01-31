import scipy
from scipy import stats as stats
import numpy as np
import matplotlib
from matplotlib import pyplot as plt











"""OTHER USEFUL FUNCTIONS"""
"""Geometry functions"""


def get_perpendicular(a=None):
    """Given 2-vector, a, returns its perpendicular
    Args:
        a (array, optional): 2-vector direction. Defaults to None.
    Returns:
        array: perpendicular to a
    """
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def vector_intercepts(vector_list_a, vector_list_b, return_collisions=False):
    """
    Each element of vector_list_a gives a line segment of the form [[x_a_0,y_a_0],[x_a_1,y_a_1]], or, in vector notation [p_a_0,p_a_1]
    (same goes for vector vector_list_b). Thus
        vector_list_A.shape = (N_a,2,2)
        vector_list_B.shape = (N_b,2,2)
    where N_a is the number of vectors defined in vector_list_a

    Each line segments define an (infinite) line, parameterised by line_a = p_a_0 + l_a.(p_a_1-p_a_0).
    We want to find the intersection between these lines in terms of the parameters l_a and l_b.
    Iff l_a and l_b are BOTH between 0 and 1 then the line segments intersect. Thus the goal is to return an array, I,  of shape
        I.shape = (N_a,N_b,2)
    where, if I[n_a,n_b][0] and I[n_a,n_b][1] are both between 0 and 1 then it means line segments vector_list_a[n_a] and vector_list_b[n_b] intersect.

    To do this we consider solving the equation line_a = line_b. The solution to this is:
        l_a = dot((p_b_0 - p_a_0) , (p_b_1 - p_b_0)_p) / dot((p_a_1 - p_a_0) , (p_b_1 - p_b_0)_p)
        l_b = dot((p_a_0 - p_b_0) , (p_a_1 - p_a_0)_p) / dot((p_b_1 - p_b_0) , (p_a_1 - p_a_0)_p)
    where "_p" denotes the perpendicular (in two-D [x,y]_p = [-y,x]). Using notation
        l_a = dot(d0,sb_p) / dot(sa,sb_p)
        l_b = dot(-d0,sa_p) / dot(sb,sa_p)
    for
        d0 = p_b_0 - p_a_0
        sa = p_a_1 - p_a_0
        sb = p_b_1 - p_b_0
    We will calculate these first.

    If return_collisions == True, the list of intercepts is used to assess whether each pair of segments actually collide (True) or not (False)
    and this bollean array (shape = (N_a,N_b)) is returned instead.
    """
    assert (vector_list_a.shape[-2:] == (2, 2)) and (
        vector_list_b.shape[-2:] == (2, 2)
    ), "vector_list_a and vector_list_b must be shape (_,2,2), _ is optional"
    vector_list_a = vector_list_a.reshape(-1, 2, 2)
    vector_list_b = vector_list_b.reshape(-1, 2, 2)
    vector_list_a = vector_list_a + np.random.normal(
        scale=1e-6, size=vector_list_a.shape
    )
    vector_list_b = vector_list_b + np.random.normal(
        scale=1e-6, size=vector_list_b.shape
    )

    N_a = vector_list_a.shape[0]
    N_b = vector_list_b.shape[0]

    d0 = np.expand_dims(vector_list_b[:, 0, :], axis=0) - np.expand_dims(
        vector_list_a[:, 0, :], axis=1
    )  # d0.shape = (N_a,N_b,2)
    sa = vector_list_a[:, 1, :] - vector_list_a[:, 0, :]  # sa.shape = (N_a,2)
    sb = vector_list_b[:, 1, :] - vector_list_b[:, 0, :]  # sb.shape = (N_b,2)
    sa_p = np.flip(sa.copy(), axis=1)
    sa_p[:, 0] = -sa_p[:, 0]  # sa_p.shape = (N_a,2)
    sb_p = np.flip(sb.copy(), axis=1)
    sb_p[:, 0] = -sb_p[:, 0]  # sb.shape = (N_b,2)

    """Now we can go ahead and solve for the line segments
    since d0 has shape (N_a,N_b,2) in order to perform the dot product we must first reshape sa (etc.) by tiling to shape (N_a,N_b,2)
    """
    sa = np.tile(np.expand_dims(sa, axis=1), reps=(1, N_b, 1))  # sa.shape = (N_a,N_b,2)
    sb = np.tile(np.expand_dims(sb, axis=0), reps=(N_a, 1, 1))  # sb.shape = (N_a,N_b,2)
    sa_p = np.tile(
        np.expand_dims(sa_p, axis=1), reps=(1, N_b, 1)
    )  # sa.shape = (N_a,N_b,2)
    sb_p = np.tile(
        np.expand_dims(sb_p, axis=0), reps=(N_a, 1, 1)
    )  # sb.shape = (N_a,N_b,2)
    """The dot product can now be performed by broadcast multiplying the arraays then summing over the last axis"""
    l_a = (d0 * sb_p).sum(axis=-1) / (sa * sb_p).sum(axis=-1)  # la.shape=(N_a,N_b)
    l_b = (-d0 * sa_p).sum(axis=-1) / (sb * sa_p).sum(axis=-1)  # la.shape=(N_a,N_b)

    intercepts = np.stack((l_a, l_b), axis=-1)
    if return_collisions == True:
        direct_collision = ((intercepts[:, :, 0] > 0) * (intercepts[:, :, 0] < 1) * (intercepts[:, :, 1] > 0) * (intercepts[:, :, 1] < 1))
        return direct_collision
    else:
        return intercepts


def shortest_vectors_from_points_to_lines(positions, vectors):
    """
    Takes a list of positions and a list of vectors (line segments) and returns the pairwise  vectors of shortest distance
    FROM the vector segments TO the positions.
    Suppose we have a list of N_p positions and a list of N_v line segments (or vectors). Each position is a point like [x_p,y_p], or p_p as a vector.
    Each vector is defined by two points [[x_v_0,y_v_0],[x_v_1,y_v_1]], or [p_v_0,p_v_1]. Thus
        positions.shape = (N_p,2)
        vectors.shape = (N_v,2,2)

    Each vector defines an infinite line, parameterised by line_v = p_v_0 + l_v . (p_v_1 - p_v_0).
    We want to solve for the l_v defining the point on the line with the shortest distance to p_p. This is given by:
        l_v = dot((p_p-p_v_0),(p_v_1-p_v_0)/dot((p_v_1-p_v_0),(p_v_1-p_v_0)).
    Or, using a diferrent notation
        l_v = dot(d,s)/dot(s,s)
    where
        d = p_p-p_v_0
        s = p_v_1-p_v_0"""
    assert (positions.shape[-1] == 2) and (
        vectors.shape[-2:] == (2, 2)
    ), "positions and vectors must have shapes (_,2) and (_,2,2) respectively. _ is optional"
    positions = positions.reshape(-1, 2)
    vectors = vectors.reshape(-1, 2, 2)
    positions = positions + np.random.normal(scale=1e-6, size=positions.shape)
    vectors = vectors + np.random.normal(scale=1e-6, size=vectors.shape)

    N_p = positions.shape[0]
    N_v = vectors.shape[0]

    d = np.expand_dims(positions, axis=1) - np.expand_dims(
        vectors[:, 0, :], axis=0
    )  # d.shape = (N_p,N_v,2)
    s = vectors[:, 1, :] - vectors[:, 0, :]  # vectors.shape = (N_v,2)

    """in order to do the dot product we must reshaope s to be d's shape."""
    s_ = np.tile(
        np.expand_dims(s.copy(), axis=0), reps=(N_p, 1, 1)
    )  # s_.shape = (N_p,N_v,2)
    """now do the dot product by broadcast multiplying the arraays then summing over the last axis"""

    l_v = (d * s).sum(axis=-1) / (s * s).sum(axis=-1)  # l_v.shape = (N_p,N_v)

    """
    Now we can actually find the vector of shortest distance from the line segments to the points which is given by the size of the perpendicular
        perp = p_p - (p_v_0 + l_v.s_)

    But notice that if l_v > 1 then the perpendicular drops onto a part of the line which doesn't exist.
    In fact the shortest distance is to the point on the line segment where l_v = 1. Likewise for l_v < 0.
    To fix this we should limit l_v to be between 1 and 0
    """
    l_v[l_v > 1] = 1
    l_v[l_v < 0] = 0

    """we must reshape p_p and p_v_0 to be shape (N_p,N_v,2), also reshape l_v to be shape (N_p, N_v,1) so we can broadcast multiply it wist s_"""
    p_p = np.tile(
        np.expand_dims(positions, axis=1), reps=(1, N_v, 1)
    )  # p_p.shape = (N_p,N_v,2)
    p_v_0 = np.tile(
        np.expand_dims(vectors[:, 0, :], axis=0), reps=(N_p, 1, 1)
    )  # p_v_0.shape = (N_p,N_v,2)
    l_v = np.expand_dims(l_v, axis=-1)

    perp = p_p - (p_v_0 + l_v * s_)  # perp.shape = (N_p,N_v,2)

    return perp


def get_line_segments_between(pos1, pos2):
    """Takes two position arrays and returns the array of pair-wise line segments between positions in each array (from pos1 to pos2).
    Args:
        pos1 (array): (N x dimensionality) array of positions
        pos2 (array): (M x dimensionality) array of positions
    Returns:
        (N x M x 2 x dimensionality) array of vectors from pos1's to pos2's"""

    pos1_ = pos1.reshape(-1, 1, pos1.shape[-1])
    pos2_ = pos2.reshape(1, -1, pos2.shape[-1])
    pos1 = np.repeat(pos1_, pos2_.shape[1], axis=1)
    pos2 = np.repeat(pos2_, pos1_.shape[0], axis=0)
    lines = np.stack((pos1, pos2), axis=-2)
    return lines


def get_vectors_between(pos1=None, pos2=None, line_segments=None):
    """Takes two position arrays and returns the array of pair-wise vectors between positions in each array (from pos1 to pos2).
    Args:
        pos1 (array): (N x dimensionality) array of positions
        pos2 (array): (M x dimensionality) array of positions
        line_segments: if you already have the line segments, just pass these
    Returns:
            (N x M x dimensionality) array of vectors from pos1's to pos2's"""
    if line_segments is None:
        line_segments = get_line_segments_between(pos1, pos2)
    vectors = line_segments[..., 0, :] - line_segments[..., 1, :]
    return vectors


def get_distances_between(pos1=None, pos2=None, vectors=None):
    """Takes two position arrays and returns the array of pair-wise euclidean distances between positions in each array (from pos1 to pos2).
    Args:
        pos1 (array): (N x dimensionality) array of positions
        pos2 (array): (M x dimensionality) array of positions
        vectors: if you already have the pair-wise vectors between pos1 and pos2, just pass these
    Returns:
            (N x M) array of distances from pos1's to pos2's"""
    if vectors is None:
        vectors = get_vectors_between(pos1, pos2)
    distances = np.linalg.norm(vectors, axis=-1)
    return distances


def get_angle(segment,is_array=False):
    """Given a 'segment' (either 2x2 start and end positions or 2x1 direction bearing)
         returns the 'angle' of this segment modulo 2pi
    Args:
        segment (array): The segment, (2,2) or (2,) array
        is_array(bool): If array is True the first dimension is taken as the list dimension while the next 1 or 2 as the segment/vector dimensions. So, for example, you can pass in a list of vectors shape (5,2) or a list of segments shape (5,2,2,) as long as you set this True (or else a list of 2 vectors might get confused with a single segment!)
    Returns:
        float: angle of segment
    """
    segment = np.array(segment)
    #decide if we are dealing with vectors or segments 
    is_vec = True #whether we're dealing with vectors (2,) or segments (2,2,)
    a_segment = segment
    if is_array == True: 
        a_segment = segment[0]
        N = segment.shape[0]
    if a_segment.shape == (2,2,): is_vec = False
    # reshape so segments have shape (N,2,2,) and vectors have shape (N,2,)
    if (not is_array and is_vec): segment = segment.reshape(1,2)
    if (not is_array and not is_vec): segment = segment.reshape(1,2,2)

    eps = 1e-6
    if is_vec: angs = np.mod(np.arctan2(segment[:,1], (segment[:,0] + eps)), 2 * np.pi)
    elif not is_vec:
        angs = np.mod(
            np.arctan2(
                (segment[:,1,1] - segment[:,0,1]), (segment[:,1,0] - segment[:,0,0] + eps)
            ),
            2 * np.pi,
        )
    
    if not is_array: angs = angs[0]

    return angs

def rotate(vector, theta):
    """rotates a vector shape (2,) by angle theta.
    Args:
        vector (array): the 2d vector
        theta (flaot): the rotation angle
    """
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    vector_new = np.matmul(R, vector)
    return vector_new


def wall_bounce(current_velocity, wall):
    """Given current direction and wall returns a new direction which is the result of reflecting off that wall
    Args:
        current_direction (array): the current direction vector
        wall (array): start and end coordinates of the wall
    Returns:
        array: new direction
    """
    wall_perp = get_perpendicular(wall[1] - wall[0])
    if np.dot(wall_perp, current_velocity) <= 0:
        wall_perp = (
            -wall_perp
        )  # it is now the get_perpendicular with smallest angle to dir
    wall_par = wall[1] - wall[0]
    if np.dot(wall_par, current_velocity) <= 0:
        wall_par = -wall_par  # it is now the parallel with smallest angle to dir
    wall_par, wall_perp = (
        wall_par / np.linalg.norm(wall_par),
        wall_perp / np.linalg.norm(wall_perp),
    )  # normalise
    new_velocity = wall_par * np.dot(current_velocity, wall_par) - wall_perp * np.dot(
        current_velocity, wall_perp
    )

    return new_velocity


def pi_domain(x):
    """Converts x (in radians) to be on domain [-pi,pi]
    Args: x (flat or array): angles
    Returns:x: angles recast onto [-pi,pi] domain
    """
    x = np.array(x)
    x_ = x.reshape(-1)
    x_ = x_ % (2 * np.pi)
    x_[x_ > np.pi] = -2 * np.pi + x_[x_ > np.pi]
    x = x_.reshape(x.shape)
    return x


"""Stochastic-assistance functions"""


def ornstein_uhlenbeck(dt, x, drift=0.0, noise_scale=0.2, coherence_time=5.0):
    """An ornstein uhlenbeck process in x.
    x can be multidimensional
    Args:
        dt: update time step
        x: the stochastic variable being updated
        drift (float, or same type as x, optional): [description]. Defaults to 0.
        noise_scale (float, or same type as x, optional): Magnitude of deviations from drift. Defaults to 0.2 (20 cm s^-1 if units of x are in metres).
        coherence_time (float, optional):
        Effectively over what time scale you expect x to change. Can be a vector (one timescale for each element of x) directions. Defaults to 5.

    Returns:
        dx (same type as x); the required update to x
    """
    x = np.array(x)
    drift = drift * np.ones_like(x)
    noise_scale = noise_scale * np.ones_like(x)
    coherence_time = coherence_time * np.ones_like(x)
    sigma = np.sqrt((2 * noise_scale ** 2) / (coherence_time * dt))
    theta = 1 / coherence_time
    dx = theta * (drift - x) * dt + sigma * np.random.normal(size=x.shape, scale=dt)
    return dx


def interpolate_and_smooth(x, y, sigma=None):
    """Interpolates with cublic spline x and y to 10x resolution then smooths these with a gaussian kernel of width sigma.
    Currently this only works for 1-dimensional x.
    Args:
        x
        y
        sigma
    Returns (x_new,y_new)
    """
    from scipy.ndimage.filters import gaussian_filter1d
    from scipy.interpolate import interp1d

    y_cubic = interp1d(x, y, kind="cubic")
    x_new = np.arange(x[0], x[-1], (x[1] - x[0]) / 10)
    y_interpolated = y_cubic(x_new)
    if sigma is not None:
        y_smoothed = gaussian_filter1d(
            y_interpolated, sigma=sigma / (x_new[1] - x_new[0])
        )
        return x_new, y_smoothed
    else:
        return x_new, y_interpolated


def normal_to_rayleigh(x, sigma=1):
    """Converts a normally distributed variable (mean 0, var 1) to a rayleigh distributed variable (sigma)
    """
    x = stats.norm.cdf(x)  # norm to uniform)
    x = sigma * np.sqrt(-2 * np.log(1 - x))  # uniform to rayleigh
    return x


def rayleigh_to_normal(x, sigma=1):
    """Converts a rayleigh distributed variable (sigma) to a normally distributed variable (mean 0, var 1)
    """
    x = 1 - np.exp(-(x ** 2) / (2 * sigma ** 2))  # rayleigh to uniform
    x = min(max(1e-6, x), 1 - 1e-6)
    x = stats.norm.ppf(x)  # uniform to normal
    return x


def gaussian(x, mu, sigma, norm=None):
    """Gaussian function. x, mu and sigma can be any shape as long as they are all the same (or strictly, all broadcastable)
    Args:
        x: input
        mu ; mean
        sigma; standard deviation
        norm: if provided the maximum value will be the norm
    Returns gaussian(x;mu,sigma)
    """
    g = -((x - mu) ** 2)
    g = g / (2 * sigma ** 2)
    g = np.exp(g)
    norm = norm or (1 / (np.sqrt(2 * np.pi * sigma ** 2)))
    g = g * norm
    return g


def von_mises(theta, mu, sigma, norm=None):
    """Von Mises function. theta, mu and sigma can be any shape as long as they are all the same (or strictly, all broadcastable).
    sigma is the standard deviation (in radians) which is converted to the von mises spread parameter,
    kappa = 1 / sigma^2 (note this approximation is only true for small, sigma << 2pi, spreads). All quantities must be given in radians.
    Args:
        x: input
        mu ; mean
        sigma; standard deviation
        norm: if provided the maximum (i.e. in the centre) value will be the norm
    Returns von_mises(x;mu,sigma)
    """
    kappa = 1 / (sigma ** 2)
    v = np.exp(kappa * np.cos(theta - mu))
    norm = norm or (np.exp(kappa) / (2 * np.pi * scipy.special.i0(kappa)))
    norm = norm / np.exp(kappa)
    v = v * norm
    return v


"""Plotting functions"""


def bin_data_for_histogramming(data, extent, dx, weights=None):
    """Bins data ready for plotting.
    So for example if the data is 1D the extent is broken up into bins (leftmost edge = extent[0], rightmost edge = extent[1]) and then data is
    histogrammed into these bins.
    weights weights the histogramming process so the contribution of each data point to a bin count is the weight, not 1.

    Args:
        data (array): (2,N) for 2D or (N,) for 1D)
        extent (_type_): _description_
        dx (_type_): _description_
        weights (_type_, optional): _description_. Defaults to None.

    Returns:
        (heatmap,bin_centres): if 1D
        (heatmap): if 2D
    """
    if len(extent) == 2:  # dimensionality = "1D"
        bins = np.arange(extent[0], extent[1] + dx, dx)
        heatmap, xedges = np.histogram(data, bins=bins, weights=weights)
        centres = (xedges[1:] + xedges[:-1]) / 2
        return (heatmap, centres)

    elif len(extent) == 4:  # dimensionality = "2D"
        bins_x = np.arange(extent[0], extent[1] + dx, dx)
        bins_y = np.arange(extent[2], extent[3] + dx, dx)
        heatmap, xedges, yedges = np.histogram2d(
            data[:, 0], data[:, 1], bins=[bins_x, bins_y], weights=weights
        )
        heatmap = heatmap.T[::-1, :]
        return heatmap


def mountain_plot(
    X,
    NbyX,
    color="C0",
    xlabel="",
    ylabel="",
    xlim=None,
    fig=None,
    ax=None,
    norm_by="max",
    overlap=0.8,
    shift=4,
    **kwargs,
):
    """Make a mountain plot.
    NbyX is an N by X array of all the plots to display.
    The nth plot is shown at height n, line are scaled so the maximum value across all of them is 0.7,
    then they are all seperated by 1 (sot they don't overlap)

    Args:
        X: independent variable to go on X axis
        NbyX: dependent variables to go on y axis
        color: plot color. Defaults to "C0".
        xlabel (str, optional): x axis label. Defaults to "".
        ylabel (str, optional): y axis label. Defaults to "".
        xlim (_type_, optional): fix xlim to this is desired. Defaults to None.
        fig (_type_, optional): fig to plot over if desired. Defaults to None.
        ax (_type_, optional): ax to plot on if desider. Defaults to None.
        norm_by: what to normalise each line of the mountainplot by.
           If "max", norms by the maximum firing rate found across all the neurons. Otherwise, pass a float (useful if you want to compare different neural datsets apples-to-apples)
        overlap: how much each plots overlap by (> 1 = overlap, < 1 = no overlap) (overlap is not relevant if you also set "norm_by")
        shift: distance between lines in mm

    Returns:
        fig, ax: _description_
    """
    c = color or "C1"
    c = np.array(matplotlib.colors.to_rgb(c))
    fc = 0.3 * c + (1 - 0.3) * np.array([1, 1, 1])  # convert rgb+alpha to rgb

    if norm_by == "max":
        NbyX = overlap * NbyX / np.max(np.abs(NbyX))
    else:
        NbyX = overlap * NbyX / norm_by
    if fig is None and ax is None:
        fig, ax = plt.subplots(
            figsize=(4, len(NbyX) * shift / 25)
        )  # ~<shift>mm gap between lines
    zorder = 1
    for i in range(len(NbyX)):
        ax.plot(X, NbyX[i] + i + 1, c=c, zorder=zorder)
        zorder -= 0.01
        ax.fill_between(X, NbyX[i] + i + 1, i + 1, color=fc, zorder=zorder, alpha=0.9)
        zorder -= 0.01
    ax.spines["left"].set_bounds(1, len(NbyX))
    ax.spines["bottom"].set_position(("outward", 1))
    ax.spines["left"].set_position(("outward", 1))
    ax.set_yticks([1, len(NbyX)])
    ax.set_ylim(1 - 0.5, len(NbyX) + 1.1*overlap)
    ax.set_xticks(np.arange(max(X + 0.1)))
    ax.spines["left"].set_color(None)
    ax.spines["right"].set_color(None)
    ax.spines["top"].set_color(None)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim()
    if xlim is not None:
        ax.set_xlim(right=xlim)

    return fig, ax


"""Other"""


def update_class_params(Class, params: dict):
    """Updates parameters from a dictionary.
    All parameters found in params will be updated to new value
    Args:
        params (dict): dictionary of parameters to change
        initialise (bool, optional): [description]. Defaults to False.
    """
    for key, value in params.items():
        setattr(Class, key, value)


def activate(x, activation="sigmoid", deriv=False, other_args={}):
    """Activation function function

    Args:
        x (the input (vector))
        activation: which type of fucntion to use (this is overwritten by 'activation' key in other_args)
        deriv (bool, optional): Whether it return f(x) or df(x)/dx. Defaults to False.
        other_args: Dictionary of parameters including other_args["activation"] = str for what type of activation (sigmoid, linear) and other params e.g.
          sigmoid midpoi n, max firing rate...
        Oother args my contain your own bespoke activation function under key other_args["function"]

    Returns:
        f(x) or df(x)/dx : array same as x
    """
    # write your own:
    try:
        my_activation_function = other_args["function"]
        return my_activation_function(x, deriv=deriv)
    except KeyError:
        pass

    # otherwise use on of these
    try:
        name = other_args["activation"]
    except KeyError:
        name = activation

    assert name in [
        "linear",
        "sigmoid",
        "relu",
        "tanh",
        "retanh",
    ]

    if name == "linear":
        if deriv == False:
            return x
        elif deriv == True:
            return np.ones(x.shape)

    if name == "sigmoid":
        # default sigmoid parameters set so that
        # max_f = max firing rate = 1
        # mid_x = middle point on domain = 1
        # width_x = distance from 5percent_x to 95percent_x = 1
        other_args_default = {"max_fr": 1, "min_fr": 0, "mid_x": 1, "width_x": 2}
        other_args_default.update(other_args)
        other_args = other_args_default
        max_fr, min_fr, width_x, mid_x = (
            other_args["max_fr"],
            other_args["min_fr"],
            other_args["width_x"],
            other_args["mid_x"],
        )
        beta = np.log((1 - 0.05) / 0.05) / (0.5 * width_x)  # sigmoid width
        if deriv == False:
            return ((max_fr - min_fr) / (1 + np.exp(-beta * (x - mid_x)))) + min_fr
        elif deriv == True:
            f = activate(x, deriv=False, other_args=other_args)
            return beta * (f - min_fr) * (1 - (f - min_fr) / (max_fr - min_fr))

    if name == "relu":
        other_args_default = {"gain": 1, "threshold": 0}
        for key in other_args.keys():
            other_args_default[key] = other_args[key]
        other_args = other_args_default
        if deriv == False:
            return other_args["gain"] * np.maximum(0, x - other_args["threshold"])
        elif deriv == True:
            return other_args["gain"] * ((x - other_args["threshold"]) > 0)

    if name == "tanh":
        other_args_default = {"gain": 1, "threshold": 0}
        for key in other_args.keys():
            other_args_default[key] = other_args[key]
        other_args = other_args_default
        if deriv == False:
            return other_args["gain"] * np.tanh(x - other_args["threshold"])
        elif deriv == True:
            return other_args["gain"] * (1 - np.tanh(x) ** 2)

    if name == "retanh":
        other_args_default = {"gain": 1, "threshold": 0}
        for key in other_args.keys():
            other_args_default[key] = other_args[key]
        other_args = other_args_default
        if deriv == False:
            return other_args["gain"] * np.maximum(
                0, np.tanh(x - other_args["threshold"])
            )
        elif deriv == True:
            return (
                other_args["gain"] * (1 - np.tanh(x) ** 2) * ((x - other_args["threshold"]) > 0))
