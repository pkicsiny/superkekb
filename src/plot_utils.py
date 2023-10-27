import xobjects as xo
import scipy
import numpy as np
import os
import matplotlib
from scipy.optimize import curve_fit 
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm
matplotlib.rcParams['font.size'] = 32
matplotlib.rcParams['figure.subplot.left'] = 0.18
matplotlib.rcParams['figure.subplot.bottom'] = 0.16
matplotlib.rcParams['figure.subplot.right'] = 0.92
matplotlib.rcParams['figure.subplot.top'] = 0.9
matplotlib.rcParams['figure.figsize'] = (12,8)
from matplotlib.collections import LineCollection


def get_all_edges(bool_img):
    """
    Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
    The returned array edges has he dimension (n, 2, 2).
    Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
    Note that the indices of a pixel also denote the coordinates of its lower left corner.
    """
    edges = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            edges.append(np.array([[i, j+1],
                                   [i+1, j+1]], dtype=float))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            edges.append(np.array([[i+1, j],
                                   [i+1, j+1]], dtype=float))
        # South
        if j == 0 or not bool_img[i, j-1]:
            edges.append(np.array([[i, j],
                                   [i+1, j]], dtype=float))
        # West
        if i == 0 or not bool_img[i-1, j]:
            edges.append(np.array([[i, j],
                                   [i, j+1]], dtype=float))

    if not edges:
        return np.zeros((0, 2, 2), dtype=float)
    else:
        return np.array(edges, dtype=float)
    
def close_loop_edges(edges):
    """
    Combine thee edges defined by 'get_all_edges' to closed loops around objects.
    If there are multiple disconnected objects a list of closed loops is returned.
    Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
    """

    loop_list = []
    while edges.size != 0:

        loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
        edges = np.delete(edges, 0, axis=0)

        while edges.size != 0:
            # Get next edge (=edge with common node)
            ij = np.nonzero((edges == loop[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                loop.append(loop[0])
                # Uncomment to to make the start of the loop invisible when plotting
                #loop.append(loop[1])
                break

            loop.append(edges[i, (j + 1) % 2, :])
            edges = np.delete(edges, i, axis=0)

        loop_list.append(np.array(loop))

    return loop_list


def plot_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    edges = get_all_edges(bool_img=bool_img)
    edges = edges - 0.5  # convert indices to coordinates; TODO adjust according to image extent
    outlines = close_loop_edges(edges=edges)
    cl = LineCollection(outlines, **kwargs)
    ax.add_collection(cl)
    
    
def gauss_fit(x, mu, sigma): 
    y = 1 / (np.sqrt(2*np.pi)*sigma) * np.exp(-.5 * ((x - mu) / sigma)**2) 
    return y 

def fit_gauss(data, save_path=None):
    
    # histogram of data
    counts, bins, patches = plt.hist(data, bins=100, density=1);
    
    # scipy gauss fit uses np.std and np.mean
    mu, sigma = scipy.stats.norm.fit(data)
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, best_fit_line, label=f"scipy: mean={mu:.2e}, std={sigma:.2e}")
    
    # custom gauss fit
    bin_centers = 0.5 * ( bins[1:] + bins[:-1] ) 
    par, cov = curve_fit(gauss_fit, bin_centers, counts, p0=[mu, sigma]) 
    best_fit_line_2 = scipy.stats.norm.pdf(bins, par[0], par[1])
    plt.plot(bins, best_fit_line_2, label=f"plot_utils.gauss_fit: mean={par[0]:.2e}, sigma={par[1]:.2e}")
    
    plt.legend(fontsize=12)
    if save_path is not None:
        plt.savefig(save_path)
        
    # return params from custom gauss fit
    return par[0], par[1]


def tablelegend(ax, col_labels=None, row_labels=None, title_label="", *args, **kwargs):
    
    import matplotlib.legend as mlegend
    from matplotlib.patches import Rectangle
    """
    Place a table legend on the axes.
    
    Creates a legend where the labels are not directly placed with the artists, 
    but are used as row and column headers, looking like this:
    
    title_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |
    
    
    Parameters
    ----------
    
    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.
        
    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.
        
    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.
        
    title_label : str, optional
        Label for the top left corner in the legend table.
        
    ncol : int
        Number of columns.
        

    Other Parameters
    ----------------
    
    Refer to `matplotlib.legend.Legend` for other parameters.
    
    """
    #################### same as `matplotlib.axes.Axes.legend` #####################
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')
    
    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        title_label = [title_label]
        
        # blank rectangle handle
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]
        
        # empty label
        empty = [""]
        
        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol
        
        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow
        
        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_

    
def standard_plot_co(data_tuple, params_tuple, n_iters, n_ip, record_freq=1, save_path=None, save_name=None, w=10):
    from datetime import datetime

    x_co_arr, y_co_arr, z_co_arr, px_co_arr, py_co_arr, delta_co_arr = data_tuple
    sigma_x, sigma_y, sigma_z, sigma_px, sigma_py, sigma_delta = params_tuple

    x_co_arr_norm     = np.array(    x_co_arr) / sigma_x
    y_co_arr_norm     = np.array(    y_co_arr) / sigma_y
    z_co_arr_norm     = np.array(    z_co_arr) / sigma_z 
    px_co_arr_norm    = np.array(   px_co_arr) / sigma_px
    py_co_arr_norm    = np.array(   py_co_arr) / sigma_py
    delta_co_arr_norm = np.array(delta_co_arr) / sigma_delta
    
    n_iters = int(n_iters)
    n_ip = int(n_ip)
    turns_arr = np.linspace(record_freq, n_iters, int(n_iters/record_freq))

    w = np.abs(w)
    x_co = np.mean(x_co_arr_norm[-w:])
    y_co = np.mean(y_co_arr_norm[-w:])
    z_co = np.mean(z_co_arr_norm[-w:])

    px_co    = np.mean(px_co_arr_norm[-w:])
    py_co    = np.mean(py_co_arr_norm[-w:])
    delta_co = np.mean(delta_co_arr_norm[-w:])

    fig, ax = plt.subplots(3,3, figsize=(36,30))
    
    ax[0,0].plot(turns_arr, x_co_arr_norm, c="b")
    ax[1,0].plot(turns_arr, y_co_arr_norm, c="b")
    ax[2,0].plot(turns_arr, z_co_arr_norm, c="b")

    ax[0,1].plot(turns_arr,    px_co_arr_norm, c="b")
    ax[1,1].plot(turns_arr,    py_co_arr_norm, c="b")
    ax[2,1].plot(turns_arr, delta_co_arr_norm, c="b")
    
    ax[0,2].plot(100*x_co_arr_norm, 100*   px_co_arr_norm, "bo", zorder=0)
    ax[1,2].plot(100*y_co_arr_norm, 100*   py_co_arr_norm, "bo", zorder=0)
    ax[2,2].plot(100*z_co_arr_norm, 100*delta_co_arr_norm, "bo", zorder=0)

    # first n iterations
    ax[0,2].plot(100*x_co_arr_norm[:5], 100*   px_co_arr_norm[:5], "go-", zorder=1)
    ax[1,2].plot(100*y_co_arr_norm[:5], 100*   py_co_arr_norm[:5], "go-", zorder=1)
    ax[2,2].plot(100*z_co_arr_norm[:5], 100*delta_co_arr_norm[:5], "go-", zorder=1)
    
    ax[0,0].axhline(x_co, c="r", label=f"$\mathrm{{x_{{co}}^{{[-{w:d}:]}}=}}${x_co:.2e} [$\mathrm{{\sigma_0}}$]")
    ax[1,0].axhline(y_co, c="r", label=f"$\mathrm{{y_{{co}}^{{[-{w:d}:]}}=}}${y_co:.2e} [$\mathrm{{\sigma_0}}$]")
    ax[2,0].axhline(z_co, c="r", label=f"$\mathrm{{z_{{co}}^{{[-{w:d}:]}}=}}${z_co:.2e} [$\mathrm{{\sigma_0}}$]")
    
    ax[0,1].axhline(   px_co, c="r", label=f"$\mathrm{{   p_{{x,co}}^{{[-{w:d}:]}}=}}${   px_co:.2e} [$\mathrm{{\sigma_0}}$]")
    ax[1,1].axhline(   py_co, c="r", label=f"$\mathrm{{   p_{{y,co}}^{{[-{w:d}:]}}=}}${   py_co:.2e} [$\mathrm{{\sigma_0}}$]")
    ax[2,1].axhline(delta_co, c="r", label=f"$\mathrm{{\delta_{{co}}^{{[-{w:d}:]}}=}}${delta_co:.2e} [$\mathrm{{\sigma_0}}$]")
    
    ax[0,2].scatter(100*x_co, 100*   px_co, c="r", marker="x", s=100, zorder=2, label=f"($\mathrm{{x^{{[-{w:d}:]}}_{{co}}}}$, $\mathrm{{     p_{{x,co}}^{{[-{w:d}:]}}}}$) [$\mathrm{{\sigma_0}}$, %]: ({100*x_co:.3e}, {100*   px_co:.3e})")
    ax[1,2].scatter(100*y_co, 100*   py_co, c="r", marker="x", s=100, zorder=2, label=f"($\mathrm{{y^{{[-{w:d}:]}}_{{co}}}}$, $\mathrm{{     p_{{y,co}}^{{[-{w:d}:]}}}}$) [$\mathrm{{\sigma_0}}$, %]: ({100*y_co:.3e}, {100*   py_co:.3e})")
    ax[2,2].scatter(100*z_co, 100*delta_co, c="r", marker="x", s=100, zorder=2, label=f"($\mathrm{{z^{{[-{w:d}:]}}_{{co}}}}$, $\mathrm{{\delta_{{  co}}^{{[-{w:d}:]}}}}$) [$\mathrm{{\sigma_0}}$, %]: ({100*z_co:.3e}, {100*delta_co:.3e})")
    
    ax[0,0].legend(fontsize=24)
    ax[1,0].legend(fontsize=24)
    ax[2,0].legend(fontsize=24)
    ax[0,1].legend(fontsize=24)
    ax[1,1].legend(fontsize=24)
    ax[2,1].legend(fontsize=24)
    ax[0,2].legend(fontsize=24)
    ax[1,2].legend(fontsize=24)
    ax[2,2].legend(fontsize=24)
    
    ax[0,0].set_xlabel("Collisions [1]")
    ax[1,0].set_xlabel("Collisions [1]")
    ax[2,0].set_xlabel("Collisions [1]")
    ax[0,1].set_xlabel("Collisions [1]")
    ax[1,1].set_xlabel("Collisions [1]")
    ax[2,1].set_xlabel("Collisions [1]")
    ax[0,2].set_xlabel(r"$\mathrm{x_{co}}$ [$\mathrm{\sigma_0}$, %]")
    ax[1,2].set_xlabel(r"$\mathrm{y_{co}}$ [$\mathrm{\sigma_0}$, %]")
    ax[2,2].set_xlabel(r"$\mathrm{z_{co}}$ [$\mathrm{\sigma_0}$, %]")
    
    ax[0,0].set_ylabel(r"$\mathrm{x_{co}}$ [$\mathrm{\sigma_0}$]")
    ax[1,0].set_ylabel(r"$\mathrm{y_{co}}$ [$\mathrm{\sigma_0}$]")
    ax[2,0].set_ylabel(r"$\mathrm{z_{co}}$ [$\mathrm{\sigma_0}$]")
    ax[0,1].set_ylabel(r"$\mathrm{   p_{x,co}}$ [$\mathrm{\sigma_0}$]")
    ax[1,1].set_ylabel(r"$\mathrm{   p_{y,co}}$ [$\mathrm{\sigma_0}$]")
    ax[2,1].set_ylabel(r"$\mathrm{\delta_{co}}$ [$\mathrm{\sigma_0}$]")
    ax[0,2].set_ylabel(r"$\mathrm{   p_{x,co}}$ [$\mathrm{\sigma_0}$, %]")
    ax[1,2].set_ylabel(r"$\mathrm{   p_{y,co}}$ [$\mathrm{\sigma_0}$, %]")
    ax[2,2].set_ylabel(r"$\mathrm{\delta_{co}}$ [$\mathrm{\sigma_0}$, %]")
    fig.tight_layout()

    today = datetime.today().strftime('%Y_%m_%d')
    
    # save the plot
    if save_path is not None:

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if save_name is None:
            save_name = f"{today}_standard_plot_co_{n_iters:d}_iters_{n_ip:d}_ip.png"
        else:
            save_name = f"{save_name}_standard_plot_co_{n_iters:d}_iters_{n_ip:d}_ip.png"
            
        plt.savefig(os.path.join(save_path, save_name), bbox_inches="tight")
    return fig, ax

        
def standard_plot_std(data_tuple, params_tuple, n_iters, n_ip, record_freq=1, anal_z_tuple_std=None, alive=None, save_path=None, save_name=None, w=10):
    from datetime import datetime

    x_std_arr, y_std_arr, z_std_arr, px_std_arr, py_std_arr, delta_std_arr, emit_x_arr, emit_y_arr, emit_s_arr = data_tuple
    sigma_x, sigma_y, sigma_z, sigma_px, sigma_py, sigma_delta, physemit_x, physemit_y, physemit_s = params_tuple

    x_std_arr     = np.array(x_std_arr    )
    y_std_arr     = np.array(y_std_arr    ) 
    z_std_arr     = np.array(z_std_arr    ) 
    px_std_arr    = np.array(px_std_arr   ) 
    py_std_arr    = np.array(py_std_arr   ) 
    delta_std_arr = np.array(delta_std_arr) 
    emit_x_arr    = np.array(emit_x_arr   ) 
    emit_y_arr    = np.array(emit_y_arr   ) 
    emit_s_arr    = np.array(emit_s_arr   ) 
    
    n_iters = int(n_iters)
    n_ip = int(n_ip)
    turns_arr = np.linspace(record_freq, n_iters, int(n_iters/record_freq))
    w = np.abs(w)
    
    fig, ax = plt.subplots(3,3, figsize=(36,30))
    
    # x, y, z
    ax[0,0].plot(turns_arr, x_std_arr, c="b", label=f"$\mathrm{{\sigma_{{eq}}^{{[-{w:d}:]}}/\sigma_0}}=${np.mean(x_std_arr[-w:])/sigma_x:.2e}")
    ax[1,0].plot(turns_arr, y_std_arr, c="b", label=f"$\mathrm{{\sigma_{{eq}}^{{[-{w:d}:]}}/\sigma_0}}=${np.mean(y_std_arr[-w:])/sigma_y:.2e}")
    ax[2,0].plot(turns_arr, z_std_arr, c="b", label=f"$\mathrm{{\sigma_{{eq}}^{{[-{w:d}:]}}/\sigma_0}}=${np.mean(z_std_arr[-w:])/sigma_z:.2e}")
    
    # px, py, delta
    ax[0,1].plot(turns_arr,    px_std_arr, c="b", label=f"$\mathrm{{\sigma_{{eq}}^{{[-{w:d}:]}}/\sigma_0}}=${np.mean(px_std_arr[-w:])/sigma_px:.2e}")
    ax[1,1].plot(turns_arr,    py_std_arr, c="b", label=f"$\mathrm{{\sigma_{{eq}}^{{[-{w:d}:]}}/\sigma_0}}=${np.mean(py_std_arr[-w:])/sigma_py:.2e}")
    ax[2,1].plot(turns_arr, delta_std_arr, c="b", label=f"$\mathrm{{\sigma_{{eq}}^{{[-{w:d}:]}}/\sigma_0}}=${np.mean(delta_std_arr[-w:])/sigma_delta:.2e}")
    
    # phase spaces
    ax[0,2].plot(turns_arr, emit_x_arr, c="b", label=f"$\mathrm{{\sigma_{{eq}}^{{[-{w:d}:]}}/\sigma_0}}=${np.mean(emit_x_arr[-w:])/physemit_x:.2e}")
    ax[1,2].plot(turns_arr, emit_y_arr, c="b", label=f"$\mathrm{{\sigma_{{eq}}^{{[-{w:d}:]}}/\sigma_0}}=${np.mean(emit_y_arr[-w:])/physemit_y:.2e}")
    ax[2,2].plot(turns_arr, emit_s_arr, c="b", label=f"$\mathrm{{\sigma_{{eq}}^{{[-{w:d}:]}}/\sigma_0}}=${np.mean(emit_s_arr[-w:])/physemit_s:.2e}")
    
    # alive is a 1D array of the number of alive particles
    if alive is not None:
        ax2 = ax[1,2].twinx()
        ax2.plot(turns_arr, alive, c="g")
        ax2.set_ylabel(r"$\mathrm{N_{m}}$ alive [1]")

    if anal_z_tuple_std is not None:
        sigma_z_eq, sigma_delta_eq, physemit_s_eq, emit_damping_rate_s = anal_z_tuple_std
        sigma_z_fit     = np.sqrt((    sigma_z**2-    sigma_z_eq**2) * np.exp(-emit_damping_rate_s*turns_arr) + sigma_z_eq**2)
        sigma_delta_fit = np.sqrt((sigma_delta**2-sigma_delta_eq**2) * np.exp(-emit_damping_rate_s*turns_arr) + sigma_delta_eq**2)
        emit_s_fit      =         (    physemit_s-    physemit_s_eq) * np.exp(-emit_damping_rate_s*turns_arr) + physemit_s_eq
        ax[2,0].plot(turns_arr,     sigma_z_fit, c="r", linewidth=3, label="formula")
        ax[2,1].plot(turns_arr, sigma_delta_fit, c="r", linewidth=3, label="formula")
        ax[2,2].plot(turns_arr,      emit_s_fit, c="r", linewidth=3, label="formula")

        
    
    ax[0,0].axhline(sigma_x, c="m", label=r"$\mathrm{\sigma_0}$")
    ax[1,0].axhline(sigma_y, c="m", label=r"$\mathrm{\sigma_0}$")
    ax[2,0].axhline(sigma_z, c="m", label=r"$\mathrm{\sigma_0}$")

    ax[0,1].axhline(   sigma_px, c="m", label=r"$\mathrm{\sigma_0}$")
    ax[1,1].axhline(   sigma_py, c="m", label=r"$\mathrm{\sigma_0}$")
    ax[2,1].axhline(sigma_delta, c="m", label=r"$\mathrm{\sigma_0}$")
    
    ax[0,2].axhline(physemit_x, c="m", label=r"$\mathrm{\sigma_0}$")
    ax[1,2].axhline(physemit_y, c="m", label=r"$\mathrm{\sigma_0}$")
    ax[2,2].axhline(physemit_s, c="m", label=r"$\mathrm{\sigma_0}$")
    if anal_z_tuple_std is not None:
        ax[2,0].axhline(    sigma_z_eq, c="k", label=r"$\mathrm{\sigma_{eq}}$")
        ax[2,1].axhline(sigma_delta_eq, c="k", label=r"$\mathrm{\sigma_{eq}}$")
        ax[2,2].axhline( physemit_s_eq, c="k", label=r"$\mathrm{\sigma_{eq}}$")
    
    ax[0,0].legend(fontsize=24)
    ax[1,0].legend(fontsize=24)
    ax[2,0].legend(fontsize=24)
    ax[0,1].legend(fontsize=24)
    ax[1,1].legend(fontsize=24)
    ax[2,1].legend(fontsize=24)
    ax[0,2].legend(fontsize=24)
    ax[1,2].legend(fontsize=24)
    ax[2,2].legend(fontsize=24)
    
    ax[0,0].set_xlabel("Collisions [1]")
    ax[1,0].set_xlabel("Collisions [1]")
    ax[2,0].set_xlabel("Collisions [1]")
    ax[0,1].set_xlabel("Collisions [1]")
    ax[1,1].set_xlabel("Collisions [1]")
    ax[2,1].set_xlabel("Collisions [1]")
    ax[0,2].set_xlabel("Collisions [1]")
    ax[1,2].set_xlabel("Collisions [1]")
    ax[2,2].set_xlabel("Collisions [1]")
    
    ax[0,0].set_ylabel(r"$\mathrm{\sigma_{x}}$ [m]")
    ax[1,0].set_ylabel(r"$\mathrm{\sigma_{y}}$ [m]")
    ax[2,0].set_ylabel(r"$\mathrm{\sigma_{z}}$ [m]")
    ax[0,1].set_ylabel(r"$\mathrm{\sigma_{p_x}}$ [1]")
    ax[1,1].set_ylabel(r"$\mathrm{\sigma_{p_y}}$ [1]")
    ax[2,1].set_ylabel(r"$\mathrm{\sigma_{\delta}}$ [1]")
    ax[0,2].set_ylabel(r"$\mathrm{\varepsilon_{x}}$ [m]")
    ax[1,2].set_ylabel(r"$\mathrm{\varepsilon_{y}}$ [m]")
    ax[2,2].set_ylabel(r"$\mathrm{\varepsilon_{s}}$ [m]")
    fig.tight_layout()

    today = datetime.today().strftime('%Y_%m_%d')
    
    # save the plot
    if save_path is not None:

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if save_name is None:
            save_name = f"{today}_standard_plot_std_{n_iters:d}_iters_{n_ip:d}_ip.png"
        else:
            save_name = f"{save_name}_standard_plot_std_{n_iters:d}_iters_{n_ip:d}_ip.png"
            
        plt.savefig(os.path.join(save_path, save_name), bbox_inches="tight")
    return fig, ax
    
    
def plot_initialized_beam(beam_params, sim_params, xp_beam, plots_path):

    xp_beam.move(_context=xo.context_default)
    
    # check if beam is matched to the start of lattice e.g. RF or CS
    if "beta_x_start" in sim_params.keys():
        beta_x_norm = np.sqrt( sim_params['beta_x_start'] / beam_params["beta_x"] )
    else: 
        beta_x_norm = 1
    if "beta_y_start" in sim_params.keys():
        beta_y_norm = np.sqrt( sim_params['beta_y_start'] / beam_params["beta_y"] )
    else:
        beta_y_norm = 1

    matplotlib.rcParams['font.size'] = 22
    fig, ax, = plt.subplots(1, 3, figsize=(30, 15), subplot_kw=dict(projection='3d'))
    
    ax[0].plot(xp_beam.x[:2] / (beam_params["sigma_x"] * beta_x_norm), xp_beam.px[:2] / (beam_params["sigma_px"] / beta_x_norm), xp_beam.delta[:2] / beam_params["sigma_delta"], "ro", markersize=2)
    ax[0].plot(xp_beam.x[2:] / (beam_params["sigma_x"] * beta_x_norm), xp_beam.px[2:] / (beam_params["sigma_px"] / beta_x_norm), xp_beam.delta[2:] / beam_params["sigma_delta"], "bo", markersize=2, alpha=.03)
    
    ax[1].plot(xp_beam.y[:2] / (beam_params["sigma_y"] * beta_y_norm), xp_beam.py[:2] / (beam_params["sigma_py"] / beta_y_norm), xp_beam.delta[:2] / beam_params["sigma_delta"], "ro", markersize=2)
    ax[1].plot(xp_beam.y[2:] / (beam_params["sigma_y"] * beta_y_norm), xp_beam.py[2:] / (beam_params["sigma_py"] / beta_y_norm), xp_beam.delta[2:] / beam_params["sigma_delta"], "bo", markersize=2, alpha=.03)
    
    ax[2].plot(xp_beam.x[:2] / (beam_params["sigma_x"] * beta_x_norm), xp_beam.y[:2] / (beam_params["sigma_y"] * beta_y_norm), xp_beam.delta[:2] / beam_params["sigma_delta"], "ro", markersize=2)
    ax[2].plot(xp_beam.x[2:] / (beam_params["sigma_x"] * beta_x_norm), xp_beam.y[2:] / (beam_params["sigma_y"] * beta_y_norm), xp_beam.delta[2:] / beam_params["sigma_delta"], "bo", markersize=2, alpha=.03)
    
    
    labelpad=25
    ax[0].set_xlabel(f"x [$\sigma_x$]", labelpad=labelpad)
    ax[1].set_xlabel(f"y [$\sigma_y$]", labelpad=labelpad)
    ax[2].set_xlabel(f"x [$\sigma_x$]", labelpad=labelpad)
    
    ax[0].set_ylabel(f"$p_x$ [$\sigma_{{p_x}}$]", labelpad=labelpad)
    ax[1].set_ylabel(f"$p_y$ [$\sigma_{{p_y}}$]", labelpad=labelpad)
    ax[2].set_ylabel(f"y [$\sigma_y$]", labelpad=labelpad)
    
    ax[0].set_zlabel(f"$\delta$ [$\sigma_{{\delta}}$]", labelpad=labelpad)
    ax[1].set_zlabel(f"$\delta$ [$\sigma_{{\delta}}$]", labelpad=labelpad)
    ax[2].set_zlabel(f"$\delta$ [$\sigma_{{\delta}}$]", labelpad=labelpad)
   
    try: 
        title = f"{len(xp_beam.x):d} particles\n$J_{{max}}$={sim_params['j_max']} [$\sigma$], z=0 [m], $\delta_{{max}}$={sim_params['delta_max']} [$\sigma_{{\delta}}$]"
    except:
        title = f"{len(xp_beam.x):d} particles"
    plt.suptitle(title, y=0.8)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_path, f"initialized_beam.png"), bbox_inches="tight")


def plot_trajectory(coords_dict, beam_params, var_idx=0, particle_idx_list=[0], label_list=[""]):
    """
    coords_dict dimensions: (# var setups, # particles, # turns)
    """
    
    assert len(particle_idx_list) == len(label_list), "particle index and label list must be of equal length!"
    
    fig, ax = plt.subplots(3,2, figsize=(24, 24))
    
    keys = ["x", "px", "y", "py", "z", "delta"]
    units = ["[σ_x]", "[σ_px]", "[σ_y]", "[σ_py]", "[σ_z]", "[σ_δ]"]
    for r in range(3):
        for c in range(2):
            
            for i, p in enumerate(particle_idx_list):
                #print("[{}/{}]".format(i+1, len(particle_idx_list)))
                ax[r,c].plot(coords_dict["b1"][keys[int(2*r+c%2)]][var_idx][p]      /beam_params["sigma_{}".format(keys[int(2*r+c%2)])], label=label_list[i])
            
            ax[r,c].set_xlabel("Turn")
            ax[r,c].set_ylabel("{} {}".format(keys[int(2*r+c%2)], units[int(2*r+c%2)]))
            ax[r,c].legend()
            
    return fig


def freeze_header(df, num_rows=30, num_columns=10, step_rows=1,
                  step_columns=1):
    """
    idea: https://stackoverflow.com/questions/28778668/freeze-header-in-pandas-dataframe
    Freeze theheaders (column and index names) of a Pandas DataFrame. A widget
    enables t£o slide through the rows and columns.
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame to display
    num_rows : int, optional
        Number of rows to display
    num_columns : int, optional
        Number of columns to display
    step_rows : int, optional
        Step in the rows
    step_columns : int, optional
        Step in the columns
    Returns
    -------
    Displays the DataFrame with the widget
    """
    from ipywidgets import interact, IntSlider

    @interact(last_row=IntSlider(min=min(num_rows, df.shape[0]),
                                 max=df.shape[0],
                                 step=step_rows,
                                 description='rows',
                                 readout=False,
                                 disabled=False,
                                 continuous_update=True,
                                 orientation='horizontal',
                                 slider_color='purple'),
              last_column=IntSlider(min=min(num_columns, df.shape[1]),
                                    max=df.shape[1],
                                    step=step_columns,
                                    description='columns',
                                    readout=False,
                                    disabled=False,
                                    continuous_update=True,
                                    orientation='horizontal',
                                    slider_color='purple'))
    def _freeze_header(last_row, last_column):
        display(df.iloc[max(0, last_row-num_rows):last_row,
                        max(0, last_column-num_columns):last_column])
        

def hist(x,y):
    
    # calculate plot axis ranges
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    
    # divide range into 100 equal bins inclusive max and create mesh
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    # npart (z, xp) grid points equally distributed in axis ranges
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # npart (z, xp) coordinate pairs of weak beam particles
    values = np.vstack([x, y])
    
    # smooth values with gaussian kernel
    kernel = st.gaussian_kde(values)

    # sample the smoothed distribution at equidistant grid points and reshape into 100 x 100 array to be plotted
    f = np.reshape(kernel(positions).T, xx.shape)

    return xx, yy, f        
