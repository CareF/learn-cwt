"""Transfer matrix method for simple TE Hermisian waveguide modes."""

import numpy as np
import scipy.optimize


def _transfer_matrix_te_single(layer_width: np.ndarray, layer_eps: np.ndarray,
                               k0: float, beta: float):
    # layer_alpha_sq = beta**2 - k0**2 * layer_eps
    # tmatrices = np.empty((layer_width.size, 2, 2))
    # for n, (alpha_sq, width) in enumerate(zip(layer_alpha_sq, layer_width)):
    #     if alpha_sq > 0:
    #         alpha = np.sqrt(alpha_sq)
    #         phi = alpha * width
    #         tmatrix = np.array([[np.cosh(phi), np.sinh(phi) / alpha],
    #                             [np.sinh(phi) * alpha, np.cosh(phi)]])
    #     else:
    #         k = np.sqrt(-alpha_sq)
    #         phi = k * width
    #         tmatrix = np.array([[np.cos(phi), np.sinc(phi / np.pi) * width],
    #                             [-k * np.sin(phi), np.cos(phi)]])
    #     tmatrices[n] = tmatrix
    # return tmatrices
    ks = np.sqrt(k0**2 * layer_eps - beta**2 + 0j)
    phis = ks * layer_width
    tmatrices = np.moveaxis(np.array(
        [[np.cos(phis), np.sinc(phis / np.pi) * layer_width],
         [-ks * np.sin(phis), np.cos(phis)]]
    ), -1, 0)
    assert np.abs(tmatrices.imag).max() < 1e-6, f"{tmatrices=}, {phis=}, {k0=}"
    return tmatrices.real


def transfer_matrix_te(layer_width: np.ndarray, layer_eps: np.ndarray,
                       k0: float, beta: float):
    """
    Solve for the total transfer matrix of
    \\partial_z^2 E + eps(z) k0^2 E = \\beta^2 E

    The transfer matrix here is defined as:
      (E(z_i), E'(z_i))^T = M_i * (E(z_{i-1}), E'(z_{i-1}))^T
      M = \\prid_i M_i
    where M_i is the transfer matrix of the i-th layer and M is the total
    transfer matrix.

    len(layer_width) == len(layer_eps).
    """
    assert layer_width.size == layer_eps.size
    assert np.all(layer_width > 0)
    return np.linalg.multi_dot(_transfer_matrix_te_single(
        layer_width, layer_eps, k0, beta))


def guided_mode_te(layer_width: np.ndarray, layer_eps: np.ndarray,
                   beta: float):
    """
    Find the k0 (= omega/c) of the first guided mode. When no guided mode is
    found, return NaN. If there are multiple guided modes, return the first.

    Guided mode are when the field on both sides of the structure decays
    exponentially. Mathematically it means that

    len(layer_eps) = len(layer_width) - 2, the first and last layer are
    the top and bottom environment, which are assumed to be infinite.
    """
    class _KNotFoundError(Exception):
        pass

    # E(z < 0) = A exp(alpha_i z) --> (E, E') ~ (1, alpha_i)
    # E(z > L) = B exp(-alpha_f z) --> (E, E') ~ (1, -alpha_f)
    # M (1, alpha_i)^T ~ (1, -alpha_f)^T -->
    # alpha_f M_11 + alpha_i alpha_f M_12 + M_21 + alpha_i M_22 = 0
    # The guided mode corresponds to k0 that satisfies the above equation.
    k_max = np.min(beta / np.sqrt(layer_eps[[0, -1]]))

    def determinant(k0):
        if k0 > k_max or k0 < 0:
            raise _KNotFoundError(f"k out of range {k0=}, {k_max=}")
        alpha_i = np.sqrt(beta**2 - layer_eps[0] * k0**2)
        alpha_f = np.sqrt(beta**2 - layer_eps[-1] * k0**2)
        transfer_mat = transfer_matrix_te(layer_width, layer_eps[1:-1],
                                          k0, beta)
        return np.array([alpha_f, 1]) @ transfer_mat @ np.array([1, alpha_i])
    eps = 1E-5
    # ks = np.linspace(1.8, k_max, 1000)
    # plt.plot(ks, [determinant(k) for k in ks])
    # plt.axhline(0, color='k', linestyle='--')
    # plt.ylim(-10, 10)
    try:
        k_trial = np.linspace(eps, k_max - eps, 100)
        dets = [determinant(k) for k in k_trial]
        try:
            # find the first index dets switches sign
            idx = np.where(np.diff(np.signbit(dets)))[0][0]
        except IndexError:
            # no sign change
            return np.nan
        k0, r = scipy.optimize.brentq(determinant,
                                      k_trial[idx], k_trial[idx+1],
                                      full_output=True, disp=False)
    except _KNotFoundError:
        return np.nan
    if not r.converged:
        return np.nan
    return k0


def populate_layer_var(z: np.ndarray, layer_width: np.ndarray,
                       layer_var: np.ndarray):
    """Return the layered variable at z.
    len(layer_var) == len(layer_width) + 1, the first and last layer are the
    environment variables.

    For visualization and numerical integration purpose.
    """
    layer_pos = [-np.inf, 0.0, *np.cumsum(layer_width), np.inf]
    ret = np.empty_like(z)
    for n, var in enumerate(layer_var):
        ret[(z >= layer_pos[n]) & (z < layer_pos[n+1])] = var
    return ret


def populate_mode_te(z: np.ndarray, k0: float, beta: float,
                     layer_width: np.ndarray, layer_eps: np.ndarray,
                     ):
    """Return E(z) of the TE mode, assuming (E(0), E'(0)) ~ (1, alpha_i),
    normalized to \\int eps(z)|E|^2 dz = 1.

    z = 0 is the start of the first layer, z < 0 corresponds to layer_eps[0].
    """
    field = np.empty_like(z)
    layer_pos = np.cumsum([0, *layer_width])
    ks_sq = k0**2 * layer_eps - beta**2
    alpha_i = np.sqrt(-ks_sq[0])
    alpha_f = np.sqrt(-ks_sq[-1])
    ks = np.sqrt(ks_sq[1:-1] + 0j)

    field_inter = np.empty((layer_width.size + 1, 2))
    field_inter[0] = np.array([1, alpha_i])
    tmatrices = _transfer_matrix_te_single(layer_width, layer_eps[1:-1],
                                           k0, beta)
    for n, tmatrix in enumerate(tmatrices):
        field_inter[n + 1] = tmatrix @ field_inter[n]

    field[z < 0] = field_inter[0, 0] * np.exp(alpha_i * z[z < 0])
    # in case of bounded mode
    # field[z >= layer_pos[-1]] = field_inter[-1, 0] * np.exp(
    #     -alpha_f * (z[z >= layer_pos[-1]] - layer_pos[-1]))
    last_mask = z >= layer_pos[-1]
    last_phi = alpha_f * (z[last_mask] - layer_pos[-1])
    field[last_mask] = field_inter[-1] @ np.array(
        [np.cosh(last_phi), np.sinh(last_phi) / alpha_f])
    for n, (k, field_left) in enumerate(zip(ks, field_inter[:-1])):
        mask = (z >= layer_pos[n]) & (z < layer_pos[n + 1])
        z_layer = z[mask] - layer_pos[n]
        phi_layer = k * z_layer
        field_cmpx = field_left @ np.array([
            np.cos(phi_layer), np.sinc(phi_layer / np.pi) * z_layer])
        assert np.abs(field_cmpx.imag).max() < 1e-6, f"{field_cmpx=}"
        field[mask] = field_cmpx.real
    field /= np.trapezoid(
        field**2 * populate_layer_var(z, layer_width, layer_eps), z)
    return field
