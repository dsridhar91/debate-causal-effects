import numpy as np
from scipy.special import logit, expit
from scipy.optimize import minimize

from helpers import truncate_by_g, mse, cross_entropy, truncate_all_by_g


def _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps):
    """
    Helper for psi_tmle_bin_outcome

    Returns q_\eps (t,x)
    (i.e., value of perturbed predictor at t, eps, x; where q_t0, q_t1, g are all evaluated at x
    """
    h = t * (1./g) - (1.-t) / (1. - g)
    full_lq = (1.-t)*logit(q_t0) + t*logit(q_t1)  # logit predictions from unperturbed model
    logit_perturb = full_lq + eps * h
    return expit(logit_perturb)


def psi_tmle_bin_outcome(q_t0, q_t1, g, t, y, truncate_level=0.05):
    # TODO: make me useable
    # solve the perturbation problem

    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    eps_hat = minimize(lambda eps: cross_entropy(y, _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps))
                       , 0., method='Nelder-Mead')

    eps_hat = eps_hat.x[0]

    def q1(t_cf):
        return _perturbed_model_bin_outcome(q_t0, q_t1, g, t_cf, eps_hat)

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    return np.mean(ite)


def psi_tmle_cont_outcome(q_t0, q_t1, g, t, y, eps_hat=None, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    g_loss = mse(g, t)
    h = t * (1.0/g) - (1.0-t) / (1.0 - g)
    full_q = (1.0-t)*q_t0 + t*q_t1 # predictions from unperturbed model

    if eps_hat is None:
        eps_hat = np.sum(h*(y-full_q)) / np.sum(np.square(h))

    def q1(t_cf):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q + eps_hat * h_cf

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    psi_tmle = np.mean(ite)

    # standard deviation computation relies on asymptotic expansion of non-parametric estimator, see van der Laan and Rose p 96
    ic = h*(y-q1(t)) + ite - psi_tmle
    psi_tmle_std = np.std(ic) / np.sqrt(t.shape[0])
    initial_loss = np.mean(np.square(full_q-y))
    final_loss = np.mean(np.square(q1(t)-y))

    # print("tmle epsilon_hat: ", eps_hat)
    # print("initial risk: {}".format(initial_loss))
    # print("final risk: {}".format(final_loss))

    return psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss


def psi_iptw(q_t0, q_t1, g, t, y, truncate_level=0.05):
    ite=(t / g - (1-t) / (1-g))*y
    return np.mean(truncate_by_g(ite, g, level=truncate_level)), np.std(truncate_by_g(ite, g, level=truncate_level))/np.sqrt(q_t0.shape[0])



def psi_aiptw(q_t0, q_t1, g, t, y, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    full_q = q_t0 * (1 - t) + q_t1 * t
    h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
    ite = h * (y - full_q) + q_t1 - q_t0

    return np.mean(ite), np.std(ite)/np.sqrt(ite.shape[0])



def psi_q_only(q_t0, q_t1, g, t, y, truncate_level=0.):
    ite = (q_t1 - q_t0)
    return np.mean(truncate_by_g(ite, g, level=truncate_level)), np.std(truncate_by_g(ite, g, level=truncate_level))/np.sqrt(q_t0.shape[0])


def psi_very_naive(t, y):
    return y[t == 1].mean() - y[t == 0].mean()


def ate_estimates(q_t0, q_t1, g, t, y, truncate_level=0.05):

    very_naive = psi_very_naive(t,y)
    q_only = psi_q_only(q_t0, q_t1, g, t, y, truncate_level=truncate_level)
    iptw = psi_iptw(q_t0, q_t1, g, t, y, truncate_level=truncate_level)
    aiptw = psi_aiptw(q_t0, q_t1, g, t, y, truncate_level=truncate_level)
    tmle = psi_tmle_cont_outcome(q_t0, q_t1, g, t, y, truncate_level=truncate_level)[:2]

    estimates = {'very_naive': very_naive,
                 'q_only': q_only,
                 'iptw': iptw,
                 'tmle': tmle,
                 'aiptw': aiptw}

    return estimates



def main():
    pass


if __name__ == "__main__":
    main()
