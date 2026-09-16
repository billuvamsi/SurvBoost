import numbers
from typing import Optional

import numpy as np
import scipy.special as sc
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import Interval, StrOptions
from sksurv.base import SurvivalAnalysisMixin
from sksurv.functions import StepFunction
from sksurv.util import check_array_survival
from torch.autograd import Variable

__all__ = ["Model"]


class GammaIncc(torch.autograd.Function):
    """Custom autograd function for Regularized Incomplete Gamma survival."""
    @staticmethod
    def forward(ctx, a, x):
        ctx.save_for_backward(a, x)
        return torch.special.gammaincc(a, x)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, x = ctx.saved_tensors
        # Gamma(a, x) backward w.r.t x
        grad_x = -torch.exp((a - 1) * torch.log(x + 1e-8) - x - torch.lgamma(a)) * grad_output
        
        # Central difference for gradient w.r.t a because torch lacks this derivative
        eps = 1e-4
        fa = torch.special.gammaincc(a + eps, x)
        ba = torch.special.gammaincc(torch.clamp(a - eps, min=1e-8), x)
        grad_a = (fa - ba) / (2 * eps) * grad_output
        return grad_a, grad_x


class SurvBoost(SurvivalAnalysisMixin, BaseEstimator):
    r"""Gradient boosting for survival data using composition of fully parametric distributions.

     SurvBoost with LogNormal and Generalized Gamma heads.

    Parameters
    ----------
    weibull_heads : int, default=2
    loglogistic_heads : int, default=2
    lognormal_heads : int, default=2
    gengamma_heads : int, default=2
    n_estimators : int, default=100
    max_depth : int, default=1
    learning_rate : float, default=0.1
    alpha : float, default=0.0
    l1_ratio : float, default=0.5
    uniform_heads : bool, default=False
    heads_activation : {'relu', 'softmax'}, default='relu'
    random_state : int, optional
    """

    _parameter_constraints = {
        "weibull_heads": [Interval(numbers.Integral, 0, None, closed="left")],
        "loglogistic_heads": [Interval(numbers.Integral, 0, None, closed="left")],
        "lognormal_heads": [Interval(numbers.Integral, 0, None, closed="left")],
        "gengamma_heads": [Interval(numbers.Integral, 0, None, closed="left")],
        "n_estimators": [Interval(numbers.Integral, 1, None, closed="left")],
        "max_depth": [Interval(numbers.Integral, 1, None, closed="left")],
        "learning_rate": [Interval(numbers.Real, 0, None, closed="neither")],
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
        "l1_ratio": [Interval(numbers.Real, 0, 1, closed="both")],
        "uniform_heads": [bool],
        "heads_activation": [StrOptions({"relu", "softmax"})],
    }

    def __init__(
        self,
        weibull_heads: int = 2,
        loglogistic_heads: int = 2,
        lognormal_heads: int = 2,
        gengamma_heads: int = 2,
        n_estimators: int = 100,
        max_depth: int = 1,
        learning_rate: float = 0.1,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        uniform_heads: bool = False,
        heads_activation: str = "relu",
        random_state: Optional[int] = None,
    ):
        self.weibull_heads = weibull_heads
        self.loglogistic_heads = loglogistic_heads
        self.lognormal_heads = lognormal_heads
        self.gengamma_heads = gengamma_heads

        self.heads = (
            weibull_heads + loglogistic_heads + lognormal_heads + gengamma_heads
        )
        if self.heads == 0:
            self.weibull_heads = 1
            self.heads = 1

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.uniform_heads = uniform_heads
        self.heads_activation = heads_activation
        self.random_state = random_state

        self._base_timeline = np.linspace(0, 1, 100)

    # Indexes
    @property
    def _weibull_slice(self):
        return slice(0, self.weibull_heads)

    @property
    def _loglogistic_slice(self):
        s = self.weibull_heads
        return slice(s, s + self.loglogistic_heads)

    @property
    def _lognormal_slice(self):
        s = self.weibull_heads + self.loglogistic_heads
        return slice(s, s + self.lognormal_heads)

    @property
    def _gengamma_slice(self):
        s = self.weibull_heads + self.loglogistic_heads + self.lognormal_heads
        return slice(s, s + self.gengamma_heads)

    def _init_state(self):
        seed = np.random.default_rng(self.random_state)
        self.init_eta_ = seed.random(self.heads) + 0.5
        self.eta_heads_ = [[] for _ in range(self.heads)]
        self.init_k_ = seed.random(self.heads) * 2
        self.k_heads_ = [[] for _ in range(self.heads)]
        self.init_rho_ = seed.random(self.heads) * 2
        self.rho_heads_ = [[] for _ in range(self.heads)]
        self.init_w_ = seed.random(self.heads)
        self.w_heads_ = [[] for _ in range(self.heads)]

    def _predict_component(self, X, init_vals, regs_list):
        output = np.ones((len(X), self.heads)) * init_vals.reshape((1, -1))
        for i, regs in enumerate(regs_list):
            if len(regs) == 0:
                continue
            preds = np.concatenate([reg.predict(X).reshape((-1, 1)) for reg in regs], axis=1)
            output[:, i] += self.learning_rate * np.sum(preds, axis=1)
        return output

    def _predict_ws(self, X):
        if self.uniform_heads:
            return np.ones((len(X), self.heads)) / self.heads
        return self._predict_component(X, self.init_w_, self.w_heads_)

    def _predict_params(self, X):
        etas = self._predict_component(X, self.init_eta_, self.eta_heads_).reshape((-1, self.heads, 1))
        ks = self._predict_component(X, self.init_k_, self.k_heads_).reshape((-1, self.heads, 1))
        rhos = self._predict_component(X, self.init_rho_, self.rho_heads_).reshape((-1, self.heads, 1))
        ws = self._predict_ws(X).reshape((-1, self.heads, 1))
        return np.concatenate([etas, ks, rhos, ws], -1)

    # --- Distributions ---
    # Weibull
    def _weibull_hazard(self, eta, k, times):
        return k * eta * times ** (k - 1)

    def _weibull_cum_hazard(self, eta, k, times):
        return eta * times ** k

    # Log-Logistic
    def _loglogistic_hazard(self, eta, k, times):
        return eta * k * times ** (k - 1) / (1 + eta * times ** k)

    def _loglogistic_cum_hazard(self, eta, k, times):
        if torch.is_tensor(times):
            return torch.log1p(eta * times ** k)
        return np.log1p(eta * times ** k)

    # Log-Normal (NOVEL)
    def _lognormal_hazard(self, eta, k, times):
        if torch.is_tensor(times):
            mu = torch.log(eta + 1e-8)
            sigma = 1.0 / (k + 1e-8)
            z = (torch.log(times + 1e-12) - mu) / sigma
            log_f = -torch.log(times + 1e-12) - torch.log(sigma) - 0.9189385332 - 0.5 * z**2
            s = 0.5 * torch.special.erfc(z / 1.41421356237)
            return torch.exp(log_f) / torch.clamp(s, min=1e-8, max=1.0)
        else:
            mu = np.log(eta + 1e-8)
            sigma = 1.0 / (k + 1e-8)
            z = (np.log(times + 1e-12) - mu) / sigma
            log_f = -np.log(times + 1e-12) - np.log(sigma) - 0.9189385332 - 0.5 * z**2
            s = 0.5 * sc.erfc(z / 1.41421356237)
            return np.exp(log_f) / np.clip(s, 1e-8, 1.0)

    def _lognormal_cum_hazard(self, eta, k, times):
        if torch.is_tensor(times):
            mu = torch.log(eta + 1e-8)
            sigma = 1.0 / (k + 1e-8)
            z = (torch.log(times + 1e-12) - mu) / sigma
            s = 0.5 * torch.special.erfc(z / 1.41421356237)
            return -torch.log(torch.clamp(s, min=1e-8, max=1.0))
        else:
            mu = np.log(eta + 1e-8)
            sigma = 1.0 / (k + 1e-8)
            z = (np.log(times + 1e-12) - mu) / sigma
            s = 0.5 * sc.erfc(z / 1.41421356237)
            return -np.log(np.clip(s, 1e-8, 1.0))

    # Generalized Gamma (NOVEL)
    def _gengamma_hazard(self, eta, k, rho, times):
        dp = torch.clamp(k / (rho + 1e-8), min=1e-2, max=100.0) if torch.is_tensor(times) else np.clip(k / (rho + 1e-8), 1e-2, 100.0)
        if torch.is_tensor(times):
            x = torch.clamp((times / (eta + 1e-8)) ** rho, min=1e-8, max=100.0)
            log_f = torch.log(rho + 1e-8) + (k - 1) * torch.log(times + 1e-8) - x - k * torch.log(eta + 1e-8) - torch.lgamma(dp)
            s = torch.clamp(GammaIncc.apply(dp, x), min=1e-8, max=1.0)
            return torch.exp(log_f) / s
        else:
            x = np.clip((times / (eta + 1e-8)) ** rho, 1e-8, 100.0)
            log_f = np.log(rho + 1e-8) + (k - 1) * np.log(times + 1e-8) - x - k * np.log(eta + 1e-8) - sc.gammaln(dp)
            s = np.clip(sc.gammaincc(dp, x), 1e-8, 1.0)
            return np.exp(log_f) / s

    def _gengamma_cum_hazard(self, eta, k, rho, times):
        dp = torch.clamp(k / (rho + 1e-8), min=1e-2, max=100.0) if torch.is_tensor(times) else np.clip(k / (rho + 1e-8), 1e-2, 100.0)
        if torch.is_tensor(times):
            x = torch.clamp((times / (eta + 1e-8)) ** rho, min=1e-8, max=100.0)
            s = torch.clamp(GammaIncc.apply(dp, x), min=1e-8, max=1.0)
            return -torch.log(s)
        else:
            x = np.clip((times / (eta + 1e-8)) ** rho, 1e-8, 100.0)
            s = np.clip(sc.gammaincc(dp, x), 1e-8, 1.0)
            return -np.log(s)

    # --- Training ---
    def _get_neg_grads(self, params, events, times):
        params_torch = Variable(torch.tensor(params).float(), requires_grad=True)

        etas = F.relu(params_torch[:, :, 0])
        ks = F.relu(params_torch[:, :, 1])
        rhos = F.relu(params_torch[:, :, 2])
        if self.heads_activation == "relu":
            ws = F.relu(params_torch[:, :, 3])
        else:
            ws = F.softmax(params_torch[:, :, 3], dim=1)

        hazard = torch.zeros(len(times))
        cum_hazard = torch.zeros(len(times))

        if self.weibull_heads > 0:
            sl = self._weibull_slice
            hazard += (self._weibull_hazard(etas[:, sl], ks[:, sl], times) * ws[:, sl]).sum(dim=1)
            cum_hazard += (self._weibull_cum_hazard(etas[:, sl], ks[:, sl], times) * ws[:, sl]).sum(dim=1)

        if self.loglogistic_heads > 0:
            sl = self._loglogistic_slice
            hazard += (self._loglogistic_hazard(etas[:, sl], ks[:, sl], times) * ws[:, sl]).sum(dim=1)
            cum_hazard += (self._loglogistic_cum_hazard(etas[:, sl], ks[:, sl], times) * ws[:, sl]).sum(dim=1)

        if self.lognormal_heads > 0:
            sl = self._lognormal_slice
            hazard += (self._lognormal_hazard(etas[:, sl], ks[:, sl], times) * ws[:, sl]).sum(dim=1)
            cum_hazard += (self._lognormal_cum_hazard(etas[:, sl], ks[:, sl], times) * ws[:, sl]).sum(dim=1)

        if self.gengamma_heads > 0:
            sl = self._gengamma_slice
            hazard += (self._gengamma_hazard(etas[:, sl], ks[:, sl], rhos[:, sl], times) * ws[:, sl]).sum(dim=1)
            cum_hazard += (self._gengamma_cum_hazard(etas[:, sl], ks[:, sl], rhos[:, sl], times) * ws[:, sl]).sum(dim=1)

        hazard = torch.clamp(hazard, min=1e-8)

        log_likelihood = (events * torch.log(hazard) - cum_hazard).mean()
        l1_reg = torch.abs(params_torch).mean()
        l2_reg = (params_torch ** 2).mean()
        elastic_net_reg = self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg
        loss = -log_likelihood + self.alpha * elastic_net_reg

        loss.backward()
        grad = params_torch.grad.numpy()
        grad[np.isnan(grad)] = 0.0
        return -(grad / (np.abs(grad).max() + 1e-8))

    def _fit_base_learner(self, X, y) -> DecisionTreeRegressor:
        reg = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
        reg.fit(X, y)
        return reg

    def _fit(self, X, events, times) -> None:
        events = torch.tensor(events.copy()).float().reshape((-1,))
        times = torch.tensor(times.copy()).float().reshape((-1, 1))

        for _ in range(self.n_estimators):
            params = self._predict_params(X)
            neg_grads = self._get_neg_grads(params, events, times)
            eta_grads = neg_grads[:, :, 0]
            k_grads = neg_grads[:, :, 1]
            rho_grads = neg_grads[:, :, 2]
            w_grads = neg_grads[:, :, 3]

            for i in range(self.heads):
                self.eta_heads_[i].append(self._fit_base_learner(X, eta_grads[:, i]))
                self.k_heads_[i].append(self._fit_base_learner(X, k_grads[:, i]))
                self.rho_heads_[i].append(self._fit_base_learner(X, rho_grads[:, i]))
                if not self.uniform_heads:
                    self.w_heads_[i].append(self._fit_base_learner(X, w_grads[:, i]))

    def fit(self, X, y) -> "SurvBoost":
        self._validate_params()
        X = self._validate_data(X)
        events, times = check_array_survival(X, y)
        self.max_time_ = times.max()
        times = times / self.max_time_
        self.unique_times_ = np.unique(times)
        self._init_state()
        self._fit(X, events, times)
        return self

    def predict(self, X):
        X = self._validate_data(X, reset=False)
        cum_hazard = self._predict_cumulative_hazard(X, self._base_timeline)
        survival = np.exp(-cum_hazard)
        return -(survival.sum(axis=1) / len(self._base_timeline))

    def _predict_cumulative_hazard(self, X, times):
        check_is_fitted(self, "unique_times_")
        params = torch.tensor(self._predict_params(X)).float()
        etas = F.relu(params[:, :, 0]).numpy().reshape((-1, self.heads, 1))
        ks = F.relu(params[:, :, 1]).numpy().reshape((-1, self.heads, 1))
        rhos = F.relu(params[:, :, 2]).numpy().reshape((-1, self.heads, 1))
        ws = F.relu(params[:, :, 3]).numpy().reshape((-1, self.heads, 1)) if self.heads_activation == "relu" else F.softmax(params[:, :, 3], dim=1).numpy().reshape((-1, self.heads, 1))

        cum_hazard = np.zeros((len(X), len(times)))

        if self.weibull_heads > 0:
            sl = self._weibull_slice
            cum_hazard += (self._weibull_cum_hazard(etas[:, sl], ks[:, sl], times) * ws[:, sl]).sum(axis=1)

        if self.loglogistic_heads > 0:
            sl = self._loglogistic_slice
            cum_hazard += (self._loglogistic_cum_hazard(etas[:, sl], ks[:, sl], times) * ws[:, sl]).sum(axis=1)

        if self.lognormal_heads > 0:
            sl = self._lognormal_slice
            cum_hazard += (self._lognormal_cum_hazard(etas[:, sl], ks[:, sl], times) * ws[:, sl]).sum(axis=1)

        if self.gengamma_heads > 0:
            sl = self._gengamma_slice
            cum_hazard += (self._gengamma_cum_hazard(etas[:, sl], ks[:, sl], rhos[:, sl], times) * ws[:, sl]).sum(axis=1)

        return cum_hazard

    def predict_cumulative_hazard_function(self, X, return_array=False):
        times = self.unique_times_ if return_array else self._base_timeline
        cum_hazard = self._predict_cumulative_hazard(X, times)
        if return_array: return cum_hazard
        times = self.max_time_ * times
        return np.array([StepFunction(times, cum_hazard[i]) for i in range(len(X))])

    def predict_survival_function(self, X, return_array=False):
        times = self.unique_times_ if return_array else self._base_timeline
        cum_hazard = self._predict_cumulative_hazard(X, times)
        survival = np.exp(-cum_hazard)
        if return_array: return survival
        times = self.max_time_ * times
        return np.array([StepFunction(times, survival[i]) for i in range(len(X))])

    def get_head_info(self):
        return {
            "weibull": self.weibull_heads,
            "loglogistic": self.loglogistic_heads,
            "lognormal": self.lognormal_heads,
            "gengamma": self.gengamma_heads,
            "total": self.heads,
        }
