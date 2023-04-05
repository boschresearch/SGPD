# -*- coding: utf-8 -*-
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Katharina Ensinger, katharina.ensinger@de.bosch.com

"""sparse GP implementation"""
from sparse_GP.kernel_utils.ARD_kernel import full_kernel
import itertools
import math
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Function, Variable, grad

FloatTensor = torch.FloatTensor
Parameter = torch.nn.Parameter


class MatheronGP(torch.nn.Module):

    """Implementation of sampling from weight space representation of GP

    Notes:
        Equations are based on [1]
        [1]: James T. Wilson, Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, Marc Peter Deisenroth:
        "Efficiently Sampling Functions from Gaussian Process Posteriors", International Conference on Machine Learning, 2020

    Args:
        d (int): dimension.
        n (int, optional): number of inducing inputs
        psi (torch.Tensor, optional): initial inducing inputs of shape (n, d)
        mu (torch.Tensor, optional): initial variational mean, i.e. mean of the process at the inducing points, of
            shape (n, 1)
        log_K (torch.Tensor, optional): initial representation of the log of the variational covariance matrix, of shape
            (n, 1). I.e. the vector contains the variances of the process at the inducing points. All covariances are
            considered zero
        device (torch.device): device
        log_L (torch.Tensor, optional): initial log lengthscale of the kernel function of shape (d,)
            (we assume an ARD-kernel)
        log_sigma_f (torch.Tensor, optional): torch.Tensor log of the variance of the ARD-kernel

    Attributes:
        dim (int): dimension
        n_latent (int): number of inducing inputs
        mu ((n_latent,1)-torch.Tensor, optional): variational mean
        log_K ((n_latent)-torch.Tensor, optional): logarithm of the variational covariance matrix
        log_L ((1,dim)-torch.Tensor, optional): log of the lengthscale
        log_sigma_f (torch.Tensor, optional): log of the GP variance
        psi ((n_latent, dim)-torch.Tensor): inducing inputs
        K_psi ((n_latent, n_latent)-torch.Tensor): kernel matrix at inducing points
        K_psi_cholesky ((n_latent, n_latent)-torch.Tensor): choleksy decomposition
        jitter_matrix (float): jitter * identity_matrix ist added to kernel for numerical stability
        n (int) : number of features for prior (corresponds to l in paper)
        theta = torch.Tensor((n, dim))
        prior_weights = torch.Tensor((1, 2 * self.n))
        mode (str out of "mean_predicions", "sample_predictions") = sampling method   
    """

    def __init__(
        self,
        d: int = 2,
        n: Optional[int] = None,
        psi: Optional[torch.tensor] = None,
        mu: Optional[torch.tensor] = None,
        log_K: Optional[torch.tensor] = None,
        device: torch.device = None,
        log_L: Optional[torch.Tensor] = None,
        log_sigma_f=None,
    ):
        super().__init__()
        self.device = "cpu"
        n = n if n is not None else 16
        psi = (
            psi
            if psi is not None
            else torch.tensor([[-2, -3]]).repeat(n, 1)
            + torch.mul(
                torch.tensor([[2, 3.5]]).repeat(n, 1) - torch.tensor([[-2, -3]]).repeat(n, 1), torch.rand((n, 1))
            )
        )
        mu = (
            mu if mu is not None else torch.zeros(n, 1, dtype=torch.float) + 0.05 * torch.randn(n, 1, dtype=torch.float)
        )
        log_K = log_K if log_K is not None else torch.log(1e-6 * torch.ones(n, dtype=torch.float))
        log_L = log_L if log_L is not None else torch.log(torch.sqrt((1.0) * torch.ones(1, 2, dtype=torch.float)))
        log_sigma_f = log_sigma_f if log_sigma_f is not None else torch.log(torch.tensor([0.001], dtype=torch.float))
        self.eps = Variable(torch.randn(1, 1, dtype=torch.float))
        self.dim = d
        self.n_latent = n
        self.mu = torch.nn.Parameter(mu, requires_grad=True)
        self.log_K = torch.nn.Parameter(log_K, requires_grad=True)
        self.log_L = torch.nn.Parameter(log_L, requires_grad=True)
        self.log_sigma_f = torch.nn.Parameter(log_sigma_f, requires_grad=True)
        self.psi = torch.nn.Parameter(psi, requires_grad=True)
        self.K_psi = torch.zeros(n, n, dtype=torch.float, device=self.device)
        self.K_psi_cholesky = torch.zeros(n, n, dtype=torch.float, device=self.device)
        self.jitter = 1e-7
        self.n = 10000
        self.to(self.device)
        self.theta = torch.normal(mean=0, std=1, size=(self.n, self.dim))
        self.prior_weights = torch.normal(mean=0, std=1, size=((1, 2 * self.n)))
        self.mode = "mean_predictions"
        self.parameter_list = {"psi":[], "mu":[], "log_K":[], "K_psi":[], "K_psi_cholesky":[], "log_L":[], "log_sigma_f":[], "theta":[], "weights":[], "targets":[]}
        self.local_theta = []
        self.local_weights = []
        self.local_targets = [] 
        
    @property
    def L_sqrt(self):
        return torch.exp(0.5 * self.log_L)
    
    @property    
    def L(self):
        return torch.exp(self.log_L)
    
    @property        
    def sigma_f(self):
        return torch.exp(self.log_sigma_f)
    
    @property    
    def K(self):
        return torch.exp(self.log_K).to(self.device)

    def kl(self):
        """ computing KL divergence between prior and variational posterior
            prior: zeros mean, covariance kernel K(psi,psi)
            posterior: mu_psi, K_psi
        """
        K_psi = self.full_kernel(self.psi, self.psi)
        dim = self.n_latent
        sig_x = torch.diag(self.K).to(self.device)
        p = torch.distributions.multivariate_normal.MultivariateNormal(self.mu.reshape(-1), sig_x)
        q = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), K_psi+self.jitter * torch.eye(K_psi.shape[0]))  
        KL = torch.distributions.kl_divergence(p, q).mean()
        return KL

    def draw_targets(self):
        """sample from variational distribution mu_psi, K_psi and store in targets"""
        eps_variational = torch.randn(size=(self.n_latent, 1), dtype=torch.float, device=self.device)
        targets = self.mu + torch.mul(torch.sqrt(self.K).view(self.n_latent, 1), eps_variational)
        self.targets = targets

    def mean_targets(self):
        """get mean of the variational distribution mu_psi an store in targets"""
        self.targets = self.mu.clone()

    def set_mode(self, mode: str):
        """mode out of "mean_predictions" or "sample_predictions" 
           mean predictions are not deterministic but set GP mean"""
        self.mode = mode

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """ see eq. 13 from Wilson et al., function-space update is performed via
        \sum_{j=1}^m v_j k(.,z_j)"""
        y = self.targets
        L = self.K_psi_cholesky
        k_x_star = self.full_kernel(x, self.psi)
        basis_evaluation = torch.matmul(self.prior_weights, self.compute_basis(self.psi).t())
        x_val = y - basis_evaluation.t()
        res, _ = torch.triangular_solve(x_val, L, upper=False)
        res_2, _ = torch.triangular_solve(k_x_star.t(), L, upper=False)
        update = torch.matmul(res_2.t(), res)
        return update

    def full_kernel(self, left_var: torch.Tensor, right_var: torch.Tensor) -> torch.Tensor:
        """ computes the kernel function of two inputs K with K_ij = k(x_i, x_j)
        Args:
            left_var ((batchsize, dim) - torch.Tensor): left variables.
            right_var ((batchsize, dim) - torch.Tensor): right variable.

        Returns:
            K (TYPE): DESCRIPTION.

        """
        # divide by lengthscale
        K = full_kernel(left_var, right_var, self.sigma_f, self.L_sqrt.view(1,self.dim))
        return K

    def sample_theta(self):
        """spectral density, corresponds to omega on page 4, sample from N(0,diag(1/L))"""
        Sigma_dist = torch.diag((1 / self.L).view(self.dim))
        theta_dist = torch.distributions.MultivariateNormal(torch.zeros(self.dim), Sigma_dist)
        theta = theta_dist.sample((self.n,))
        self.theta = theta

    def sample_basis_weights(self):
        """ initialize function draw by sampling tau, theta and weights"""
        self.sample_theta()
        self.set_prior_weights()

    def compute_basis(self, x: torch.Tensor) -> torch.Tensor:
        """ computes feature matrix with basis functions \sqrt(\sigma_f^2/S)(cos (x^t w), sin(x^t w))

        Args:
            x ((batchsize, dim) - torch.Tensor): inputs.

        Returns:
            (batchsize, 2 * self.n) - torch.Tensor

        """
        x_inputs = torch.matmul(self.theta, x.t()).t()
        phi_cos = torch.sqrt(torch.Tensor([self.sigma_f / self.n])) * torch.cos(x_inputs)
        phi_sin = torch.sqrt(torch.Tensor([self.sigma_f / self.n])) * torch.sin(x_inputs)
        return torch.cat((phi_cos, phi_sin), 1)


    def set_prior_weights(self):
        """ sample w in Eq. 33"""
        self.prior_weights = torch.normal(mean=0.0, std=1.0, size=((1, 2 * self.n)))

    def sample_prior(self, x) -> torch.Tensor:
        """ see first part in Eq. (33)
        Args:
            x ((batchsize, dim)-Tensor): inputs.

        Returns:
            output ((batchsize, self.n)- torch.Tensor): weight-space prior .

        """
        basis = self.compute_basis(x)
        output = torch.matmul(self.prior_weights, basis.t())
        return output

    def diff_prior(self, x) -> torch.Tensor: 
        """computes derivative of prior in Matheron's rule', especially important for determinant of symplectic solvers""" 
        x_inputs = torch.matmul(self.theta, x.t()).t()
        phi_cos = -torch.sqrt(torch.Tensor([self.sigma_f / self.n])) * torch.sin(x_inputs)
        phi_sin = torch.sqrt(torch.Tensor([self.sigma_f / self.n])) * torch.cos(x_inputs)
        theta_cat = torch.cat((self.theta, self.theta), 0)
        output = torch.matmul(self.prior_weights*theta_cat.t(), (torch.cat((phi_cos, phi_sin), 1)).t())
        return output 
    
    def diff_update(self, x) -> torch.Tensor:
        """computes update of Matheron's rule, especially important for determinant of symplectic solvers""" 
        K_star = self.diff_kernel(x)
        k_x = self.alpha_diff(K_star)
        y = self.targets
        L = self.K_psi_cholesky
        k_x_star = self.diff_kernel(x)
        basis_evaluation = torch.matmul(self.prior_weights, self.compute_basis(self.psi).t())
        x_val = y - basis_evaluation.t()
        res, _ = torch.triangular_solve(x_val, L, upper=False)
        res_2, _ = torch.triangular_solve(k_x_star, L, upper=False)
        update = torch.matmul(res_2.t(), res)       
        return update 
        
    def diff_draw(self, x) -> torch.Tensor:
        """computes derviative of GP drawn from Matheron's rule by adding prior derivative and update derivative""" 
        res = self.diff_prior(x)+self.diff_update(x)
        return res 

    def mu_x(self, alpha: torch.Tensor) -> torch.Tensor:
        """ evaluates mean function
            mu(x)=alpha(x)mu_psi
        """
        mu_out = torch.mm(alpha, self.targets)
        return mu_out

    def alpha(self, K_x: torch.Tensor) -> torch.Tensor:
        """ computing alpha, mu and sigma at input"""
        K_psi = self.K_psi
        L = self.K_psi_cholesky
        res, _ = torch.triangular_solve(K_x.t(), L, upper=False)
        res, _ = torch.triangular_solve(res, L.t(), upper=True)
        return res.t()

    def psi_kernel(self):
        """computes k(psi,psi) and cholesky decomposition"""
        K_psi = self.full_kernel(self.psi, self.psi)
        K_psi = K_psi + self.jitter * torch.eye(K_psi.shape[0])
        K_psi_cholesky = torch.cholesky(K_psi)
        self.K_psi = K_psi
        self.K_psi_cholesky = K_psi_cholesky

    def sample_function(self, x: torch.Tensor) -> torch.Tensor:
        """ sample function from Eq. 33

        Args:
            x ((batchsize, dim)-torch.Tensor): evaluation points.

        Returns:
            y ((batchsize, 1)-torch.Tensor): samples from GP posterior.

        """
        y = self.sample_prior(x) + self.update(x).t()
        return y

    def diff_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """ evaluation of k(dx, psi)
            http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf (Eq. (2))
        Args:
            x ((batchsize, dim)- torch.Tensor): inputs.

        Returns:
            K_star ((batchsize, self.n_latent)): k(dx, psi)

        """
        psi = self.psi
        K_psi = self.K_psi
        left_kernel = self.full_kernel(x, psi).t()
        diff_part = -torch.div(x - psi, self.L.view(1, self.dim))
        K_star = diff_part * left_kernel
        return K_star

    def alpha_diff(self, K_star: torch.Tensor) -> torch.Tensor:
        """ computes k(dx',psi)K(psi,psi)^(-1)

        Args:
            x ((batchsize, d) - torch.Tensor): input.

        Returns:
            TYPE: DESCRIPTION.

        """
        res, _ = torch.triangular_solve(K_star, self.K_psi_cholesky, upper=False)
        res, _ = torch.triangular_solve(res, self.K_psi_cholesky.t(), upper=True)
        return res.t()

    def mu_diff(self, x: torch.Tensor) -> torch.Tensor:
        """compute mean for derivative GP"""
        psi = self.psi
        mu_z = self.mu
        K_star = self.diff_kernel(x)
        k_x = self.alpha_diff(K_star)
        res = torch.matmul(k_x, mu_z)
        return res

    def save_local_model(self):
        """saves current weights and inducing targets from current configuration""" 
        self.local_theta.append(self.theta.clone())
        self.local_weights.append(self.prior_weights.clone())
        self.local_targets.append(self.targets.clone())
        
    def reset_local_model(self):
        """resets current weights and inducing targetrs from current configuration""" 
        self.local_theta = []
        self.local_weights = []
        self.local_targets = []
        
    def save_global_model(self):
        """save model configuration for multiple draws from function""" 
        self.parameter_list["psi"].append(self.psi.clone())
        self.parameter_list["mu"].append(self.mu.clone())
        self.parameter_list["log_K"].append(self.log_K.clone())
        self.parameter_list["K_psi"].append(self.K_psi.clone())
        self.parameter_list["K_psi_cholesky"].append(self.K_psi_cholesky.clone())
        self.parameter_list["log_L"].append(self.log_L.clone())
        self.parameter_list["log_sigma_f"].append(self.log_sigma_f.clone()) 
        self.parameter_list["theta"].append(self.local_theta)
        self.parameter_list["weights"].append(self.local_weights)
        self.parameter_list["targets"].append(self.local_targets)
        
    def load_model(self, model_idx, seed_idx):
        """loads model configuration for multiple draws from function""" 
        self.psi = torch.nn.Parameter(self.parameter_list["psi"][model_idx])
        self.mu = torch.nn.Parameter(self.parameter_list["mu"][model_idx])
        self.log_K = torch.nn.Parameter(self.parameter_list["log_K"][model_idx])
        self.K_psi = torch.nn.Parameter(self.parameter_list["K_psi"][model_idx])
        self.K_psi_cholesky = torch.nn.Parameter(self.parameter_list["K_psi_cholesky"][model_idx])
        self.log_L = torch.nn.Parameter(self.parameter_list["log_L"][model_idx])
        self.log_sigma_f = torch.nn.Parameter(self.parameter_list["log_sigma_f"][model_idx])
        self.theta = self.parameter_list["theta"][model_idx][seed_idx]
        self.prior_weights = self.parameter_list["weights"][model_idx][seed_idx]               
        self.targets = self.parameter_list["targets"][model_idx][seed_idx]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ evaluates GP sample of batch

        Args:
            x ((batchsize,dim)-torch.Tensor): input of sparse GP.

        Returns:
            torch.Tensor: sample of sparse GP.

        """
        if self.mode == "mean_predictions":
            K_x = self.full_kernel(x, self.psi)
            alpha = self.alpha(K_x)
            y = self.mu_x(alpha)
        else:
            y = self.sample_function(x)
        return y.view(x.shape[0], 1)


class ODE(torch.nn.Module):
    """constrained dynamics from two independent GPs

    Args:
        gp (torch.nn.Module): first dimension.
        gp1 (torch.nn.Module): second dimension

    Attributes:
        gp (torch.nn.Module): first dimension.
        gp1 (torch.nn.Module): second dimension

    """

    def __init__(self, gp:  Optional[torch.nn.ModuleList] = None, gp1:  Optional[torch.nn.ModuleList] = None):
        super().__init__()
        self.gp = gp if gp is not None else []
        self.gp1 = gp1 if gp1 is not None else []

    def draw_targets(self):
        """draw targets from mean and covariance for both GPs""" 
        self.gp.draw_targets()
        self.gp1.draw_targets()

    def set_mode(self, mode:str):
        if mode == "mean_predictions":
            self.gp.mode = "mean_predictions"
            self.gp1.mode = "mean_predictions"
        else:
            self.gp.mode = "sample_predictions"
            self.gp1.mode = "sample_predictions"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ evaluates sample of batch of approximated rigid body dynamics

        Args:
            x ((batchsize,dim)-torch.Tensor): input of sparse GP.

        Returns:
            torch.Tensor: sample of dynamics

        """
        n = x.shape[0]
        x0 = x[:, 1:3]
        x1 = torch.cat((x[:, 0], x[:, 2])).reshape(n, 2)
        y0 = self.gp(x0)
        y1 = self.gp1(x1)
        # nescessary constraint for quadratic invariant
        y2 = -(x[:, 0] * y0 + x[:, 1] * y1) / x[:, 2]
        output = torch.cat((y0, torch.cat((y1, y2), 1)), 1)
        return output

    def kl(self):
        """compute KL-divergence"""
        return self.gp.kl() + self.gp1.kl()

    def psi_kernel(self):
        """update kernel"""
        self.gp.psi_kernel()
        self.gp1.psi_kernel()

    def mean_targets(self):
        """set targets to variational mean for both GPs"""
        self.gp.mean_targets()
        self.gp1.mean_targets()

    def sample_basis_weights(self):
        """sample weights for both GPs"""
        self.gp.sample_basis_weights()
        self.gp1.sample_basis_weights()

    def save_local_model(self):
        """save model configuration for both GP models""" 
        self.gp.save_local_model()
        self.gp2.save_local_model()
        
    def reset_local_model(self):
        """reset model configuration for both GP models""" 
        self.gp.reset_loca_model()
        self.gp2.reset_loca_model()
        
    def save_global_model(self):
        """save GP models over multiple draws from GPs for both GPs""" 
        self.gp.save_global_model()
        self.gp2.save_global_model()
        
    def load_model(self, model_idx, seed_idx):
        """load saved models for both GPs""" 
        self.gp.load_model(model_idx, seed_idx)
        self.gp2.load_model(model_idx, seed_idx)  

    def full_kernel(left_var: torch.Tensor, right_var: torch.Tensor, sigma_f: torch.Tensor, L_sqrt: torch.Tensor) -> torch.Tensor:
        """ computes the kernel function of two inputs K with K_ij = k(x_i, x_j)
        Args:
            left_var ((batchsize, dim) - torch.Tensor): left variables.
            right_var ((batchsize, dim) - torch.Tensor): right variable.
    
        Returns:
            K (TYPE): DESCRIPTION.
    
        """
        # divide by lengthscale
        X = left_var / L_sqrt
        X2 = right_var / L_sqrt 
        Xs = (X ** 2).sum(dim=1)
        X2s = (X2 ** 2).sum(dim=1)
        dist = -2 * X.mm(X2.t()) + Xs.view(-1, 1) + X2s.view(1, -1)
        K = sigma_f*torch.exp(-0.5 * dist)
        return K


class MatheronDerivative(MatheronGP):
    """Implementation of sampling from derivative of weight space representation of GP

    Notes:
        Equations are based on [1]
        [1]: James T. Wilson, Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, Marc Peter Deisenroth:
        "Efficiently Sampling Functions from Gaussian Process Posteriors", International Conference on Machine Learning, 2020

    Args:
        dt (int): step size (defaults to 0.1)
    Attributes:
        dim (int): dimension (special case 2)
        n_latent (int): number of inducing inputs
        dt : step size 
        mu ((n_latent,1)-torch.Tensor, optional): variational mean
        log_K ((n_latent)-torch.Tensor, optional): logarithm of the variational covariance matrix
        log_L ((1,dim)-torch.Tensor, optional): log of the lengthscale
        log_sigma_f (torch.Tensor, optional): log of the GP variance
        psi ((n_latent, dim)-torch.Tensor): inducing inputs
    """

    def __init__(self, dt: float = 0.1):
        super().__init__()
        n = 4
        self.dim = 2
        self.n_latent = n ** 2
        self.dt = dt
        self.mu = torch.nn.Parameter(
            torch.zeros(self.n_latent, 1, dtype=torch.float) + 0.05 * torch.randn(self.n_latent, 1, dtype=torch.float),
            requires_grad=True,
        )
        self.log_K = torch.nn.Parameter(
            torch.log(1e-7 * torch.ones(self.n_latent, dtype=torch.float)), requires_grad=True
        )
        self.log_L = torch.nn.Parameter(
            torch.log(torch.tensor([2.0], dtype=torch.float)) * torch.ones(2), requires_grad=True
        )
        self.log_sigma_f = torch.nn.Parameter(torch.log(torch.tensor([0.0001], dtype=torch.float)), requires_grad=True)
        psi_x = torch.from_numpy(np.linspace(-0.5, 0.5, n)).float()
        psi_y = torch.from_numpy(np.linspace(-0.5, 0.5, n)).float()
        psi_x, psi_y = torch.meshgrid(psi_x, psi_y)
        psi_x = torch.reshape(psi_x, [(n) ** 2, 1])
        psi_y = torch.reshape(psi_y, [(n) ** 2, 1])
        self.psi = torch.nn.Parameter(torch.cat([psi_x, psi_y], dim=1), requires_grad=True)

    def first_der(self, x: torch.Tensor) -> torch.Tensor:
        """k'(x,psi)=[[k(du_1, psi)],...,[k(dv_n,psi)]]
        """
        left_kernel = self.full_kernel(x, self.psi)
        diff_x = -torch.div(x[:, 0].view(-1, 1) - self.psi[:, 0].view(1, -1), torch.exp(self.L[0]))
        diff_y = -torch.div(x[:, 1].view(-1, 1) - self.psi[:, 1].view(1, -1), torch.exp(self.L[1]))
        return diff_x * left_kernel, diff_y * left_kernel

    def compute_diff_basis(self, x: torch.Tensor):
        """computes derivative of Matheron GP prior"""
        x_inputs = torch.matmul(self.theta, x.t()).t()
        phi_cos_der = -torch.sqrt(torch.Tensor([self.sigma_f/ self.n])) * torch.sin(x_inputs)
        phi_sin_der = torch.sqrt(torch.Tensor([self.sigma_f / self.n])) * torch.cos(x_inputs)
        return torch.cat((phi_cos_der, phi_sin_der), 1)

    def sample_prior(self, x: torch.Tensor):
        """ sample function from Eq. 33 with phi_i = \sqrt(\sigma_f^2/S)\omega_i (-sin(x^T\omega), cos(x^T \omega))

        Args:
            x ((batchsize, dim)-torch.Tensor): evaluation points.

        Returns:
            y ((batchsize, 1)-torch.Tensor): samples from GP posterior.

        """
        theta = torch.cat((self.theta, self.theta))
        basis = self.compute_diff_basis(x)
        output_p = torch.matmul(self.prior_weights * theta[:, 0], basis.t())
        output_q = torch.matmul(self.prior_weights * theta[:, 1], basis.t())
        y = torch.cat((output_p.t(), output_q.t()), 1)
        return y

    def update(self, x: torch.Tensor):
        """computes derivative of update""" 
        y = self.targets
        L = self.K_psi_cholesky
        k_diff_p, k_diff_q = self.first_der(x)
        basis_evaluation = torch.matmul(self.prior_weights, self.compute_basis(self.psi).t())
        x_val = y - basis_evaluation.t()
        res, _ = torch.triangular_solve(x_val, L, upper=False)
        res_p, _ = torch.triangular_solve(k_diff_p.t(), L, upper=False)
        res_q, _ = torch.triangular_solve(k_diff_q.t(), L, upper=False)
        update_p = torch.matmul(res_p.t(), res)
        update_q = torch.matmul(res_q.t(), res)
        return torch.cat((update_p, update_q), 1).t()


    def forward(self, x: torch.Tensor):
        """ evaluates GP sample of batch

        Args:
            x ((batchsize,dim)-torch.Tensor): input of sparse GP.

        Returns:
            torch.Tensor: sample of sparse GP.

        """
        if self.mode == "mean_predictions":
            k_p, k_q = self.first_der(x)
            alpha_p = self.alpha(k_p)
            alpha_q = self.alpha(k_q)
            mu_p = self.mu_x(alpha_p)
            mu_q = self.mu_x(alpha_q)
            y = torch.cat((mu_p, mu_q), 1)
        else:
            y = self.sample_function(x)
        return y.view(x.shape[0], 2)


class StackedGP(torch.nn.Module):

    """stacking independent MatheronGP modules

    Args:
        d (int): dimension.
        GP_List (torch.nn.ModuleList): list of Matheronmodules
        device (torch.device): device

    Attributes:
        d (int): dimension.
        GP_List (torch.nn.ModuleList): list of sparse GP modules
        device (torch.device): device
        d_new (int): (d/2): cut in the middle of dim (for Hamiltonian problems)
    """

    def __init__(self, GP_list: Optional[torch.nn.ModuleList] = None, device= "cpu"):
        super().__init__()
        self.d = len(GP_list)
        self.GP_list = GP_list if GP_list is not None else []
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_new = int(self.d / 2)

    def kl(self):
        """sum kl-divergence contrubitions"""
        kl = 0
        for GP in self.GP_list:
            kl = kl + GP.kl()
        return kl

    def psi_kernel(self):
        """construct kernel at sparse inputs for each GP"""
        for GP in self.GP_list:
            GP.psi_kernel()

    def det(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """compute flow determinant for explicit Euler scheme det(\psi')
           
        """
        det = 0
        identity = torch.eye(x.shape[1])
        dynamics = torch.zeros(x.shape[1], x.shape[1])
        for i in range(self.d):
            sample = self.GP_list[i].diff_draw(x)
            dynamics[i,:] = sample.t().view(self.d)
        return torch.det(identity+dt*dynamics)

    def mean_targets(self):
        """set targets to variational mean for both GPs"""
        for gp in self.GP_list:
            gp.mean_targets()

    def draw_targets(self):
        """draw targets from variational distribution for each GP in list"""
        for gp in self.GP_list:
            gp.draw_targets()

    def sample_basis_weights(self):
        """sample weights for both GPs"""
        for gp in self.GP_list:
            gp.sample_basis_weights()

    def set_mode(self, mode:str):
        """sets mode out of "mean_predictions","sample_predictions" for each GP in list""" 
        for gp in self.GP_list:
            gp.set_mode(mode)

    def save_local_model(self):
        """saves current model configuration for each GP in list""" 
        for gp in self.GP_list:
            gp.save_local_model()
        
    def reset_local_model(self):
        """resets model configuration for each GP in list""" 
        for gp in self.GP_list:
            gp.reset_local_model()
        
    def save_global_model(self):
        """saves model configuration over multiple draws of the model for each GP in list"""
        for gp in self.GP_list:
            gp.save_global_model()
        
    def load_model(self, model_idx, seed_idx):
        """loads model configuration from list and index of draw for each GP in list""" 
        for gp in self.GP_list:
            gp.load_model(model_idx, seed_idx) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ return sample of GP dynamics

        Args:
            x ((batchsize,d)-torch.Tensor): input.

        Returns:
            output (torch.Tensor): sample.

        """
        """sample output"""
        output = torch.zeros(x.shape[0], self.d, device=self.device)
        i = 0
        for GP in self.GP_list:
            y = GP(x)
            output[:, i] = y[:, 0]
            i += 1
        return output
 
