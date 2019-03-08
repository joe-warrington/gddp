import numpy as np
import time
import gurobipy as gu
import ecos
import scipy.sparse as sp
import scipy.linalg
import copy
import pandas as pd
import itertools
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
# try:
#     from scipy.interpolate import RegularGridInterpolator
# except ImportError as e:
#     print "Couldn't import scipy.interpolate.RegularGridInterpolator!"
#     print e
from scipy.io import savemat, loadmat
from scipy.optimize import nnls, minimize_scalar
from scipy.stats import laplace
from pprint import pprint
import os
from gddp import Constraint
np.seterr(divide='ignore')

class VFApproximator(object):
    """VFApproximator generates approximate value functions for continuous state and action
    problems.
    """

    def __init__(self, system, solver='gurobi'):
        """Initialize VFApproximator object.

        :param system: System object on which the value function approximator will act
        :param solver: Solver to use ('gurobi', 'ecos', or None)
        """

        # Get properties from system
        self.m = system.m
        self.n = system.n
        self.n_beta = system.n_beta
        self.s = system

        # Create placeholders for optimization model
        self.mod = None
        self.solver = solver
        self.ecos_c = None
        self.ecos_A = None
        self.ecos_b = None
        self.ecos_G = None
        self.ecos_h = None
        self.ecos_si_constrs = None
        self.ecos_beta_constrs = None
        self.ecos_alpha_constrs = None
        self.ecos_dims = None

        # Performance logging
        self.total_solver_time = None
        self.one_stage_problems_solved = 0
        self.b_gap_eval_time = None
        self.v_integral_eval_time = None
        self.lb_computation_time = None

        # Upper-bounding object (deprecated)
        # self.bd = Bounder(self.n)

        # Empty list of value function lower-approximation constraints
        self.alpha_c_list = []
        self.alpha_c_in_model = []

        # Log of x coordinates visited
        self.x_visited = None
        # Number of iterations carried out
        self.final_j = None

        # Measurement of integral
        self.v_integral = None
        self.v_n_ind_samples = None
        np.random.seed(1000)
        self.v_integral_x_list = None

    def create_audit_samples(self, n_s, seed=None):
        """Create set of independent samples on which value function approximation quality will be
        tested.

        :param n_s: Number of samples
        :param seed: Random number generator seed
        :return: nothing
        """
        self.v_n_ind_samples = n_s
        if seed is not None:
            np.random.seed(10000 + seed)
        self.v_integral_x_list = [self.s.state_mean + np.dot(np.sqrt(self.s.state_var),
                                                             np.random.randn(self.n))
                                  for sample_no in range(self.v_n_ind_samples)]

    def create_vfa_model(self):
        """Create an initialized value function approximation model, setting up optimization
        variables and solver settings. Also create zero lower bounding function on optimal value
        function.

        :return: nothing
        """
        if self.solver == 'gurobi':
            self.mod = gu.Model("VFA")
            self.mod.params.outputflag = 0
            self.mod.params.qcpdual = 0
            self.mod.params.dualreductions = 1
            self.mod.params.optimalitytol = 1e-6
            self.mod.params.feasibilitytol = 1e-6
            self.mod.params.barconvtol = 1e-6
            self.mod.params.barqcpconvtol = 1e-6
            self.mod.params.numericfocus = 0
            x = []
            for i in range(self.m):
                x.append(self.mod.addVar(name="u_%d" % (i + 1), lb=-1e100, obj=0.))
            for i in range(self.n):
                x.append(self.mod.addVar(name="xplus_%d" % (i + 1), lb=-1e100, obj=0.))
            for i in range(self.n_beta):
                x.append(self.mod.addVar(name="beta_%d" % (i + 1), lb=-1e100, obj=1.))
            x.append(self.mod.addVar(name="alpha", lb=-1e100, obj=self.s.gamma))
            self.mod.update()

            # Placeholder for state x when building constraints (use the origin)
            # Constraints will be updated for particular xhat when optimizer is called
            xhat = np.zeros((self.n,), dtype=float)

            # Add dynamics constraint
            fxx = self.s.f_x_x(xhat)
            fux = self.s.f_u_x(xhat)
            for i in range(self.n):
                self.mod.addConstr(x[self.m + i] - gu.LinExpr([(fux[i, p], x[p])
                                                               for p in range(self.m)])
                                   == fxx[i], name="dyn_%d" % (i+1))

            # Add state-input constraints
            for i in range(self.s.D.shape[0]):
                self.mod.addConstr(gu.LinExpr([(self.s.E[i, p], x[p]) for p in range(self.m)]) <=
                                   self.s.h_x(xhat)[i],
                                   name="sic_%d" % (i+1))

            # Add beta epigraph constraints
            for i, li in enumerate(self.s.li):
                qe = gu.QuadExpr()
                for p in range(self.m):
                    qe += 0.5 * x[p] * gu.LinExpr([(self.s.Ri[i][p, q], x[q])
                                                   for q in range(self.m)])
                self.mod.addConstr(gu.LinExpr([(li[p], x[self.m + self.n + p])
                                               for p in range(self.n_beta)]) >=
                                   self.s.phi_i_x(i, xhat) + qe +
                                   gu.LinExpr([(self.s.ri[i][p], x[p]) for p in range(self.m)]) +
                                   self.s.ti[i], name="beta_%d" % (i+1))

            # Add initial alpha epigraph constraint
            initial_lb = Constraint(self.n, 0, 0., None, None)
            self.add_alpha_constraint(initial_lb)

            # Update model to reflect constraints
            self.mod.update()

        elif self.solver == 'ecos':
            self.ecos_c = np.hstack((np.zeros((self.m + self.n,), dtype=float),
                                     np.ones((self.n_beta,), dtype=float),
                                     np.array([self.s.gamma])))

            # Placeholder for state x when building constraints (use the origin)
            # Constraints will be updated for particular xhat when optimizer is called
            xhat = np.zeros((self.n,), dtype=float)

            # Add dynamics constraints
            fxx = self.s.f_x_x(xhat)
            fux = self.s.f_u_x(xhat)
            self.ecos_A = sp.csc_matrix(np.hstack((-fux,
                                           np.eye(self.n),
                                           np.zeros((self.n, self.n_beta + 1), dtype=float))))
            self.ecos_b = fxx

            # Add state-input constraints
            G_sic = sp.csc_matrix(np.hstack((-self.s.E,
                                             np.zeros((self.s.E.shape[0],
                                                       self.n + self.n_beta + 1)))))
            h_sic = np.dot(self.s.D, xhat) - np.squeeze(self.s.h)
            self.ecos_si_constrs = (G_sic, h_sic)

            # Add beta epigraph constraints
            self.ecos_beta_constrs = []
            for i, li in enumerate(self.s.li):
                quad = scipy.linalg.block_diag(self.s.Ri[i],
                                               np.zeros((self.n + self.n_beta + 1,
                                                         self.n + self.n_beta + 1)))
                lin = np.hstack((self.s.ri[i], np.zeros(self.n), -li, np.zeros(1,)))
                const = self.s.phi_i_x(i, xhat) + self.s.ti[i]
                G_out, h_out, type_out = self.build_ecos_constraint(const, lin, quad)
                self.ecos_beta_constrs.append((G_out, h_out, type_out))

            # Add initial alpha epigraph constraints
            self.ecos_alpha_constrs = []
            initial_lb = Constraint(self.n, 0, 0., None, None)
            self.add_alpha_constraint(initial_lb)

        else:
            if self.s.brute_force_solve:
                initial_lb = Constraint(self.n, 0, 0., None, None)
                self.add_alpha_constraint(initial_lb)
            else:
                print "Solver " + self.solver + " not implemented!"
                raise SystemExit()

    def update_constraints_for_xhat(self, xhat, xplus_constraint):
        """Update constraint of optimization model to account for the current state, which enters
        into the one-stage optimization as a fixed parameter.

        :param xhat: Fixed state parameter for the problem
        :param xplus_constraint: Add a constraint that the successor state should take a particular
        value
        :return: nothing
        """

        fxx = self.s.f_x_x(xhat)
        fux = self.s.f_u_x(xhat)
        if self.solver == 'gurobi':
            x = [v for v in self.mod.getVars()]
            # Update dynamics
            for i in range(self.n):
                c = self.mod.getConstrByName('dyn_%d' % (i + 1))
                self.mod.remove(c)
                self.mod.addConstr(x[self.m + i] - gu.LinExpr([(fux[i, p], x[p])
                                                               for p in range(self.m)])
                                   == fxx[i], name="dyn_%d" % (i + 1))
            # Update state-input constraints
            for i in range(self.s.D.shape[0]):
                c = self.mod.getConstrByName('sic_%d' % (i + 1))
                self.mod.remove(c)
                self.mod.addConstr(gu.LinExpr([(self.s.E[i, p], x[p]) for p in range(self.m)]) <=
                                   self.s.h[i] - np.dot(self.s.D[i, :], xhat),
                                   name="sic_%d" % (i + 1))
            # Add beta epigraph constraints
            for i, li in enumerate(self.s.li):
                c = [q for q in self.mod.getQConstrs() if q.QCName == "beta_%d" % (i + 1)]
                self.mod.remove(c[0])
                qe = gu.QuadExpr()
                for p in range(self.m):
                    qe += 0.5 * x[p] * gu.LinExpr([(self.s.Ri[i][p, q], x[q])
                                                   for q in range(self.m)])
                self.mod.addConstr(gu.LinExpr([(li[p], x[self.m + self.n + p])
                                               for p in range(self.n_beta)]) >=
                                   self.s.phi_i_x(i, xhat) + qe +
                                   gu.LinExpr([(self.s.ri[i][p], x[p]) for p in range(self.m)]) +
                                   self.s.ti[i], name="beta_%d" % (i + 1))

            if xplus_constraint is not None:
                # Remove any existing x_plus constraints
                c = [l for l in self.mod.getConstrs() if "forced_xplus_" in l.ConstrName]
                for l in c:
                    self.mod.remove(l)
                assert xplus_constraint.shape == (self.n,), "Forced x_plus has wrong dimension"

                # n_subspace = xplus_constraint.shape[1]
                # print "n_subspace:", n_subspace
                for j in range(self.n):
                    # b_power_A = np.dot(xplus_constraint[:, j],
                    #                    -np.linalg.matrix_power(self.s.A_mat, self.n - n_subspace))
                    # print "    ", j, b_power_A
                    # self.mod.addConstr(gu.LinExpr([(b_power_A[i], x[self.m + i])
                    #                                for i in range(self.n)]) <= 1e-2,
                    #                    name='forced_xplus_u_%d' % (j + 1))
                    # self.mod.addConstr(gu.LinExpr([(b_power_A[i], x[self.m + i])
                    #                                for i in range(self.n)]) >= -1e-2,
                    #                    name='forced_xplus_l_%d' % (j + 1))
                    self.mod.addConstr(x[self.m + j] == xplus_constraint[j],
                                       name='forced_xplus_l_%d' % (j + 1))
            else:
                c = [l for l in self.mod.getConstrs() if "forced_xplus_" in l.ConstrName]
                for l in c:
                    self.mod.remove(l)
        elif self.solver == 'ecos':
            # Update dynamics
            self.ecos_A = sp.csc_matrix(np.hstack((-fux,
                                                   np.eye(self.n),
                                                   np.zeros((self.n, self.n_beta + 1),
                                                            dtype=float))))
            self.ecos_b = fxx
            # Update state-input constraints
            self.ecos_si_constrs = (self.ecos_si_constrs[0],
                                    np.dot(self.s.D, xhat) - np.squeeze(self.s.h))
            # Update beta epigraph constraints
            new_beta_constrs = []
            for i, tup in enumerate(self.ecos_beta_constrs):
                c = self.s.phi_i_x(i, xhat) + self.s.ti[i]
                if tup[2] == 'soc':
                    new_beta_constrs.append((tup[0], np.array([0.5 * (1 - c)] +
                                                              [0.] * (tup[1].shape[0] - 2) +
                                                              [0.5 * (1 + c)]), 'soc'))
                elif tup[2] == 'lin':  # Linear constraint
                    new_beta_constrs.append((tup[0], np.array([-c]), 'lin'))
                else:
                    print "Unrecognized constraint type: " + tup[2]
                    raise SystemExit()
            self.ecos_beta_constrs = new_beta_constrs
        else:
            if self.s.brute_force_solve:
                pass
            else:
                print "Solver " + self.solver + " not implemented!"
                raise SystemExit()

    @staticmethod
    def build_ecos_constraint(c, a, Q):
        """Reformulate constraint c + a'x + 0.5 x'Qx <= 0
        into the form Gx <=_K h for Q >= 0 to be compatible with ECOS format
        """
        assert a.shape[0] == Q.shape[0] == Q.shape[1]
        d = a.shape[0]
        rank_Q = np.linalg.matrix_rank(Q)
        if rank_Q > 0:
            # Constraint requires SOC representation
            w, V = scipy.linalg.eigh(Q)
            w_red, V_red = w[-rank_Q:], V[:, -rank_Q:]
            sqrt_Q = V_red * np.diag(np.sqrt(w_red))  # "Tall-thin" matrix square-root
            assert np.linalg.norm(np.dot(sqrt_Q, sqrt_Q.T) - Q) <= 1e-8
            h = np.array([0.5 * (1 - c)] +
                         [0.] * rank_Q +
                         [0.5 * (1 + c)])
            G = sp.csc_matrix(np.vstack((0.5 * np.reshape(a, (1, d)),
                                         -1. / np.sqrt(2) * sqrt_Q.T,
                                         -0.5 * np.reshape(a, (1, d)))))
            assert h.shape[0] == G.shape[0]
            return G, h, 'soc'
        else:
            # Constraint is linear
            h = np.array([-c])
            G = sp.csc_matrix(np.reshape(a, (1, d)))
            assert h.shape[0] == G.shape[0]
            return G, h, 'lin'

    def solve_for_xhat(self, xhat, print_sol=False, extract_constr=True, print_errors=False,
                       iter_no=None, alt_bound=False, xplus_constraint=None):
        """ Solve one-stage optimization problem, optionally returning a lower-bounding function
        on the optimal value function.

        :param xhat: Fixed state parameter to solve the one stage problem for
        :param print_sol: Boolean, print solution details
        :param extract_constr: Boolean, extract constraint (returns different values if so)
        :param print_errors: Boolean, print error information
        :param iter_no: Integer, iteration number of the GDDP algorithm
        :param alt_bound: Boolean, return alternative lower-bounding function using other dual
        :param xplus_constraint: e-element array or None, Force successor state to this value
        :return: If extract_constr: new lower bound, (u*, x+*, beta*, alpha*)
                 Otherwise: u*, optimal cost, stage cost
        """

        self.update_constraints_for_xhat(xhat, xplus_constraint)
        check_lambda_a = False

        if self.s.name == 'Inverted pendulum':
            xhat[0] = divmod(xhat[0] + np.pi, 2 * np.pi)[1] - np.pi

        if self.solver == 'gurobi':
            ts1 = time.time()

            if extract_constr:  # Change to cautious settings for extracting accurate multipliers
                self.mod.params.qcpdual = 1
                self.mod.params.dualreductions = 0
                self.mod.params.optimalitytol = 1e-8
                self.mod.params.feasibilitytol = 1e-8
                self.mod.params.barconvtol = 1e-8
                self.mod.params.barqcpconvtol = 1e-8
                self.mod.params.numericfocus = 3

            self.mod.optimize()
            self.total_solver_time += time.time() - ts1
            self.one_stage_problems_solved += 1

            if extract_constr:  # Change settings back to less cautious ones
                self.mod.params.qcpdual = 0
                self.mod.params.dualreductions = 1
                self.mod.params.optimalitytol = 1e-6
                self.mod.params.feasibilitytol = 1e-6
                self.mod.params.barconvtol = 1e-6
                self.mod.params.barqcpconvtol = 1e-6
                self.mod.params.numericfocus = 0

            if self.mod.status in [2, 13]:
                if self.mod.status == 13:
                    print "  Warning: Optimization status 13 (suboptimal) encountered!"
                opt_cost = self.mod.getObjective().getValue()
                sol_vec = [v.x for v in self.mod.getVars()]

                # Can just return the optimal input and cost if we are only interested in that
                if extract_constr is False:
                    xplus = np.array(sol_vec[self.m:self.m + self.n])
                    stage_cost = opt_cost - self.s.gamma * self.eval_vfa(xplus)
                    return sol_vec[:self.m], opt_cost, stage_cost

                dodgy_bound = False
                pi = np.zeros(self.n, dtype=float)
                for i in range(self.n):
                    c_name = 'dyn_%d' % (i + 1)
                    try:
                        pi[i] = self.mod.getConstrByName(c_name).Pi
                    except Exception as e:
                        dodgy_bound = True
                        if print_errors:
                            print "Can't extract multiplier for constraint " + c_name + "! Using 0."
                            print e
                pi = np.array(pi)

                lambda_c = np.zeros(self.s.D.shape[0])
                for i in range(self.s.D.shape[0]):
                    c_name = 'sic_%d' % (i + 1)
                    try:
                        lambda_c[i] = self.mod.getConstrByName(c_name).Pi
                    except Exception as e:
                        dodgy_bound = True
                        if print_errors:
                            print "Can't extract multiplier for constraint " + c_name + "! Using 0."
                            print e

                lambda_b = np.zeros(self.s.n_beta_c, dtype=float)
                for i in range(self.s.n_beta_c):
                    c_name = 'beta_%d' % (i + 1)
                    try:
                        if self.mod.getConstrByName(c_name) is not None:
                            lambda_b[i] = self.mod.getConstrByName(c_name).Pi
                        elif len([d for d in self.mod.getQConstrs() if d.QCName == c_name]) == 1:
                            lambda_b[i] = [c for c in self.mod.getQConstrs()
                                           if c.QCName == c_name][0].QCPi
                    except Exception as e:
                        dodgy_bound = True
                        if print_errors:
                            print "Can't extract multiplier for constraint " + c_name + "! Using 0."
                            print e

                # lambda_a is only needed for audit purposes and can therefore be commented out
                lambda_a = []
                if check_lambda_a:
                    for i in range(len(self.alpha_c_list)):
                        if self.mod.getConstrByName('alpha_%d' % i) is not None:
                            lambda_a.append(self.mod.getConstrByName('alpha_%d' % i).Pi)
                        elif len([c for c in self.mod.getQConstrs()
                                  if c.QCName == 'alpha_%d' % i]) == 1:
                            try:
                                lambda_a.append([c for c in self.mod.getQConstrs()
                                                 if c.QCName == 'alpha_%d' % i][0].QCPi)
                            except Exception as e:
                                lambda_a.append(0.)
                                # dodgy_bound = True
                                # if print_errors:
                                print "Can't extract multiplier for constraint " + 'alpha_%d' % i + "!"
                                print e
                    lambda_a = np.array(lambda_a)

            else:
                print "Optimization problem did not solve: status %d" % self.mod.status
                raise SystemExit()
        elif self.solver == 'ecos':
            self.ecos_dims = {'l': 0, 'q': [], 'e': 0}
            G_quad, h_quad = None, None
            # Add state-input constraints
            G_lin = self.ecos_si_constrs[0]
            h_lin = self.ecos_si_constrs[1]
            self.ecos_dims['l'] += self.ecos_si_constrs[0].shape[0]
            for i, tup in enumerate(self.ecos_beta_constrs + self.ecos_alpha_constrs):
                if tup[2] == 'soc':
                    if G_quad is None:
                        G_quad = tup[0]
                        h_quad = tup[1]
                    else:
                        G_quad = sp.vstack((G_quad, tup[0]), format='csc')
                        h_quad = np.hstack((h_quad, tup[1]))
                    self.ecos_dims['q'].append(tup[0].shape[0])
                elif tup[2] == 'lin':
                    assert tup[0].shape[0] == 1
                    G_lin = sp.vstack((G_lin, tup[0]), format='csc')
                    h_lin = np.hstack((h_lin, tup[1]))
                    self.ecos_dims['l'] += tup[0].shape[0]
                else:
                    print "Unrecognized beta constraint type: " + tup[2]
                    raise SystemExit()

            if G_quad is not None:
                G = sp.vstack((G_lin, G_quad), format='csc')
                h = np.hstack((h_lin, h_quad))
            else:
                G = G_lin
                h = h_lin

            # print "c", self.ecos_c
            # print "A, b", self.ecos_A.todense(), self.ecos_b
            # print "G", G.todense(), isinstance(G, sp.csc_matrix)
            # print "h", h

            ts1 = time.time()

            sol = ecos.solve(self.ecos_c, G, h, self.ecos_dims,
                             self.ecos_A, self.ecos_b, verbose=False)
            self.one_stage_problems_solved += 1
            if sol['info']['exitFlag'] not in [0, 10]:
                print "ECOS did not solve to optimality: exit flag %d" % sol['info']['exitFlag']
                pprint(sol['info'])
                raise SystemExit()

            self.total_solver_time += time.time() - ts1

            sol_vec = sol['x']
            opt_cost = sol['info']['pcost']

            # Can just return the optimal input if we are only interested in that
            if extract_constr is False:
                xplus = sol_vec[self.m:self.m+self.n]
                stage_cost = opt_cost - self.s.gamma * self.eval_vfa(xplus)
                return sol_vec[:self.m], opt_cost, stage_cost

            # pprint(sol)
            dodgy_bound = False
            # print sol['info']

            u_opt = sol_vec[:self.m]
            x_plus_opt = sol_vec[self.m:self.m+self.n]
            beta_opt = sol_vec[self.m+self.n:-1]
            alpha_opt = sol_vec[-1]
            pi = sol['y']
            lambda_c = sol['z'][:self.ecos_si_constrs[0].shape[0]]

            # Solve KKT system for lambda_b
            r_mat = np.hstack((r.T + np.dot(self.s.Ri[i], sol_vec[:self.m]).T
                               for i, r in enumerate(self.s.ri)))
            l_mat = np.hstack((l.reshape((self.s.n_beta, 1)) for l in self.s.li))
            beta_mat = np.vstack((r_mat, l_mat))
            beta_vec = np.hstack((np.dot(self.s.f_u_x(xhat).T, pi) + np.dot(self.s.E.T, lambda_c),
                                  np.ones(self.n_beta,)))
            for i, r in enumerate(self.s.ri):
                slack = -(self.s.phi_i_x(i, xhat) + np.dot(r, u_opt) +
                          0.5 * np.dot(u_opt, np.dot(self.s.Ri[i], u_opt)) + self.s.ti[i] -
                          np.dot(self.s.li[i], beta_opt))
                if slack >= 1e-6:
                    slack_mat = np.zeros((1, self.s.n_beta_c), dtype=float)
                    slack_mat[0, i] = 1.
                    beta_mat = np.vstack((beta_mat,
                                          slack_mat))
                    beta_vec = np.hstack((beta_vec, np.zeros((1,))))
            lambda_b, _ = nnls(beta_mat, beta_vec)

            # Solve KKT system for lambda_a
            lambda_a = []
            if check_lambda_a:
                j_so_far = len(self.alpha_c_in_model)
                q_mat = np.zeros((self.n, j_so_far), dtype=float)
                for j, c in enumerate(self.alpha_c_in_model):
                    q_mat[:, j] = c.lin.T
                    if not c.no_hessian:
                        q_mat[:, j] += np.dot(c.hessian, x_plus_opt).T
                alpha_mat = np.vstack((q_mat, np.ones((1, j_so_far))))
                alpha_vec = np.hstack((-pi, np.array([self.s.gamma])))
                for j, c in enumerate(self.alpha_c_in_model):
                    slack = alpha_opt - c.eval_at(x_plus_opt)
                    if slack >= 1e-2:
                        alpha_mat = np.vstack((alpha_mat,
                                               np.hstack((np.zeros((1, j)),
                                                          np.ones((1, 1)),
                                                          np.zeros((1, j_so_far - j - 1))))))
                        alpha_vec = np.hstack((alpha_vec, np.zeros((1,))))
                lambda_a, _ = nnls(alpha_mat, alpha_vec)

            pi = -pi  # ECOS uses opposite sign convention for == constraint multiplier to Gurobi
        else:
            if self.s.brute_force_solve:
                if extract_constr:
                    dodgy_bound = False
                    sol_vec, opt_cost, delta_1, delta_alt, duals, kkt_res = \
                        self.brute_force_solve_for_xhat(xhat, True, print_errors,
                                                        iter_no=iter_no,
                                                        alt_bound=alt_bound)
                    lambda_a, lambda_b, lambda_c, pi = duals
                    xplus = sol_vec[self.m:self.n + self.m]
                    self.one_stage_problems_solved += 1
                else:
                    sol_vec, opt_cost = self.brute_force_solve_for_xhat(xhat, False, print_errors,
                                                                        iter_no=iter_no,
                                                                        alt_bound=alt_bound)
                    self.one_stage_problems_solved += 1
                    xplus = sol_vec[self.m:self.n+self.m]
                    stage_cost = opt_cost - self.s.gamma * self.eval_vfa(xplus)
                    return sol_vec[:self.m], opt_cost, stage_cost
            else:
                print "Solver " + self.solver + " not implemented!"
                raise SystemExit()

        if print_sol:
            print "Optimal cost:", opt_cost
            print "u:", [round(p, 2) for p in sol_vec[:self.m]], \
                "x+:", [round(p, 2) for p in sol_vec[self.m:self.m + self.n]], \
                "beta:", [round(p, 2)
                          for p in sol_vec[self.m+self.n:self.m+self.n+self.n_beta]], \
                "alpha:", round(sol_vec[-1], 2)
            print "pi:", [round(p, 2) for p in pi], \
                "lambda_c:", [round(p, 5) for p in lambda_c], \
                "lambda_beta:", [round(p, 5) for p in lambda_b], \
                "lambda_alpha:", [round(p, 5) for p in lambda_a]

        if not dodgy_bound:
            assert np.linalg.norm(np.dot(np.hstack((l.reshape((-1, 1)) for l in self.s.li)),
                                         lambda_b)
                                  - np.ones((self.s.n_beta,))) <= 1e-2, \
                "Incorrect lambda_b sum: %.5f\n" % np.sum(lambda_b)
            assert np.all([b >= -1e-4 for b in lambda_b]), \
                "Not all lambda_b multipliers positive! " + repr(lambda_b)

        if check_lambda_a:
            if np.abs(np.sum(lambda_a) - self.s.gamma) >= 1e-3:
                print "  Warning: Incorrect lambda_a sum: %.5f. To fix." % np.sum(lambda_a)
            # if np.abs(np.sum(lambda_a) - self.s.gamma) >= 0.01:
            #     dodgy_bound = True  # Commenting this out because duals fixed below
            assert np.all([a > -1e-4 for a in lambda_a]), \
                "Not all lambda_a multipliers positive! " + repr(lambda_a)

        if self.s.brute_force_solve:
            # Entering this section of code implies extract_constr is False
            new_bound = Constraint(n=self.n, id_in=len(self.alpha_c_list), const_in=0.,
                                   lin_in=None, hessian_in=None, sys_in=self.s,
                                   duals_in=duals, constrs_in=self.alpha_c_in_model,
                                   xhat_in=xhat, xplus_opt_in=xplus)
            existing_vfa = self.eval_vfa(xhat)
            val_without_delta_1 = new_bound.eval_at(xhat, exclude_zeta_1=True)
            _ = new_bound.zeta_1(xhat, value_to_beat=existing_vfa - val_without_delta_1)

            duality_gap_1 = opt_cost - new_bound.eval_at(xhat)
            constraint_gain_1 = new_bound.eval_at(xhat) - self.eval_vfa(xhat)
            print "    Primal optimal: %.4f, dual optimal: %.4f, duality gap: %.4f" % \
                  (opt_cost, opt_cost - duality_gap_1, duality_gap_1)
            print "    Existing value function at this x: %.4f" % self.eval_vfa(xhat)
            print "    Gain from new constraint at this x: %.4f" % constraint_gain_1

            if alt_bound:
                duality_gap_2 = opt_cost - (lambda_b[0] * self.s.phi_i_x(0, xhat) -
                                            np.dot(lambda_c, self.s.h_x(xhat)) +
                                            delta_alt)
                constraint_gain_2 = (opt_cost - duality_gap_2) - self.eval_vfa(xhat)
                print "    Primal optimal: %.4f, dual optimal 2: %.4f, duality gap 2: %.4f" % \
                      (opt_cost, opt_cost - duality_gap_2, duality_gap_2)
                print "    Gain 2 from new constraint at this x: %.4f" % constraint_gain_2

                if constraint_gain_2 > 0.5 and constraint_gain_1 < 1e-3 and not dodgy_bound:
                    # Replace standard duality constraint with fancy alternative-dual one
                    print "    Using gain 2."
                    bs = self.create_gridded_lb(pi, lambda_a, lambda_b, lambda_c, xhat,
                                                lb=-3.2, ub=3.2, grid_res=0.1)
                    new_bound = Constraint(n=self.n, id_in=len(self.alpha_c_list), const_in=0.,
                                           lin_in=None, hessian_in=None, sys_in=self.s,
                                           duals_in=None, xhat_in=None, gridded_values=bs)
                    assert np.abs(new_bound.eval_at(xhat) - (opt_cost - duality_gap_2)) <= 1e-3, \
                        "New constraint defined does not take correct value at xhat = " + str(xhat) + \
                        "\n    g(xhat) = %.4f, opt_cost - duality_gap_2 = %.4f." % (new_bound.eval_at(xhat),
                                                                                    opt_cost - duality_gap_2)
                if constraint_gain_1 < 1e-3 and constraint_gain_2 < 1e-3:
                    new_bound.worthless = True
            else:
                if constraint_gain_1 < 1e-3:
                    new_bound.worthless = True

            if kkt_res >= 1e-3:
                dodgy_bound = True

        else:
            f_x_const, f_x_lin = self.s.f_x_x(None)

            # (lambda_a, lambda_b, lambda_c, pi) = self.fix_duals((np.array(lambda_a),
            #                                                      np.array(lambda_b),
            #                                                      np.array(lambda_c),
            #                                                      np.array(pi)))

            bound_out_const = (opt_cost +
                               np.dot(lambda_c, np.dot(self.s.D, xhat)) -
                               np.dot(pi, self.s.f_x_x(xhat) - f_x_const))
            bound_out_lin = -np.dot(lambda_c, self.s.D) + np.dot(pi, f_x_lin)
            bound_out_quad = 0.
            for i in range(self.s.n_beta_c):
                bound_out_const += lambda_b[i] * self.s.phi_i_x_const[i]
                bound_out_const -= lambda_b[i] * self.s.phi_i_x(i, xhat)
                bound_out_lin += lambda_b[i] * self.s.phi_i_x_lin[i]
                bound_out_quad += lambda_b[i] * self.s.phi_i_x_quad[i]

            new_bound = Constraint(n=self.n, id_in=len(self.alpha_c_list),
                                   const_in=bound_out_const, lin_in=bound_out_lin,
                                   hessian_in=bound_out_quad)

            if print_sol:
                print "s", bound_out_const, "p", bound_out_lin, "P", bound_out_quad

        # Code specific to VFA_Region subclass:
        if hasattr(self, "region") and hasattr(self, "regions"):  # region=function, regions=dict
            assert hasattr(self, 'bound_all_regions')
            if not self.bound_all_regions:
                current_region = self.region(xhat)
                new_bound.assign_regions([current_region],
                                         self.regions[current_region][0],
                                         self.regions[current_region][1])
            else:
                [box_lb1, box_ub1, box_lb2, box_ub2] = self.region_outer_bounds
                new_bound.assign_regions([r for r in self.regions.iterkeys()],
                                         A=np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]),
                                         b=[box_ub1, -box_lb1, box_ub2, -box_lb2])

        new_bound_value_at_eq = new_bound.eval_at(np.array([0.] * self.n))
        if new_bound_value_at_eq > 1e-3:
            print "  New bound takes value %.3f at origin! Not adding to model." % new_bound_value_at_eq
            new_bound.dodgy = True

        if dodgy_bound:
            new_bound.dodgy = True

        return new_bound, (sol_vec[:self.m], sol_vec[self.m:self.m+self.n],
                           sol_vec[self.m+self.n:-1], sol_vec[-1])

    def brute_force_solve_for_xhat(self, xhat, extract_constr=True, print_errors=False,
                                   u_grid_pts=10, iter_no=None, alt_bound=False, method='bisection'):
        """Solve one-stage problem by brute-force (gridding) approach.

        :param xhat: Current state
        :param extract_constr: Boolean, extract new lower-bounding function
        :param print_errors: Boolean, print error information
        :param u_grid_pts: Number of input grid points if using grid approach
        :param iter_no: Integer, iteration number of the GDDP algorithm
        :param alt_bound: Boolean, derive alternative bounding function using other dual
        :param method: 'grid' or 'bisection', use standard gridding or golden ratio bisection
        :return:
        """
        # print "  Solving brute-force optimization for x =", xhat

        if method == "grid":
            # Discrete gridding method for scalar input only
            u_min, u_max = -1., 1.
            u_res = (u_max - u_min) / float(u_grid_pts)
            u_range = np.arange(u_min,
                                u_max + u_res,
                                u_res)
            n_u_grid = len(u_range)
            opt_u, opt_cost = np.zeros(self.m, dtype=float), np.inf
            for inds in itertools.product(*[range(n_u_grid) for dim in range(self.m)]):
                u = np.array([u_range[uind] for uind in inds], dtype=float)
                cost = self.eval_u_cost(u, xhat, just_return_cost=True)
                if cost < opt_cost:
                    opt_u = copy.copy(u)
                    opt_cost = copy.copy(cost)
            return opt_u, opt_cost

        # If this code is reached, assumed to have asked for 'bisection' method

        if self.s.name == 'Inverted pendulum':
            xhat[0] = divmod(xhat[0] + np.pi, 2 * np.pi)[1] - np.pi
        if self.m == 1:

            u_min = -self.s.h[0]
            u_max = self.s.h[1]

            bisection_tol = 1e-10
            alpha_constr_tol = 1e3 * bisection_tol
            beta_constr_tol = 1e-3
            c_constr_tol = 1e-3

            # # Discrete gridding method
            # u_res = (u_max - u_min) / float(u_grid_pts)
            # u_range = np.arange(u_min,
            #                     u_max + u_res,
            #                     u_res)
            # n_u_grid = len(u_range)
            # cost = np.array([np.inf] * n_u_grid, dtype=float)
            #
            # for i, u in enumerate(u_range):  # Just for the output plots
            #     u_eval = np.array([u], dtype=float)
            #     beta_vec, alpha = self.eval_u_cost(xhat, u_eval)
            #     cost[i] = np.sum(beta_vec) + self.s.gamma * alpha

            # opt_u_index = np.argmin(cost)
            # opt_cost = cost[opt_u_index]
            # opt_u = np.array([u_range[opt_u_index]])

            # Bisection method (Golden ratio)
            opt_u = self.bisection_search(xhat, u_min, u_max, bisection_tol)

            opt_x_plus = self.s.get_x_plus(xhat, opt_u)
            opt_beta_vec = self.s.beta_vec(xhat, opt_u)
            opt_alpha = -np.inf
            for c in self.alpha_c_in_model:
                opt_alpha = max(opt_alpha, c.eval_at(opt_x_plus))
            opt_cost = np.sum(opt_beta_vec) + self.s.gamma * opt_alpha

            # if extract_constr:
            #     plt.figure()
            #     plt.plot(u_range, cost)
            #     plt.plot(opt_u, opt_cost, 'ro')
            #     plt.xlabel('u')
            #     plt.ylabel('Cost')
            #     plt.title('Iteration %d: ' % iter_no + str(xhat))
            #     plt.savefig('output/' + self.s.name + '/bf_cost_%d.pdf' % iter_no)
            #     plt.close()

            sol_vec = np.hstack((opt_u, opt_x_plus, opt_beta_vec, opt_alpha))
            if not extract_constr:
                return sol_vec, opt_cost
            else:
                # Solve KKT system to find dual variables
                alpha_constr_active = [False if opt_alpha >= c.eval_at(opt_x_plus) + alpha_constr_tol
                                       else True
                                       for c in self.alpha_c_in_model]
                # print [opt_alpha - c.eval_at(opt_x_plus) for c in self.alpha_c_in_model]
                n_alpha_active = alpha_constr_active.count(True)
                print "  %d active g_i(x+) constraints:" % n_alpha_active, \
                    [ix for ix, active in enumerate(alpha_constr_active) if active]

                beta_constr_active = [False] * self.s.n_beta_c
                for l in range(self.s.n_beta_c):
                    beta_constr_active[l] = (False if
                                             np.dot(self.s.li[l], opt_beta_vec) >=
                                             (self.s.phi_i_x(l, xhat) +
                                              np.dot(self.s.ri[l], opt_u) +
                                              0.5 * np.dot(opt_u, np.dot(self.s.Ri[l], opt_u)) +
                                              self.s.ti[l]) + beta_constr_tol
                                             else True)
                n_beta_active = beta_constr_active.count(True)

                c_constr_val = np.dot(self.s.E, opt_u) - self.s.h_x(xhat)
                c_constr_active = [False if c_constr_val[j] <= -c_constr_tol
                                   else True
                                   for j in range(self.s.E.shape[0])]
                n_c_active = c_constr_active.count(True)

                alpha_kkt_mat_list = []
                for i, tf in enumerate(alpha_constr_active):
                    if tf:
                        alpha_kkt_mat_list.append(self.alpha_c_in_model[i].grad_at(opt_x_plus))
                beta_kkt_mat_list_1 = []
                beta_kkt_mat_list_2 = []
                for i, tf in enumerate(beta_constr_active):
                    if tf:
                        beta_kkt_mat_list_1.append((self.s.ri[i] +
                                                    np.dot(self.s.Ri[i], opt_u)).reshape((self.m, 1)))
                        beta_kkt_mat_list_2.append(self.s.li[i])
                c_kkt_mat_list = []
                for i, tf in enumerate(c_constr_active):
                    if tf:
                        c_kkt_mat_list.append(self.s.E.T[:, i])

                # Store KKT stationarity conditions as kkt_mat * [duals]' = kkt_rhs
                kkt_mat = np.zeros((self.m + self.n,
                                    self.n + n_c_active + n_beta_active + n_alpha_active),
                                   dtype=float)
                kkt_rhs = np.zeros((self.m + self.n,), dtype=float)

                kkt_mat[:self.m, :self.n] = self.s.f_u_x(xhat).T
                kkt_mat[self.m:self.m+self.n, :self.n] = -np.eye(self.n)
                for i in range(n_beta_active):
                    kkt_mat[:self.m, self.n + n_c_active + i] = beta_kkt_mat_list_1[i]
                for i in range(n_c_active):
                    kkt_mat[:self.m, self.n + i] = c_kkt_mat_list[i]
                # Gradient of active g_i(x) constraints
                for i in range(n_alpha_active):
                    kkt_mat[self.m:self.m + self.n,
                            self.n + n_c_active + n_beta_active + i] = alpha_kkt_mat_list[i]

                # print opt_u, kkt_mat, kkt_rhs

                # Indicate which Lagrange multipliers are nonnegative (lambdas)
                nonnegative = [False] * self.n + [True] * (n_c_active + n_beta_active + n_alpha_active)
                # Length of each Lagrange multiplier vector
                shape_info = [self.n, n_c_active, n_beta_active, n_alpha_active]

                kkt_sol, residual = self.bf_kkt_lsq(kkt_mat, kkt_rhs, nonnegative,
                                                    shape_info,
                                                    beta_kkt_mat_list_2)

                pi = kkt_sol[:self.n]
                lambda_c_active = kkt_sol[self.n:self.n + n_c_active]
                lambda_b_active = kkt_sol[self.n + n_c_active:self.n + n_c_active + n_beta_active]
                lambda_a_active = kkt_sol[-n_alpha_active:]

                if np.abs(residual) >= 1e-3:
                    print "  Warning: KKT solution residual = %.4f" % residual
                    print "pi:", pi
                    print "lambda_c_active:", lambda_c_active
                    print "lambda_b_active:", lambda_b_active
                    print "lambda_a_active:", lambda_a_active
                    print "n_c_active:", n_c_active
                    print "n_beta_active:", n_beta_active
                    print "n_alpha_active:", n_alpha_active
                    print "kkt_mat:"
                    print kkt_mat
                    assert np.abs(residual) <= 0.1, "Something's gone very wrong"
                else:
                    print "  KKT solution residual = %.6f" % residual

                # Insert the active alpha multipliers into the vector lambda_alpha
                lambda_a, laa_index = np.zeros((len(self.alpha_c_in_model),), dtype=float), 0
                for i, tf in enumerate(alpha_constr_active):
                    if tf:
                        lambda_a[i] = lambda_a_active[laa_index]
                        laa_index += 1

                # delta_1 = self.delta_1(pi, lambda_a, grid_res=0.1,
                #                        output_dir='output/' + self.s.name,
                #                        iter_no=iter_no, xhat=xhat)

                delta_1 = None  # delta_1 is in general x-dependent and shouldn't be calculated here

                lambda_b, lba_index = np.zeros((self.s.n_beta_c,), dtype=float), 0
                for i, tf in enumerate(beta_constr_active):
                    if tf:
                        lambda_b[i] = lambda_b_active[lba_index]
                        lba_index += 1

                lambda_c, lca_index = np.zeros((self.s.h.shape[0],), dtype=float), 0
                for i, tf in enumerate(c_constr_active):
                    if tf:
                        lambda_c[i] = lambda_c_active[lca_index]
                        lca_index += 1

                if alt_bound:
                    delta_alt = self.delta_alt(pi, lambda_a, lambda_b, lambda_c,
                                               grid_res=0.1,
                                               output_dir='output/' + self.s.name, iter_no=iter_no,
                                               xhat=xhat)
                else:
                    delta_alt = None

                duals = (lambda_a, lambda_b, lambda_c, pi)

                return sol_vec, opt_cost, delta_1, delta_alt, duals, residual

        else:
            print "Cannot solve by brute force by golden ratio bisection for m > 1!"
            raise SystemExit()

    def bisection_search(self, xhat, u_min, u_max, bisection_tol):
        """Search over scalar interval [u_min, u_max] for optimal input

        :param xhat: n-element vector, current state
        :param u_min: Scalar lower bound on input
        :param u_max: Scalar upper bound on input
        :param bisection_tol: Scalar, iterations stop when successive inputs queried are this close
        :return: Optimal input, 1-element vector
        """

        # result = minimize_scalar(self.eval_u_cost, args=(xhat, True),
        #                          bounds=[u_min, u_max],
        #                          method='Golden', tol=bisection_tol)
        # scipy_result = result['x']
        # return np.array([scipy_result])

        a_n = copy.copy(u_min)
        b_n = copy.copy(u_max)
        rho = 0.5 * (np.sqrt(5.) + 1)

        c_n = b_n - (b_n - a_n) / rho
        d_n = a_n + (b_n - a_n) / rho
        while abs(c_n - d_n) > bisection_tol:
            c_cost = self.eval_u_cost(np.array([c_n]), xhat, just_return_cost=True)
            d_cost = self.eval_u_cost(np.array([d_n]), xhat, just_return_cost=True)
            if c_cost < d_cost:
                b_n = d_n
            else:
                a_n = c_n
            c_n = b_n - (b_n - a_n) / rho
            d_n = a_n + (b_n - a_n) / rho

        # assert np.abs(scipy_result - (a_n + b_n) / 2.) <= bisection_tol * 20, str(scipy_result) + " %.10f" % ((a_n + b_n) / 2.)

        return np.array([(a_n + b_n) / 2.])

    def eval_u_cost(self, u, xhat, just_return_cost=False):
        """Evaluate 1-stage cost + cost-to-go for a given input u

        :param u: m-element vector, input to apply
        :param xhat: n-element vector, current state
        :param just_return_cost: Boolean, if false return stage cost + cost-to-go. Else beta & alpha
        :return: Cost
        """
        # print 'u:', u
        # print 'xhat:', xhat
        x_plus = self.s.get_x_plus(xhat, u)
        beta_vec = self.s.beta_vec(xhat, u)
        alpha = -np.inf
        for c in self.alpha_c_in_model:
            alpha = max(alpha, c.eval_at(x_plus))
        if just_return_cost:
            return np.sum(beta_vec) + self.s.gamma * alpha
        else:
            return beta_vec, alpha

    def bf_kkt_lsq(self, A, b, nonnegative, shape_info, lambda_b_active_li):
        """Solve min_x ||Ax - b||^2 s.t. Cx = d, with some variables nonnegative, in order to solve
        KKT system for optimal dual variables of brute force problem. The constraints Cx = d consist
        of sum of lambda_alpha = gamma, and L^T lambda_beta = 1

        :param A: Matrix with same number of columns as len(nnegative)
        :param b: Vector with same number of elements as rows of A
        :param nonnegative: List of Booleans true if variable x_i >= 0
        :param shape_info: Dimensions of [pi, lambda_c, lambda_beta, lambda_alpha]
        :param lambda_b_active_li: l_i for active lambda_beta constraints
        :return: vector of optimal dual variables, residual of KKT solution
        """
        m = gu.Model('kkt_lsq')
        m.params.outputflag = 0
        n_vars = A.shape[1]
        lambda_a_indices = range(n_vars - shape_info[3], n_vars)
        assert np.sum(shape_info) == n_vars
        assert len(nonnegative) == n_vars
        x = []
        for i in range(n_vars):
            if nonnegative[i]:
                x.append(m.addVar(name="u_%d" % (i + 1), lb=0., obj=0.))
            else:
                x.append(m.addVar(name="u_%d" % (i + 1), lb=-1e100, obj=0.))

        ATA = np.dot(A.T, A)
        qe = gu.QuadExpr()
        for p in range(n_vars):
            qe += 0.5 * x[p] * gu.LinExpr([(ATA[p, q], x[q]) for q in range(n_vars)])
        bA = np.dot(b, A)
        le = gu.LinExpr([(-bA[q], x[q]) for q in range(n_vars)])
        m.setObjective(qe + le + 0.5 * np.dot(b, b))
        # Add constraint sum of lambda_alpha = gamma
        m.addConstr(gu.LinExpr([(1.0, x[i]) for i in lambda_a_indices]) == self.s.gamma)
        # Add constraint L^T lambda_beta = 1
        for i in range(lambda_b_active_li[0].shape[0]):
            m.addConstr(gu.LinExpr([(l[i], x[self.n + shape_info[1] + j])
                                    for (j, l) in enumerate(lambda_b_active_li)]) == 1)
        m.optimize()
        if m.status == 2:
            opt_cost = m.getObjective().getValue()
            sol_vec = [v.x for v in m.getVars()]
            # print "sol_vec, opt_cost:", sol_vec, opt_cost
            return sol_vec, opt_cost
        else:
            print "Optimization status for KKT system solve was %d!" % m.status
            raise SystemExit()

    def fix_duals(self, duals_in):
        """Minimize (lambda - lambda_raw)^2 subject to feasibility constraints on the dual
        variables.

        :param duals_in: Tuple lambda_a_raw, lambda_b_raw, lambda_c_raw, pi_raw
        :return: lambda_a_fixed, lambda_b_fixed, lambda_c_raw, pi_raw
        """
        lambda_a_raw, lambda_b_raw, lambda_c_raw, pi_raw = duals_in

        m_a = gu.Model('Fix alpha duals')
        m_a.params.outputflag = 0
        qe = gu.QuadExpr()
        le = gu.LinExpr()
        n_alpha = lambda_a_raw.shape[0]
        x = []
        for i in range(n_alpha):
            x.append(m_a.addVar(lb=0.0, vtype='c'))
            qe += x[i] * x[i]
            le += -2 * lambda_a_raw[i] * x[i]
        m_a.update()
        m_a.setObjective(qe + le)
        m_a.update()
        m_a.addConstr(gu.LinExpr([(1.0, var) for var in x]) == self.s.gamma)
        m_a.optimize()
        if m_a.status == 2:
            sol_vec = [v.x for v in m_a.getVars()]
            lambda_a_fixed = np.array(sol_vec)
        else:
            print "Fixing duals in constraint failed! Exiting."
            print "lambda_a status:", m_a.status
            print "lambda_a raw:", str(lambda_a_raw)
            raise SystemExit()
        m_b = gu.Model('Fix beta duals')
        m_b.params.outputflag = 0
        qe = gu.QuadExpr()
        le = gu.LinExpr()
        n_beta_c = lambda_b_raw.shape[0]
        y = []
        for i in range(n_beta_c):
            y.append(m_b.addVar(lb=0.0, vtype='c'))
            qe += y[i] * y[i]
            le += -2 * lambda_b_raw[i] * y[i]
        m_b.setObjective(qe + le)
        m_b.update()
        for l in range(self.s.n_beta):
            m_b.addConstr(gu.LinExpr([(self.s.li[i][l], var) for i, var in enumerate(y)]) == 1.0)
        m_b.optimize()
        if m_b.status == 2:
            sol_vec = [v.x for v in m_b.getVars()]
            lambda_b_fixed = np.array(sol_vec)
        else:
            print "Fixing duals in constraint failed! Exiting."
            print "lambda_b status:", m_b.status
            raise SystemExit()

        if not np.linalg.norm(lambda_a_fixed - lambda_a_raw) <= 1e-3:
            print "Warning:", str(lambda_a_fixed) + str(lambda_a_raw)
        if not np.linalg.norm(lambda_b_fixed - lambda_b_raw) <= 1e-3:
            print "Warning:", str(lambda_b_fixed) + str(lambda_b_raw)

        return lambda_a_fixed, lambda_b_fixed, lambda_c_raw, pi_raw

    def delta_alt(self, pi, lambda_a, lambda_b, lambda_c,
                  grid_res=0.01, output_dir=None, iter_no=None, xhat=None):
        """

        :param pi:
        :param lambda_a:
        :param lambda_b:
        :param lambda_c:
        :param grid_res:
        :param output_dir:
        :param iter_no:
        :param xhat:
        :return:
        """
        if self.m == 1:
            u_min = -self.s.h[0]
            u_max = self.s.h[1]

            u_range = np.arange(u_min,
                                u_max + grid_res,
                                grid_res)
            n_u_grid = len(u_range)
            delta_alt_grid = np.array([-np.inf] * n_u_grid, dtype=float)

            assert len(lambda_a) == len(self.alpha_c_in_model)

            for i, u in enumerate(u_range):
                u_eval = np.array([u], dtype=float)
                x_plus = self.s.get_x_plus(xhat, u_eval)
                delta_alt_grid[i] = (np.dot(lambda_c, np.dot(self.s.E, u_eval)) +
                                     lambda_b[0] * (np.dot(self.s.ri[0], u_eval) +
                                                    0.5 * np.dot(u_eval, np.dot(self.s.Ri[0], u_eval))) +
                                     np.sum([lambda_a[k] * c.eval_at(x_plus)
                                             for k, c in enumerate(self.alpha_c_in_model)]))

            min_u_index = np.argmin(delta_alt_grid)

            # Plot minimization
            # plt.figure()
            # plt.plot(u_range, delta_alt_grid)
            # plt.plot(u_range[min_u_index], delta_alt_grid[min_u_index], 'ro')
            # plt.ylim([np.min(delta_alt_grid) - 0.02, np.max(delta_alt_grid) + 0.02])
            # plt.xlabel('$u$')
            # plt.ylabel('UB on $\delta_1(\\pi, \\lambda_\\alpha$)')
            # plt.title('Minimization for $\\delta_1(\\pi, \\lambda_\\alpha$), iter. %d' % iter_no)
            # plt.savefig(output_dir + '/delta_alt_min_%d.pdf' % iter_no)
            # plt.close()

            return delta_alt_grid[min_u_index]
        else:
            print "Cannot evaluate delta_1(pi, alpha) for input dimension greater than 1!"

    def create_gridded_lb(self, pi, lambda_a, lambda_b, lambda_c, xhat, lb, ub, grid_res):
        if self.n == 2:
            print "  Creating gridded constraint based on alternative dual formulation"
            x1 = np.arange(lb, ub + grid_res, grid_res).tolist()
            x2 = np.arange(lb, ub + grid_res, grid_res).tolist()
            g_of_x = np.zeros((len(x1), len(x2)), dtype=float)
            for ind1, x1_val in enumerate(x1):
                print "    x1 = %.2f..." % x1_val
                for ind2, x2_val in enumerate(x2):
                    x = copy.copy(np.array([x1_val, x2_val]))
                    g_of_x[ind1, ind2] = lambda_b[0] * self.s.phi_i_x(0, x) - \
                                         np.dot(lambda_c, self.s.h_x(x)) + \
                                         self.delta_alt(pi, lambda_a, lambda_b, lambda_c, xhat=x)

            bs = RectBivariateSpline(x1, x2, g_of_x)
            return bs
        else:
            print "Cannot create gridded lb for state dimension > 2!"
            raise SystemExit()

    def consolidate_bounds(self, lb=-3.2, ub=3.2, grid_res=0.02):
        # Consolidates all lower bounds generated so far into one gridded function, and deletes
        # the existing lower-bounding constraints.
        if self.n == 2:
            print "  Consolidating lower bounding constraints..."
            x1 = np.arange(lb, ub + grid_res, grid_res).tolist()
            x2 = np.arange(lb, ub + grid_res, grid_res).tolist()
            v_of_x = np.zeros((len(x1), len(x2)), dtype=float)
            for ind1, x1_val in enumerate(x1):
                print "    x1 = %.2f..." % x1_val
                for ind2, x2_val in enumerate(x2):
                    x = copy.copy(np.array([x1_val, x2_val]))
                    v_of_x[ind1, ind2] = self.eval_vfa(x)
            bs = RectBivariateSpline(x1, x2, v_of_x)
            for c in self.alpha_c_in_model:
                self.remove_alpha_constraint(c)
            new_bound = Constraint(n=self.n, id_in=len(self.alpha_c_list), const_in=0.,
                                   lin_in=None, hessian_in=None, sys_in=self.s,
                                   duals_in=None, xhat_in=None, gridded_values=bs)
            self.add_alpha_constraint(new_bound)
        else:
            print "Cannot create gridded lb for state dimension > 2!"
            raise SystemExit()

    def value_iteration(self, lb, ub, grid_res, n_iters, conv_tol):
        """Perform standard gridded value iteration to a desired tolerance (max change in value from
        one iteration to the next at any grid point). Uses same grid range and resolution for each
        coordinate

        :param lb: Lower bound for all states
        :param ub: Upper bound for all states
        :param grid_res: Scalar grid resolution
        :param n_iters: Maximum number of iterations to carry out (breaks if it converges earlier)
        :param conv_tol: Convergence tolerance (max change in iteration across all grid points)
        :return: Tuple of (time taken, largest change in iteration, iterations taken)
        """
        init_constr = Constraint(n=self.n, id_in=0, const_in=0., sys_in=self.s,
                                 duals_in=None, gridded_values=None)
        self.add_alpha_constraint(init_constr)

        t1 = time.time()
        coord_range = np.arange(lb, ub + grid_res, grid_res)
        pts_per_coordinate = len(coord_range.tolist())
        vk_of_x = np.zeros([pts_per_coordinate] * self.n, dtype=float)
        # uk_of_x = np.zeros([pts_per_coordinate for ind in range(self.n)], dtype=float)
        converged, iters_taken = False, 0
        largest_delta = 0.
        for k in range(n_iters):
            print "Performing value iteration #%d..." % k
            largest_delta = 0.
            for inds in itertools.product(*[range(pts_per_coordinate) for dim in range(self.n)]):
                if self.n >= 4 and inds[-2] == inds[-1] == 0:
                    print "Current grid indices:", inds
                x = np.array([coord_range[xind] for xind in inds])
                _, val = self.brute_force_solve_for_xhat(x, False, False, method='grid')
                if self.eval_vfa(x) == -np.inf:
                    print "k: " + str(k) + ", x: " + str(x) + ", inds: " + str(inds)
                    for c in self.alpha_c_in_model:
                        print "   ", c.id, c.eval_at(x)
                vk_of_x.itemset(inds, val)
                largest_delta = max(largest_delta,
                                    np.abs(vk_of_x.__getitem__(inds) - self.eval_vfa(x)))
                # uk_of_x[inds[0], inds[1]] = copy.copy(u[0])
                # print vk_of_x.__getitem__(inds), self.eval_vfa(x)
            if largest_delta <= conv_tol:
                converged = True
            print "  Max vk_of_x: %.4f." % np.max(vk_of_x), "Largest delta: %.4f" % largest_delta
            print "  Time elapsed: %.1f seconds." % (time.time() - t1)
            # if self.n == 1:
            #     bs = UnivariateSpline(coord_range, vk_of_x)
            # elif self.n == 2:
            #     bs = RectBivariateSpline(coord_range, coord_range, vk_of_x)
            bs = RegularGridInterpolator([coord_range] * self.n, vk_of_x,
                                         method='linear', bounds_error=False, fill_value=0.)
                                         # Assumes all coords have same grid!
            new_c = Constraint(self.n, k+1, gridded_values=copy.deepcopy(bs))
            for old_c in self.alpha_c_in_model:
                self.remove_alpha_constraint(old_c)
            self.add_alpha_constraint(new_c)
            # self.plot_vfa(save=True, output_dir='output/'+self.s.name, iter_no=k)
            # self.plot_policy(save=True, iter_no=k, output_dir='output/'+self.s.name,
            #                  policy_in=(coord_range, coord_range, uk_of_x))
            if converged:
                iters_taken = k + 1
                print "Converged to within %.4f after %d iterations." % (conv_tol, k+1)
                break
        if not converged:
            iters_taken = n_iters
        t2 = time.time()

        return (t2 - t1), largest_delta, iters_taken

    def add_saved_constraint(self, filename):
        """Add a saved 2-dimensional lower bounding constraint to the model, loaded from filename"""
        loaded = loadmat('output/' + self.s.name + '/' + filename)
        x1 = loaded['X'][0]
        x2 = [y[0] for y in loaded['Y']]
        v_of_x = loaded['mesh']
        bs = RectBivariateSpline(x1, x2, v_of_x)
        new_c = Constraint(self.n, 1, gridded_values=bs)
        self.add_alpha_constraint(new_c)

    def add_alpha_constraint(self, constr_in, was_temp_removed=False, temp_add=False):
        """Add a value function lower-bounding constraint to the solver model (in case a solver is
        being used) and in any case, add it to the book-keeping list of all constraints currently in
        the model.

        :param constr_in: Constraint object to add to the model
        :param was_temp_removed: Boolean. If True, constr_in.in_model = True, but is not in model
        :param temp_add: Boolean. If True, add to gurobi model but don't set constr_in.in_model.
        :return: nothing
        """
        constr_id = constr_in.id
        const, lin, hess = constr_in.const, constr_in.lin, constr_in.hessian
        if temp_add:
            c_name = str(constr_id)
        else:
            c_name = "alpha_%d" % constr_id
        if self.solver == 'gurobi':
            x = [v for v in self.mod.getVars()]
            if const is None:
                const = 0.
            le = gu.LinExpr()
            if lin is not None:
                le = gu.LinExpr([(lin[p], x[self.m + p]) for p in range(self.n)])
            qe = gu.QuadExpr()
            if hess is not None:
                for p in range(self.n):
                    qe += 0.5 * x[self.m + p] * gu.LinExpr([(hess[p, q], x[self.m + q])
                                                            for q in range(self.n)])
            self.mod.addConstr(x[-1] >= const + le + qe, name=c_name)
            self.mod.update()
        elif self.solver == 'ecos':
            lin_ecos = np.hstack((np.zeros((self.m,)),
                                  lin,
                                  np.zeros((self.n_beta,)),
                                  np.array([-1.])))
            quad_ecos = scipy.linalg.block_diag(np.zeros((self.m, self.m)),
                                                hess,
                                                np.zeros((self.n_beta + 1, self.n_beta + 1)))
            G_out, h_out, type_out = self.build_ecos_constraint(const, lin_ecos, quad_ecos)
            self.ecos_alpha_constrs.append((G_out, h_out, type_out, constr_id))
        else:
            if self.s.brute_force_solve:
                pass
            else:
                print "Solver " + self.solver + " not implemented!"
                raise SystemExit()
        # Add to list of model constraints and mark as added to model
        if not was_temp_removed and not temp_add:
            self.alpha_c_list.append(constr_in)
            constr_in.added_to_model()
        # if temp_add:
        #     print "Temporarily added virtual constraint " + c_name + "."
        # elif was_temp_removed:
        #     print "Re-added temporarily removed constraint " + c_name + "."
        # else:
        #     print "Added constraint " + c_name + "."
        self.alpha_c_in_model = [c for c in self.alpha_c_list if c.in_model]

    def remove_alpha_constraint(self, constr_in, was_temp_added=False, temp_remove=False):
        """Remove a value function lower-bounding constraint from the solver model (in case solver
        being used) and in any case, remove it from the book-keeping list of all constraints
        currently in the model.

        :param constr_in: Constraint object to remove from the model
        :return: nothing
        """
        if not was_temp_added:
            assert constr_in.in_model, "Constraint " + str(constr_in.id) + " not in model!"
            c_name = 'alpha_%d' % constr_in.id
        else:
            c_name = constr_in.id
        successfully_removed = False
        if self.solver == 'gurobi':
            if self.mod.getConstrByName(c_name) is not None:
                # Constraint is linear
                self.mod.remove(self.mod.getConstrByName(c_name))
                successfully_removed = True
            else:
                for c in self.mod.getQConstrs():
                    if c.QCName == c_name:
                        self.mod.remove(c)
                        successfully_removed = True
        elif self.solver == 'ecos':
            for tup in self.ecos_alpha_constrs:
                if tup[3] == constr_in.id:
                    del tup
                    successfully_removed = True
        else:
            if self.s.brute_force_solve:
                successfully_removed = True
            else:
                print "Solver " + self.solver + " not implemented!"
                raise SystemExit()
        assert successfully_removed, "Didn't manage to remove constraint " + c_name + "!"
        if not was_temp_added and not temp_remove:
            constr_in.removed_from_model()
        # if temp_remove:
        #     print "Temporarily removed constraint " + c_name + "."
        # elif was_temp_added:
        #     print "Removed temporary virtual constraint " + c_name + "."
        # else:
        #     print "Permanently removed temporary constraint " + c_name + "."
        self.alpha_c_in_model = [c for c in self.alpha_c_list if c.in_model]

    def reset_model_constraints(self):
        """Remove all constraints from the model and start with a zero lower-bounding function"""
        for c in self.alpha_c_in_model:
            self.remove_alpha_constraint(c)
        # Add initial alpha epigraph constraint
        initial_lb = Constraint(self.n, 0, 0., None, None)
        self.add_alpha_constraint(initial_lb)

    def remove_redundant_lbs(self, n_samples, threshold=0., sigma_in=1., mu_in=1.):
        """Use a sampling approach to remove lower-bounding functions that are found not to be
        active at a certain threshold number of points

        :param n_samples: Number of Monte Carlo samples
        :param threshold: Fraction of MC samples below which a constraint should be considered
            redundant
        :param sigma_in: Standard deviation of MC samples (in all coordinates)
        :param mu_in: Mean of MC samples (in all coordinates)
        :return: nothing
        """
        sigma, mu = sigma_in, mu_in
        c_in_model = [c for c in self.alpha_c_list if c.in_model]
        n_in_model = len(c_in_model)
        val_array = np.zeros(n_in_model, dtype=float)
        active_score = np.zeros(n_in_model, dtype=int)
        for sn in range(n_samples):
            xn = sigma * np.random.randn(self.n) + mu
            for i, c in enumerate(c_in_model):
                val_array[i] = c.eval_at(xn)
            active_score[np.argmax(val_array)] += 1
        for i in range(n_in_model):
            if active_score[i] <= threshold * n_samples:
                self.remove_alpha_constraint(c_in_model[i])

    def eval_vfa(self, x_in):
        """Evaluates lower bound (max of all LB functions) at x_in by evaluating all LBs"""

        # if self.s.name == 'Inverted pendulum':
        #     x_in[0] = divmod(x_in[0] + np.pi, 2 * np.pi)[1] - np.pi
        f_out = -np.inf
        for c in self.alpha_c_in_model:
            f_out = max(f_out, c.eval_at(x_in))
        return f_out

    def print_constraint_info(self):
        """Output basic information about constraints in a Gurobi solver model."""
        if self.solver == 'gurobi':
            m = self.mod
            lin_constrs = m.getConstrs()
            quad_constrs = m.getQConstrs()
            print "Linear constraints:"
            for c in lin_constrs:
                print "  " + c.ConstrName + ": ", m.getRow(c), c.getAttr('Sense'), c.getAttr('RHS')
            print "Quadratic constraints:"
            for c in quad_constrs:
                print "  " + c.QCName + ": ", m.getQCRow(c), c.getAttr('QCSense'), c.getAttr('QCRHS')
        else:
            print "print_constraint_info(): Can only output constraint info for Gurobi models!"

    def plot_vfa(self, iter_no=None, save=False, output_dir=None):
        """Plot value function approximation in 1 or 2 dimensions. Plot bounds and resolution are
        currently hard-coded inside the function.

        :param iter_no: Integer, iteration number
        :param save: Boolean, save value function plot. If false, show on screen instead
        :param output_dir: Output directory for plots
        :return: nothing
        """
        # Extract value function lower bounds from model constraints, then plot
        plt.figure()
        if save:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        if self.n == 1:
            std_devs_to_plot = 2.
            x_min = -std_devs_to_plot * np.sqrt(self.s.state_var[0, 0])
            x_max = std_devs_to_plot * np.sqrt(self.s.state_var[0, 0])
            res = 0.025
            plot_range = [np.array([p], dtype=float) for p in np.arange(x_min, x_max + res, res)]
            v_max = [0.] * len(plot_range)
            v = np.zeros((0, len(plot_range)))

            # Evaluate all lower-bounding functions
            c_in_model = [c for c in self.alpha_c_list if c.in_model]
            for c in c_in_model:
                # Maximum of lower-bounding functions
                for ind, x_plot in enumerate(plot_range):
                    v_max[ind] = max(v_max[ind], c.eval_at(x_plot))
                v_max = np.array(v_max)
                # Individual lower-bounding functions
                v_temp = np.zeros((1, len(plot_range)), dtype=float)
                for ind, x_plot in enumerate(plot_range):
                    v_temp[0, ind] = c.eval_at(x_plot)
                v = np.vstack((v, v_temp))

            # Plot results
            plt.plot(plot_range, v.T, alpha=0.2, color='k')
            plt.plot(plot_range, v_max, linewidth=1.5, color='k', label='$\hat{V}$' + '$(x)$')

            if self.s.dare_sol is not None and self.s.pure_lqr:
                plt.plot(plot_range, [0.5 * x ** 2 * self.s.dare_sol[0, 0] for x in plot_range],
                         linewidth=1.0, color='r', label='$V^\star$' + '$(x)$ (unconstrained LQR)')

            plt.legend()
            plt.xlim([x_min, x_max])
            plt.xlabel('$x$')
            plt.ylim([-0.05, 4.00])
            if self.s.dare_sol is not None:
                dare_cost_at_bound = 0.5 * self.s.dare_sol[0, 0] * x_max * x_max
                plt.ylim([-0.05, 1.5 * dare_cost_at_bound])
            plt.ylabel('$V_J(x)$')
            if iter_no is not None:
                plt.title('Value function approximation, %d iterations' % iter_no)
            else:
                plt.title('Value function approximation')
            plt.grid()

        elif self.n == 2 or self.n == 4:
            x1_min, x1_max, res1 = -6, 1, 0.1
            x2_min, x2_max, res2 = -3, 3, 0.1
            x1 = np.arange(x1_min, x1_max + res1, res1).tolist()
            x2 = np.arange(x2_min, x2_max + res2, res2).tolist()
            mesh = np.zeros((len(x1), len(x2)), dtype=float)

            c_in_model = [c for c in self.alpha_c_list if c.in_model]
            if self.n == 2:
                for c in c_in_model:
                    for ind1, x1_plot in enumerate(x1):
                        for ind2, x2_plot in enumerate(x2):
                            mesh[ind1, ind2] = max(mesh[ind1, ind2],
                                                   c.eval_at(np.array([x1_plot, x2_plot])))
            elif self.n == 4:
                for c in c_in_model:
                    for ind1, x1_plot in enumerate(x1):
                        for ind2, x2_plot in enumerate(x2):
                            mesh[ind1, ind2] = max(mesh[ind1, ind2],
                                                   c.eval_at(np.array([x1_plot, 0., x2_plot, 0.])))

            X, Y = np.meshgrid(x1, x2)
            if self.x_visited is not None:
                savemat(output_dir + '/vfa_%d.mat' % iter_no, {'X': X, 'Y': Y, 'mesh': mesh,
                                                               'visited': self.x_visited})
            else:
                savemat(output_dir + '/vfa_%d.mat' % iter_no, {'X': X, 'Y': Y, 'mesh': mesh})

            # Axes3D.plot_wireframe(X, Y, mesh)
            plt.imshow(mesh.T, aspect='equal', origin='lower', cmap='bone',
                       extent=[x1_min, x1_max, x2_min, x2_max], vmin=0, vmax=20)
            plt.scatter(X, Y, s=2, alpha=0.2, linewidths=0)
            plt.xlim([x1_min, x1_max])
            plt.ylim([x2_min, x2_max])
            if self.x_visited is not None:
                if self.n == 2:
                    plt.xlabel('$x_1$')
                    plt.ylabel('$x_2$')
                    plt.scatter([x[0] for x in self.x_visited],
                                [x[1] for x in self.x_visited], s=3, c='w', linewidths=0)
                elif self.n == 4:
                    plt.xlabel('$r$')
                    plt.ylabel('$\\theta$')
                    plt.scatter([x[0] for x in self.x_visited],
                                [x[2] for x in self.x_visited], s=3, c='w', linewidths=0)
            if iter_no is not None:
                plt.title('Value function approximation, %d iterations' % iter_no)
            else:
                plt.title('Value function approximation')
        else:
            print "    Cannot plot value function approximation for n = %d" % self.n
            return

        # Save or plot VFA
        if save:
            plt.savefig(output_dir + '/vfa_%d.pdf' % iter_no)
        else:
            plt.show()
        plt.close()

    def plot_policy(self, iter_no=None, save=False, output_dir=None, policy_in=None):
        """Save or plot control policy for 1 or 2 dimensions. In 2D, omputes the control policy at
        every grid point unless policy_in is not None

        :param iter_no: Iteration number to use as label in graph title/filename
        :param save: Boolean. If true, saves plot. If false, shows on screen
        :param output_dir: Output directory
        :param policy_in: Tuple (x1 range, x2 range, policy values) for 2D policy plot
        :return: nothing
        """
        if self.n == 1 and self.m == 1:
            plt.figure()
            x_min, x_max, res = -3, 3, 0.01
            plot_range = [np.array([p], dtype=float) for p in np.arange(x_min, x_max + res, res)]
            u = [0.] * len(plot_range)
            for i, x_plot in enumerate(plot_range):
                u[i], _, _ = self.solve_for_xhat(x_plot, extract_constr=False)
            u = np.array(u)
            # Plot results
            plt.plot([a[0] for a in plot_range], u)
            plt.xlim([x_min, x_max])
            plt.xlabel('$x$')
            plt.ylim([-1.1, 1.1])
            plt.ylabel('$u^\star(x)$')
            plt.title('Control policy, iteration %d' % iter_no)
            plt.grid()
        elif (self.n == 2 or self.n == 4) and self.m == 1:
            plt.figure()
            if policy_in is not None:
                x1 = policy_in[0]
                x2 = policy_in[1]
                u_mesh = policy_in[2]
                x1_min, x1_max, x2_min, x2_max = np.min(x1), np.max(x1), np.min(x2), np.max(x2)
                savemat(output_dir + '/policy_%d.mat' % iter_no,
                        {'X': x1, 'Y': x2, 'u_mesh': u_mesh})
                plt.imshow(u_mesh.T, aspect='equal', origin='lower',
                           extent=[x1_min, x1_max, x2_min, x2_max])
                plt.title('Control policy, iteration %d' % iter_no)
            else:
                x1_min, x1_max, res1 = -3.2, 3.2, 0.4
                x2_min, x2_max, res2 = -1.6, 1.6, 0.2
                x1 = np.arange(x1_min, x1_max + res1, res1).tolist()
                x2 = np.arange(x2_min, x2_max + res2, res2).tolist()
                u_mesh = np.zeros((len(x1), len(x2)), dtype=float)
                if self.n == 2:
                    for ind1, x1_plot in enumerate(x1):
                        print x1_plot
                        for ind2, x2_plot in enumerate(x2):
                            u_mesh[ind1, ind2] = self.solve_for_xhat(np.array([x1_plot, x2_plot]),
                                                                     extract_constr=False)[0][0]
                elif self.n == 4:
                    for ind1, x1_plot in enumerate(x1):
                        print x1_plot
                        for ind2, x2_plot in enumerate(x2):
                            u_mesh[ind1, ind2] = self.solve_for_xhat(np.array([x1_plot, 0.,
                                                                               x2_plot, 0.]),
                                                                     extract_constr=False)[0][0]
                # Plot results
                X, Y = np.meshgrid(x1, x2)
                savemat(output_dir + '/policy_%d.mat' % iter_no, {'X': X, 'Y': Y, 'u_mesh': u_mesh})
                plt.imshow(u_mesh.T, aspect='equal', origin='lower',
                           extent=[x1_min, x1_max, x2_min, x2_max])
                plt.scatter(X, Y, s=2, alpha=0.3)
                plt.xlim([x1_min, x1_max])
                plt.ylim([x2_min, x2_max])
                if self.n == 2:
                    plt.xlabel('$x_1$')
                    plt.ylabel('$x_2$')
                elif self.n == 4:
                    plt.xlabel('$r$')
                    plt.ylabel('$\\theta$')
                plt.title('Control policy, iteration %d' % iter_no)
        else:
            print "    Cannot plot policy: state dim. %d, input dim. %d" % (self.n, self.m)
            return

        # Save or plot policy
        if save:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            plt.savefig('output/' + self.s.name + '/policy_%d.pdf' % iter_no)
        else:
            plt.show()
        plt.close()

    def plot_xalg_be(self, xalg_v_of_x, xalg_tv_of_x):
        """Plots the Bellman error at all elements of the set XAlg (i.e. the sample points used in
        the algorithm)

        :param xalg_v_of_x: List of V(x) with length equal to no. of elements in XAlg
        :param xalg_tv_of_x: List of TV(x) with length equal to no. of elements in XAlg
        :return: nothing
        """

        n_x_points = xalg_v_of_x.shape[0]

        if not os.path.exists('output/' + self.s.name):
            os.makedirs('output/' + self.s.name)
        plt.figure(figsize=(16, 8))
        plt.subplot(211)
        plt.bar(np.array(range(1, n_x_points + 1)) - 0.225, xalg_v_of_x,
                width=0.45, color='b', edgecolor=None)
        plt.bar(np.array(range(1, n_x_points + 1)) + 0.225, xalg_tv_of_x,
                width=0.45, color='r', edgecolor=None)
        plt.xlim([0, n_x_points + 1])
        if self.__class__.__name__ == "QFApproximator":
            plt.ylabel('$Q(x, u)$, $T_QQ(x, u)$')
        else:
            plt.ylabel('$V(x)$, $TV(x)$')
        plt.subplot(212)
        plt.bar(np.array(range(1, n_x_points + 1)) - 0.4,
                xalg_tv_of_x - xalg_v_of_x, edgecolor=None)
        plt.xlim([0, n_x_points + 1])
        if self.__class__.__name__ == "QFApproximator":
            plt.ylabel('$T_QQ(x, u) - Q(x, u)$')
        else:
            plt.ylabel('$TV(x) - V(x)$')
        plt.ylim([min(-1., np.min(xalg_tv_of_x - xalg_v_of_x) * 1.05),
                  np.max(xalg_tv_of_x - xalg_v_of_x) * 1.05])
        plt.savefig('output/' + self.s.name + '/xalg_BE.pdf')
        plt.close()


        plt.figure()
        plt.plot(xalg_tv_of_x, 100. * (xalg_tv_of_x - xalg_v_of_x) / xalg_v_of_x,
                 linewidth=0., marker='x')

        if self.__class__.__name__ == "QFApproximator":
            plt.ylabel('$(T_QQ(x,u) - Q(x,u))/Q(x,u)$ (%)')
            plt.xlabel('$Q(x,u)$')
        else:
            plt.ylabel('$(TV(x) - V(x))/V(x)$ (%)')
            plt.xlabel('$V(x)$')
        plt.ylim([min(-1., np.min(100. * (xalg_tv_of_x - xalg_v_of_x) / xalg_v_of_x) * 1.05),
                  min(100., np.max(100. * (xalg_tv_of_x - xalg_v_of_x) / xalg_v_of_x) * 1.05)])
        plt.savefig('output/' + self.s.name + '/xalg_BE_by_norm.pdf')
        plt.close()

    def output_convergence_results(self, data_in, nx, output_dir):
        """Save CSV files and graphs in PDF form regarding convergence of the algorithm

        :param data_in: Data structure containing quantities measured by iteration
        :param nx: Number of elements in XAlg
        :param output_dir: Output directory to save results to
        :return: nothing
        """
        (xalg_integral_results, xalg_bellman_results, xalg_cl_ub_results,
         ind_integral_results, ind_bellman_results, ind_cl_ub_results,
         lb_constr_count, convergence_j, iter_time, osp) = data_in

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        j_max = convergence_j if convergence_j is not None \
            else max([a[0] for a in xalg_integral_results])

        # Calculate DARE mean for plotting comparison
        if self.s.dare_sol is not None:
            ind_dare_mean = np.mean([0.5 * np.dot(np.dot(x, self.s.dare_sol), x)
                                 for x in self.v_integral_x_list])
            print "  DARE average cost across audit samples is %.2f" % ind_dare_mean
        else:
            ind_dare_mean = None

        # CSV of XAlg VF approximation mean
        xalg_integral_csv_array = np.array(xalg_integral_results)
        try:
            np.savetxt(output_dir + '/xalg_lb_M' + str(nx) + '.csv', xalg_integral_csv_array,
                       fmt='%d,%.5f', delimiter=',', header='Iteration,Mean LB',
                       comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/xalg_lb_M' + str(nx) + '.csv'
        # CSV of XAlg Bellman converge nce
        xalg_bellman_csv_array = np.array(xalg_bellman_results)
        try:
            np.savetxt(output_dir + '/xalg_bellman_error_M' + str(nx) + '.csv',
                       xalg_bellman_csv_array,
                       fmt='%d,%.5f,%.5f,%.5f,%.5f', delimiter=',',
                       header='Iteration,Mean TV(x)-V(x),Mean (TV(x)-V(x))/V(x),Max TV(x)-V(x),Max (TV(x)-V(x))/V(x)', comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/xalg_bellman_error_M' + str(nx) + '.csv'
        # CSV of XAlg closed-loop cost upper bound
        xalg_cl_ub_csv_array = np.array(xalg_cl_ub_results)
        try:
            np.savetxt(output_dir + '/xalg_cl_ub_M' + str(nx) + '.csv', xalg_cl_ub_csv_array,
                       fmt='%d,%.5f,%.5f', delimiter=',', header='Iteration,Mean UB,Mean UB u=0',
                       comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/xalg_cl_ub_M' + str(nx) + '.csv'

        ind_integral_csv_array = np.array(ind_integral_results)
        try:
            np.savetxt(output_dir + '/ind_lb_M' + str(nx) + '.csv', ind_integral_csv_array,
                       fmt='%d,%.5f', delimiter=',', header='Iteration,Mean LB', comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/xalg_lb_M' + str(nx) + '.csv'
        # CSV of independent Bellman convergence
        ind_bellman_csv_array = np.array(ind_bellman_results)
        try:
            np.savetxt(output_dir + '/ind_bellman_error_M' + str(nx) + '.csv', ind_bellman_csv_array,
                       fmt='%d,%.5f,%.5f,%.5f,%.5f', delimiter=',',
                       header='Iteration,Mean TV(x)-V(x),Mean (TV(x)-V(x))/V(x),Max TV(x)-V(x),Max (TV(x)-V(x))/V(x)', comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/ind_bellman_error_M' + str(nx) + '.csv'
        # CSV of independent closed-loop cost upper bound
        ind_cl_ub_csv_array = np.array(ind_cl_ub_results)
        try:
            np.savetxt(output_dir + '/ind_cl_ub_M' + str(nx) + '.csv', ind_cl_ub_csv_array,
                       fmt='%d,%.5f,%.5f', delimiter=',', header='Iteration,Mean UB,Mean UB u=0',
                       comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/ind_cl_ub_M' + str(nx) + '.csv'

        # Save timing info
        pd.DataFrame([('LB computation time', self.lb_computation_time),
                      ('Total iteration time', iter_time),
                      ('One-stage problems solved', osp)]).to_csv(output_dir + '/comp_time.csv',
                                                                  header=['Category', 'Time'],
                                                                  index=None, float_format='%.4f')

        # Plot LB and UB convergence
        plt.figure(figsize=(8, 12))
        plt.subplots(3, 1)
        plt.subplot(311)
        plt.plot([a[0] for a in xalg_integral_results], [a[1] for a in xalg_integral_results],
                 label='LB, $M=%d$ samples' % nx)
        plt.plot([a[0] for a in ind_integral_results], [a[1] for a in ind_integral_results],
                 label='LB, ind. samples')
        plt.plot([a[0] for a in xalg_cl_ub_results], [a[1] for a in xalg_cl_ub_results],
                 label='UB, $M=%d$ samples' % nx)
        plt.plot([a[0] for a in ind_cl_ub_results], [a[1] for a in ind_cl_ub_results],
                 label='UB, ind. samples')
        plt.plot([a[0] for a in xalg_cl_ub_results], [a[2] for a in xalg_cl_ub_results],
                 label='UB, u=0, $M=%d$ s' % nx)
        plt.plot([a[0] for a in ind_cl_ub_results], [a[2] for a in ind_cl_ub_results],
                 label='UB, u=0, ind. s')

        # if ind_dare_mean is not None:
        #     plt.plot(range(j_max+1), [ind_dare_mean] * (j_max+1), color='r',
        #              label='Unconstr. DARE sol')
        plt.ylabel('Mean LB and UB')
        plt.title('Mean LB and UB for V(x), %d sampled points' % nx)
        plt.ylim([-0.001, 2.0 * np.max([a[1] for a in xalg_integral_results])])
        plt.legend(loc=4)
        plt.grid()

        plt.subplot(312)
        plt.plot([a[0] for a in xalg_bellman_results],
                 [np.log10(a[1]) for a in xalg_bellman_results], label='Mean BE, $M=%d$ samples' % nx)
        plt.plot([a[0] for a in xalg_bellman_results],
                 [np.log10(a[2]) for a in xalg_bellman_results], label='Max BE, $M=%d$ samples' % nx)
        plt.plot([a[0] for a in ind_bellman_results],
                 [np.log10(a[1]) for a in ind_bellman_results], label='Mean BE, ind. samples')
        plt.plot([a[0] for a in ind_bellman_results],
                 [np.log10(a[2]) for a in ind_bellman_results], label='Max BE, ind. samples')
        if len(xalg_cl_ub_results) == len(xalg_bellman_results):
            plt.plot([a[0] for a in xalg_integral_results],
                     [np.log10(xalg_cl_ub_results[i][1] - a[1]) for i, a in enumerate(xalg_integral_results)],
                     label='Mean UB-LB, $M=%d$ samples' % nx)
        if len(ind_cl_ub_results) == len(ind_integral_results) and len(ind_integral_results) > 1:
            plt.plot([a[0] for a in ind_bellman_results],
                     [np.log10(ind_cl_ub_results[i][1] - a[1]) for i, a in enumerate(ind_integral_results)],
                     label='Mean UB-LB, ind. samples')
        plt.xlabel('Iteration number')
        plt.ylabel('$\log_{10}$ value')
        plt.title('Absolute convergence errors')
        # plt.ylim([-0.001, 10.0 * np.mean([a[1] for a in xalg_bellman_results])])
        plt.legend(loc=1)
        plt.grid()

        plt.subplot(313)
        # plt.plot([a[0] for a in xalg_bellman_results],
        #          [np.log10(a[1]) for a in xalg_bellman_results],
        #          label='Mean BE, $M=%d$ samples' % nx)
        # plt.plot([a[0] for a in xalg_bellman_results],
        #          [np.log10(a[2]) for a in xalg_bellman_results],
        #          label='Max BE, $M=%d$ samples' % nx)
        # plt.plot([a[0] for a in ind_bellman_results],
        #          [np.log10(a[1]) for a in ind_bellman_results], label='Mean BE, ind. samples')
        # plt.plot([a[0] for a in ind_bellman_results],
        #          [np.log10(a[2]) for a in ind_bellman_results], label='Max BE, ind. samples')
        if len(xalg_cl_ub_results) == len(xalg_bellman_results):
            plt.plot([a[0] for a in xalg_integral_results],
                     [np.log10(cl_ub[1]) for cl_ub in xalg_cl_ub_results],
                     label='Mean CL subopt., $M=%d$ samples' % nx)
            plt.plot([a[0] for a in xalg_integral_results],
                     [np.log10(cl_ub[2]) for cl_ub in xalg_cl_ub_results],
                     label='Mean CL subopt., u=0, $M=%d$ s.' % nx)
        if len(ind_cl_ub_results) == len(ind_integral_results) and len(ind_integral_results) > 1:
            plt.plot([a[0] for a in ind_bellman_results],
                     [np.log10(cl_ub[1]) for cl_ub in ind_cl_ub_results],
                     label='Mean CL cubopt., ind. samples')
            plt.plot([a[0] for a in ind_bellman_results],
                     [np.log10(cl_ub[2]) for cl_ub in ind_cl_ub_results],
                     label='Mean CL subopt., u=0, ind. s.')
        plt.xlabel('Iteration number')
        plt.ylabel('$\log_{10}$ value')
        plt.title('Relative convergence errors')
        # plt.ylim([-0.001, 10.0 * np.mean([a[1] for a in xalg_bellman_results])])
        plt.legend(loc=1)
        plt.grid()
        plt.savefig(output_dir + '/conv_M' + str(nx) + '.pdf')
        plt.close()

        # CSV LB constraint count
        lbc_csv_array = np.array([range(j_max), lb_constr_count[:j_max]])
        try:
            np.savetxt(output_dir + '/constr_count_M' + str(nx) + '.csv', lbc_csv_array.T,
                       fmt='%d,%d', delimiter=',', header='Iteration,No. of constraints', comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/constr_count_M' + str(nx) + '.csv'
        plt.figure()
        plt.plot(range(j_max), lb_constr_count[:j_max])
        plt.xlim([0, j_max])
        plt.xlabel('Iteration number')
        plt.ylabel('No. of constraints')
        plt.title('Number of $\\alpha$ lower-bounding constraints $g_j(\cdot)$')
        plt.grid()
        plt.savefig(output_dir + '/constr_count_M' + str(nx) + '.pdf')
        plt.close()

    def measure_many_cl_upper_bounds(self, x_list, n_steps, j, no_input=False):
        """Measure upper bound on optimal value function at each entry in a list of x_list points,
        by simulating the system for n_steps

        :param x_list: List of n-element vectors to simulate the system from
        :param n_steps: Integer number of points
        :param j: Iteration number (acts differently after 0 iterations to save time)
        :param no_input: Boolean. Set input u = 0 for upper bound calculation
        :return: List of upper bounds, indices for which UBs were finite
        """
        nx = len(x_list)
        ubs = np.inf * np.ones((nx,), dtype=float)
        for i, x in enumerate(x_list):
            ubs[i] = self.s.simulate(x0=x, fa_in=self, n_steps=n_steps, save_plot=False,
                                     return_ub=True, no_input=no_input)
            if j == 0 and ubs[i] == np.inf:  # Don't expect finite upper bounds after 0 iterations
                break  # Break to avoid simulating closed-loop system with no VFA in place
        finite_indices = np.where(ubs < np.inf)[0]
        n_finite = len(finite_indices)
        if n_finite > 0:
            print "    CL upper bound was finite for %d of %d samples (%.1f%%)." % (n_finite, nx,
                                                                                100 * float(n_finite)/nx)
        else:
            print "    No valid upper bounds found!"
        return ubs, finite_indices

    def measure_ind_bellman_gap(self):
        """Returns mean Bellman Error (BE), TV(x) - V(x), for independently sampled points x

        :return: Tuple of scalars (Mean BE, Mean relative BE, Max BE, Max relative BE)
        """
        n_s = self.v_n_ind_samples
        bellman_gap_array = np.zeros((n_s,), dtype=float)
        rel_bellman_gap_array = np.zeros((n_s,), dtype=float)
        bad_gap_count = 0
        t1 = time.time()
        for i in range(n_s):
            x_i = self.v_integral_x_list[i]
            vx = self.eval_vfa(x_i)
            _, tvx, _ = self.solve_for_xhat(x_i, extract_constr=False)

            # Make sure Bellman gap is not negative
            tol = 1e-2
            if tvx - vx <= -tol:
                bad_gap_count += 1
                print "  Warning: Negative Bellman gap for x sample #%d: %.4f!" % (i+1, tvx-vx)

            bellman_gap_array[i] = tvx - vx
            try:
                rel_bellman_gap_array[i] = (tvx - vx)/vx
            except:
                pass

        if bad_gap_count > 0:
            print "  %d / %d instances (%.1f%%) of Bellman gap were negative!" % \
                  (bad_gap_count, n_s, 100 * float(bad_gap_count) / n_s)

        t2 = time.time()

        self.b_gap_eval_time += t2 - t1

        return np.mean(bellman_gap_array), np.mean(rel_bellman_gap_array), \
            np.max(bellman_gap_array), np.max(rel_bellman_gap_array)

    def approximate(self, strategy, audit, outputs):
        """
        Create an iterative approximation of the value function for the system self.s

        :param strategy: Dictionary of parameters describing the solution strategy
        :param audit: Dict of parameters describing how progress should be tracked during solution
        :param outputs: Dict of parameters determining outputs produced
        :return: Convergence data structure
        """

        j_max, n_x_points = strategy['max_iter'], strategy['n_x_points']
        rand_seed = strategy['rand_seed']
        focus_on_origin = strategy['focus_on_origin']
        sol_strategy = strategy['sol_strategy']
        conv_tol, stop_on_conv = strategy['conv_tol'], strategy['stop_on_convergence']
        remove_red, removal_freq = strategy['remove_redundant'], strategy['removal_freq']
        removal_res = int(strategy['removal_resolution'])
        consolidate_constraints = strategy['consolidate_constraints']
        consolidation_freq = strategy['consolidation_freq']
        value_function_limit = strategy['value_function_limit']

        eval_ub, eval_ub_freq = audit['eval_ub'], audit['eval_ub_freq']
        eval_ub_final = audit['eval_ub_final']
        eval_bellman, eval_bellman_freq = audit['eval_bellman'], audit['eval_bellman_freq']
        eval_integral, eval_integral_freq = audit['eval_integral'], audit['eval_integral_freq']
        eval_convergence = audit['eval_convergence']
        eval_convergence_freq = audit['eval_convergence_freq']
        n_ind_x_points = audit['n_independent_x']

        cl_plot_j, cl_plot_final = outputs['cl_plot_j'], outputs['cl_plot_final']
        cl_plot_freq, cl_plot_n_steps = outputs['cl_plot_freq'], outputs['cl_plot_n_steps']
        vfa_plot_j, vfa_plot_final = outputs['vfa_plot_j'], outputs['vfa_plot_final']
        vfa_plot_freq = outputs['vfa_plot_freq']
        policy_plot_j, policy_plot_final = outputs['policy_plot_j'], outputs['policy_plot_final']
        policy_plot_freq = outputs['policy_plot_freq']
        suppress_outputs = outputs['suppress all']

        # Create M samples for VF fitting
        np.random.seed(rand_seed)
        distr = 'normal'
        if focus_on_origin:
            if distr == 'normal':
                xalg_list = [self.s.state_mean + np.dot(np.sqrt(self.s.state_var),
                                                        np.random.randn(self.n))
                             for k in range(n_x_points / 2)] + \
                            [self.s.state_mean + np.dot(np.sqrt(self.s.state_var) / 5.,
                                                        np.random.randn(self.n))
                             for k in range(n_x_points / 2)]
            elif distr == 'laplace':
                laplace_rv = [laplace(self.s.state_mean[m], np.diag(np.sqrt(self.s.state_var))[m])
                              for m in range(self.n)]
                laplace_rv2 = [laplace(self.s.state_mean[m], np.diag(np.sqrt(self.s.state_var))[m] / 5.)
                               for m in range(self.n)]
                xalg_list = [np.array([laplace_rv[m].rvs() for m in range(self.n)])
                             for k in range(n_x_points / 2)] + \
                            [np.array([laplace_rv2[m].rvs() for m in range(self.n)])
                             for k in range(n_x_points / 2)]
        else:
            if distr == 'normal':
                xalg_list = [self.s.state_mean + np.dot(np.sqrt(self.s.state_var),
                                                        np.random.randn(self.n))
                             for k in range(n_x_points)]
            elif distr == 'laplace':
                laplace_rv = [laplace(self.s.state_mean[m], np.diag(np.sqrt(self.s.state_var))[m])
                              for m in range(self.n)]
                xalg_list = [np.array([laplace_rv[m].rvs() for m in range(self.n)])
                             for k in range(n_x_points)]

        self.create_audit_samples(n_ind_x_points, seed=rand_seed)  # Create ind't audit samples

        gamma = self.s.gamma

        self.total_solver_time = 0.
        self.b_gap_eval_time = 0.
        self.v_integral_eval_time = 0.
        self.lb_computation_time = 0.
        t1 = time.time()

        # Results for the states visited by the algorithm
        lb_constr_count = np.zeros((j_max,), dtype=int)
        xalg_integral_results = []
        xalg_bellman_results = []
        xalg_cl_ub_results = []

        # Results for independently-generated states
        ind_integral_results = []
        ind_bellman_results = []
        ind_cl_ub_results = []

        # Convergence data
        samples_converged = False
        convergence_j = None
        self.x_visited = []

        for j in range(j_max):

            print "Iteration %d" % j

            lb_constr_count[j] = len(self.alpha_c_in_model)
            if consolidate_constraints and divmod(j, consolidation_freq)[1] == 0:
                self.consolidate_bounds()
            # Eval Bellman error convergence
            if eval_convergence and divmod(j, eval_convergence_freq)[1] == 0:
                print "  Measuring VF integral and Bellman error for M=%d elements of XAlg..." % n_x_points
                if j > 0 and sol_strategy == 'biggest_gap':
                    old_largest_bellman_error = xalg_tv_of_x[k] - xalg_v_of_x[k]
                else:
                    old_largest_bellman_error = 0.
                xalg_v_of_x, xalg_tv_of_x = [], []
                for x in xalg_list:
                    xalg_v_of_x.append(self.eval_vfa(x))
                    xalg_tv_of_x.append(self.solve_for_xhat(x, extract_constr=False, iter_no=j)[1])
                    if xalg_tv_of_x[-1] < xalg_v_of_x[-1] - 1e-4:
                        print "  Negative BE found at element %d of XAlg: %.5f" % \
                              (len(xalg_v_of_x), xalg_tv_of_x[-1] - xalg_v_of_x[-1])
                xalg_v_of_x = np.array(xalg_v_of_x)
                xalg_tv_of_x = np.array(xalg_tv_of_x)
                xalg_integral_results.append((copy.copy(j), np.mean(xalg_v_of_x)))
                xalg_bellman_results.append((copy.copy(j),
                                             np.mean(xalg_tv_of_x - xalg_v_of_x),
                                             np.mean(
                                                 (xalg_tv_of_x - xalg_v_of_x) / xalg_v_of_x),
                                             np.max(xalg_tv_of_x - xalg_v_of_x),
                                             np.max(
                                                 (xalg_tv_of_x - xalg_v_of_x) / xalg_v_of_x)))
            # Evaluate value function upper bound
            x_data, x_upper = None, None
            if eval_ub and divmod(j, eval_ub_freq)[1] == 0:
                print "  Evaluating closed-loop UB for M=%d elements of XAlg..." % n_x_points
                if not (eval_convergence and divmod(j, eval_convergence_freq)[1] == 0):
                    # Generate V(x) for points in XAlg if not done above
                    xalg_v_of_x = np.array([self.eval_vfa(x) for x in xalg_list])
                xalg_ubs, xalg_fi = self.measure_many_cl_upper_bounds(xalg_list,
                                                                      n_steps=cl_plot_n_steps,
                                                                      j=j)
                xalg_ubs_nou, xalg_fi_nou = self.measure_many_cl_upper_bounds(xalg_list,
                                                                              n_steps=cl_plot_n_steps,
                                                                              j=j,
                                                                              no_input=True)
                if len(xalg_fi) > 0:
                    xalg_cl_ub_results.append((copy.copy(j),
                                               np.sum(xalg_ubs[xalg_fi]) /
                                               np.sum(xalg_v_of_x[xalg_fi]) - 1.0,
                                               np.sum(xalg_ubs_nou[xalg_fi_nou]) /
                                               np.sum(xalg_v_of_x[xalg_fi_nou]) - 1.0))
                else:
                    xalg_cl_ub_results.append((copy.copy(j), np.inf, np.inf))
            # Measure integral of value function approximation over x samples
            if eval_integral and divmod(j, eval_integral_freq)[1] == 0:
                print "  Measuring VF integral for %d independent samples..." % n_ind_x_points
                ind_v_of_x = np.array([self.eval_vfa(x) for x in self.v_integral_x_list])
                t1 = time.time()
                ind_integral_results.append((copy.copy(j), np.mean(ind_v_of_x)))
                t2 = time.time()
                self.v_integral_eval_time += t2 - t1
            # Measure Bellman gap TV(x) - V(x) over x samples
            if eval_bellman and divmod(j, eval_bellman_freq)[1] == 0:
                print "  Measuring Bellman error for %d independent samples..." % n_ind_x_points
                avg_gap, avg_rel_gap, max_gap, max_rel_gap = self.measure_ind_bellman_gap()
                ind_bellman_results.append((copy.copy(j), avg_gap, avg_rel_gap, max_gap, max_rel_gap))
                print "  Measuring closed-loop UB for %d independent samples..." % n_ind_x_points
                if not (eval_integral and divmod(j, eval_integral_freq)[1] == 0):
                    ind_v_of_x = np.array([self.eval_vfa(x) for x in self.v_integral_x_list])
                ind_ubs, ind_fi = self.measure_many_cl_upper_bounds(self.v_integral_x_list,
                                                                    n_steps=cl_plot_n_steps,
                                                                    j=j,
                                                                    no_input=False)
                ind_ubs_nou, ind_fi_nou = self.measure_many_cl_upper_bounds(self.v_integral_x_list,
                                                                            n_steps=cl_plot_n_steps,
                                                                            j=j,
                                                                            no_input=True)
                if len(ind_fi) > 0:
                    ind_cl_ub_results.append((copy.copy(j),
                                              np.sum(ind_ubs[ind_fi]) /
                                              np.sum(ind_v_of_x[ind_fi]) - 1.0,
                                              np.sum(ind_ubs_nou[ind_fi_nou]) /
                                              np.sum(ind_v_of_x[ind_fi_nou]) - 1.0))
                else:
                    ind_cl_ub_results.append((copy.copy(j), np.inf, np.inf))
            # Remove redundant lower bounding functions (based on samples)
            if remove_red and divmod(j, removal_freq)[1] == 0:
                print "  Removing redundant lower-bounding constraints..."
                self.remove_redundant_lbs(n_samples=removal_res, threshold=0.00, sigma_in=np.pi)
            # Plot closed-loop trajectories
            if cl_plot_j and divmod(j, cl_plot_freq)[1] == 0 and not suppress_outputs:
                print "  Plotting a closed-loop trajectory..."
                # x0 = np.array([1., 0., -0.17453, 0.])
                x0 = np.ones((self.n,), dtype=float)
                self.s.simulate(x0, self, n_steps=cl_plot_n_steps, iter_no=j, save_plot=True)
                # Save timing info
                pd.DataFrame([('LB computation time', self.lb_computation_time)]).to_csv(
                    'output/' + self.s.name + '/comp_time_%d.csv' % j,
                    header=['Category', 'Time'],
                    index=None, float_format='%.4f')
            # Plot VF approximation
            if vfa_plot_j and divmod(j, vfa_plot_freq)[1] == 0 and not suppress_outputs:
                print "  Plotting value function approximation..."
                self.plot_vfa(iter_no=j, save=True, output_dir='output/' + self.s.name)
            # Plot policy
            if policy_plot_j and divmod(j, policy_plot_freq)[1] == 0 and not suppress_outputs:
                print "  Plotting control policy..."
                self.plot_policy(iter_no=j, save=True, output_dir='output/' + self.s.name)

            # Pick the next x
            if sol_strategy == 'random':
                # Choose an x sample index at random and generate a new LB constraint from it
                vfa_too_large = True
                while vfa_too_large:
                    # Keep testing different samples k until one with an open gap found
                    k = np.random.randint(0, n_x_points)
                    x_picked = xalg_list[k]
                    if self.eval_vfa(x_picked) <= value_function_limit:
                        vfa_too_large = False

                # Tests on a point x_test for debug purposes
                # if j > 55:
                #     # x_test = np.array([-0.754, -0.060, -1.944, 0.030])
                #     x_test = np.array([0.05036, 0.05092, -0.35502, -0.09855])
                #     # v_of_x_test = self.eval_vfa(x_test)
                #     # x_test_bd, x_test_sol = self.solve_for_xhat(x_test, iter_no=j, extract_constr=True)
                #     # tv_of_x_test = np.sum(x_test_sol[2]) + self.s.gamma * x_test_sol[3]
                #     # print "  V(x_test): %.5f, TV(x_test): %.5f, BE: %.5f" % \
                #     #       (v_of_x_test, tv_of_x_test, tv_of_x_test - v_of_x_test)
                #     # jd_of_x_test = x_test_bd.eval_at(x_test)
                #     # print "  J_D(x_test): %.5f" % jd_of_x_test
                #     # print "  g_i(x_test): %.5f" % x_test_bd.eval_at(x_test)
                #     print "  new_constr(x_picked): %.5f" % new_constr.eval_at(x_picked)
                #     print "  new_constr(x_test): %.5f" % new_constr.eval_at(x_test)

            elif sol_strategy == 'biggest_gap':
                old_k = copy.copy(k)
                if not xalg_bellman_results[-1][0] == j:
                    # Evaluate Bellman error at all points in XAlg if conv. check not done at it. j
                    old_largest_bellman_error = xalg_tv_of_x[k] - xalg_v_of_x[k]
                    for i, x in enumerate(xalg_list):
                        xalg_v_of_x[i] = self.eval_vfa(x)
                        _, opt_cost, _ = self.solve_for_xhat(x, iter_no=j, extract_constr=False)
                        xalg_tv_of_x[i] = opt_cost

                k = np.argsort(xalg_tv_of_x - xalg_v_of_x)[-1]  # Pick largest Bellman error
                print "  Largest BE is at sample %d/%d: %.5f" % (k+1, n_x_points,
                                                                 xalg_tv_of_x[k] - xalg_v_of_x[k])
                x_picked = xalg_list[k]
                if k == old_k and old_largest_bellman_error == xalg_tv_of_x[k] - xalg_v_of_x[k]:
                    print "Index and value of largest Bellman error didn't change!"
                    # raise SystemExit()

            elif sol_strategy == 'parallel':
                print "parallel not implemented!"
                raise SystemExit()
                # Generate a new LB constraint for each x sample
                constrs_to_add = []
                for i, x in enumerate(xalg_list):
                    new_constr, sol = self.solve_for_xhat(x, iter_no=j, extract_constr=True)
                    u, xplus, beta, alpha = sol
                    xalg_v_of_x[i] = self.eval_vfa(x)
                    xalg_tv_of_x[i] = np.sum(beta) + gamma * alpha
                    summed_ub_1s[j] += np.sum(beta) + gamma * alpha
                    if xalg_tv_of_x[i] - xalg_v_of_x[i] > new_constr_tol:
                        constrs_to_add.append(copy.deepcopy(new_constr))
                    lb_ub_results.append((j, xalg_integral_results[j], summed_ub_1s[j]))

                # Modify the VF approximation
                for c in constrs_to_add:
                    if not c.dodgy:
                        self.add_alpha_constraint(c)
                    else:
                        print "  New LB constraint not reliable. Not adding to model."
            else:
                print "Unrecognized solution strategy:", sol_strategy
                raise SystemExit()

            # Extract new lower-bounding constraint for the x sample picked
            print "  x_%d = [" % (k + 1) + ", ".join(["%.3f" % x_el for x_el in x_picked]) + "]"
            t_start_lb = time.time()
            new_constr, sol = self.solve_for_xhat(x_picked, iter_no=j, extract_constr=True)
            t_end_lb = time.time()
            self.lb_computation_time += t_end_lb - t_start_lb
            self.x_visited.append(x_picked)
            u, xplus, beta, alpha = sol
            primal_optimal_value = np.sum(beta) + gamma * alpha
            vfa_x_picked = self.eval_vfa(x_picked)
            gap_found = primal_optimal_value - vfa_x_picked

            # Modify the VF approximation
            if gap_found >= 1e-4:
                if new_constr.dodgy:
                    print "  New LB constr. at sample %d unreliable. Not adding to model." % k
                elif new_constr.worthless:
                    print "  New LB constr. doesn't increase VFA due to duality gap. Not added."
                else:
                    self.add_alpha_constraint(new_constr)
                    # new_constr.plot_function(output_dir='output/'+self.s.name, iter_no=j)
                    print "  New LB constraint added to model."
            elif gap_found <= -1e-3:
                print "  Negative Bellman error found for sample %d: %.5f!" % (k + 1, gap_found)
                print "x_picked:", x_picked
                # print "u, xplus, beta, alpha:", u, xplus, beta, alpha
                print "Alpha constraints:"
                for c in self.alpha_c_in_model:
                    print "Constraint %d:" % c.id + " V(x_picked) - g_i(x_picked):", \
                        vfa_x_picked - c.eval_at(x_picked)
                    if vfa_x_picked - c.eval_at(x_picked) == 0:
                        print "Removing constraint from the model."
                        self.remove_alpha_constraint(c)
            else:
                print "  No new constraint added for sample %d: BE of %.5f reached." % \
                      (k + 1, gap_found)

            if eval_convergence and divmod(j, eval_convergence_freq)[1] == 0 and \
                    xalg_bellman_results[-1][2] <= conv_tol and not samples_converged:
                samples_converged, convergence_j = True, j
                print "  Bellman convergence detected to within", str(conv_tol)
                if stop_on_conv:
                    break

        if eval_convergence and convergence_j is None:
            print "Completed %d iterations without convergence." % j_max
            print "Tolerance reached was %.5f." % xalg_bellman_results[-1][1]
            self.final_j = j_max
        elif eval_convergence and convergence_j is not None and stop_on_conv:
            print "Converged after %d iterations; terminated early." % convergence_j
            self.final_j = convergence_j
        elif convergence_j is not None:
            print "Converged after %d iterations; completed all %d anyway." % (convergence_j, j_max)
            self.final_j = j_max
        else:
            self.final_j = j_max

        iter_time = time.time() - t1
        print "Iterations complete in %.1f seconds. Solved %d 1-stage problems." % \
              (iter_time, self.one_stage_problems_solved)

        # Final measurements of convergence
        print "Measuring final value function characteristics:"
        print "  VF integral for M=%d elements of XAlg..." % n_x_points
        xalg_v_of_x = np.array([self.eval_vfa(x) for x in xalg_list])
        xalg_integral_results.append((self.final_j, np.mean(xalg_v_of_x)))
        print "  Bellman error for M=%d elements of XAlg..." % n_x_points
        xalg_tv_of_x = np.array([self.solve_for_xhat(x, extract_constr=False, iter_no=j)[1]
                                 for x in xalg_list])
        xalg_negative_bellman_list = []
        xalg_negative_bellman_errors = False
        for i, vx in enumerate(xalg_v_of_x):
            if xalg_tv_of_x[i] < vx - 1e-2:
                xalg_negative_bellman_list.append(i + 1)
                xalg_negative_bellman_errors = True
        if xalg_negative_bellman_errors:
            print "Negative Bellman errors found at samples", xalg_negative_bellman_list

        print "  VF integral for %d independent samples..." % n_ind_x_points
        t1 = time.time()
        ind_v_of_x = np.array([self.eval_vfa(x) for x in self.v_integral_x_list])
        t2 = time.time()
        self.v_integral_eval_time += t2 - t1
        ind_integral_results.append((self.final_j, np.mean(ind_v_of_x)))
        print "  Bellman error for %d independent samples..." % n_ind_x_points
        avg_gap, avg_rel_gap, max_gap, max_rel_gap = self.measure_ind_bellman_gap()
        ind_bellman_results.append((self.final_j, avg_gap, avg_rel_gap, max_gap, max_rel_gap))
        if eval_ub_final:
            print "  Closed-loop UB for M=%d elements of XAlg..." % n_x_points
            xalg_ubs, xalg_fi = self.measure_many_cl_upper_bounds(xalg_list,
                                                                  n_steps=cl_plot_n_steps,
                                                                  j=self.final_j)
            xalg_ubs_nou, xalg_fi_nou = self.measure_many_cl_upper_bounds(xalg_list,
                                                                          n_steps=cl_plot_n_steps,
                                                                          j=self.final_j,
                                                                          no_input=True)
            if len(xalg_fi) > 0:
                xalg_cl_ub_results.append((copy.copy(self.final_j),
                                           np.sum(xalg_ubs[xalg_fi]) / np.sum(xalg_v_of_x[xalg_fi]) - 1.0,
                                           np.sum(xalg_ubs_nou[xalg_fi_nou]) / np.sum(xalg_v_of_x[xalg_fi_nou]) - 1.0))
            else:
                xalg_cl_ub_results.append((copy.copy(self.final_j), np.inf, np.inf))
            print "  Closed-loop UB for %d independent samples..." % n_ind_x_points
            ind_ubs, ind_fi = self.measure_many_cl_upper_bounds(self.v_integral_x_list,
                                                                n_steps=cl_plot_n_steps,
                                                                j=j)
            ind_ubs_nou, ind_fi_nou = self.measure_many_cl_upper_bounds(self.v_integral_x_list,
                                                                        n_steps=cl_plot_n_steps,
                                                                        j=j,
                                                                        no_input=True)
            if len(ind_fi) > 0:
                ind_cl_ub_results.append((self.final_j,
                                          np.sum(ind_ubs[ind_fi]) / np.sum(ind_v_of_x[ind_fi]) - 1.0,
                                          np.sum(ind_ubs_nou[ind_fi_nou]) / np.sum(ind_v_of_x[ind_fi_nou]) - 1.0))
            else:
                ind_cl_ub_results.append((self.final_j, np.inf, np.inf))

        self.plot_xalg_be(xalg_v_of_x, xalg_tv_of_x)

        xalg_bellman_results.append((self.final_j,
                                     np.mean(xalg_tv_of_x - xalg_v_of_x),
                                     np.mean((xalg_tv_of_x - xalg_v_of_x)/xalg_v_of_x),
                                     np.max(xalg_tv_of_x - xalg_v_of_x),
                                     np.max((xalg_tv_of_x - xalg_v_of_x)/xalg_v_of_x)))

        # Output convergence results
        convergence_data = (xalg_integral_results, xalg_bellman_results, xalg_cl_ub_results,
                            ind_integral_results, ind_bellman_results, ind_cl_ub_results,
                            lb_constr_count, self.final_j, iter_time,
                            copy.copy(self.one_stage_problems_solved))
        self.output_convergence_results(convergence_data, n_x_points, 'output/' + self.s.name)

        # Plot closed-loop trajectories
        if cl_plot_final and not suppress_outputs:
            print "  Plotting final closed-loop trajectory..."
            x0 = np.array([1., 0., -0.17453, 0.])
            # x0 = np.ones((self.n,), dtype=float)
            self.s.simulate(x0, self, cl_plot_n_steps, iter_no=self.final_j, save_plot=True)
        # Plot VF approximation
        if vfa_plot_final and not suppress_outputs:
            x_data, x_upper = None, None
            print "  Plotting final value function approximation..."
            self.plot_vfa(iter_no=self.final_j, save=True, output_dir='output/' + self.s.name)
        # Plot policy
        if policy_plot_final and not suppress_outputs:
            print "  Plotting final control policy..."
            self.plot_policy(iter_no=self.final_j, save=True, output_dir='output/' + self.s.name)

        total_time = time.time() - t1

        print "Done in %.1f seconds, of which:" % total_time
        print "  %.1f s spent computing lower bounds," % self.lb_computation_time
        print "  %.1f s spent measuring V integral," % self.v_integral_eval_time
        print "  %.1f s spent auditing Bellman gap," % self.b_gap_eval_time
        print "  and %.1f s elsewhere." % (total_time - self.v_integral_eval_time -
                                           self.b_gap_eval_time - self.lb_computation_time)

        return convergence_data
