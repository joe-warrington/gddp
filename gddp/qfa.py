import numpy as np
import time
import gurobipy as gu
import copy
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.stats import laplace
import os
from gddp import QConstraint
from vfa import VFApproximator
np.seterr(divide='ignore')

default_qf_approx_strategy = {'max_iter': 100,
                              'n_z_points': 100, 'rand_seed': 1,
                              'on_policy': True,
                              'q_function_limit': 10000,
                              'sol_strategy': 'random', 'conv_tol': 1e-4,
                              'stop_on_convergence': False,
                              'remove_redundant': False, 'removal_freq': 10,
                              'removal_resolution': 100000,
                              'focus_on_origin': False, 'consolidate_constraints': False,
                              'consolidation_freq': False,
                              'value_function_limit': 10000}

default_qf_approx_outputs = {'cl_plot_j': False, 'cl_plot_freq': 20, 'cl_plot_final': False,
                             'cl_plot_n_steps': 50,
                             'qfa_plot_j': False, 'qfa_plot_freq': 1,
                             'qfa_plot_final': True,
                             'policy_plot_j': False, 'policy_plot_freq': 5,
                             'policy_plot_final': True,
                             'suppress all': False}

# Plotting ranges for 2D value functions
default_x1_min, default_x1_max, default_x1_res = -3, 3, 0.1
default_u1_min, default_u1_max, default_u1_res = -3, 3, 0.1
default_n1m1_vmin, default_n1m1_vmax = None, None

# Default uniform distribution properties, per dimension
x_unif_lower_default, x_unif_upper_default = [-3.], [3.]
u_unif_lower_default, u_unif_upper_default = [-1.], [1.]


class QFApproximator(VFApproximator):

    def __init__(self, system, solver='gurobi'):
        super(QFApproximator, self).__init__(system, solver)

        self.default_strategy = default_qf_approx_strategy
        self.default_outputs = default_qf_approx_outputs

        assert len(self.s.phi_i_x_quad) == 1
        assert len(self.s.phi_i_x_lin) == 1
        assert len(self.s.ri) == 1
        assert len(self.s.Ri) == 1
        assert len(self.s.ti) == 1
        assert len(self.s.li) == 1

        # Placeholders for one-stage problem model
        self.mod_x, self.mod_c, self.mod_c_c, self.mod_alpha_c = None, None, None, None
        # Placeholders for policy opt. model
        self.pmod, self.pmod_x, self.pmod_c = None, None, None
        self.pmod_c_c, self.pmod_alpha_c = None, None
        self.xalg_list = None
        self.ualg_list = None
        self.on_policy = None
        self.u_visited = None
        self.create_qf_policy_model()

    def create_qfa_model(self):
        if self.solver == 'gurobi':
            self.mod = gu.Model("QFA")
            self.mod.params.outputflag = 0
            self.mod.params.qcpdual = 0
            self.mod.params.dualreductions = 1
            self.mod.params.optimalitytol = 1e-6
            self.mod.params.feasibilitytol = 1e-6
            self.mod.params.barconvtol = 1e-6
            self.mod.params.barqcpconvtol = 1e-6
            self.mod.params.numericfocus = 0
            self.mod_x = []
            for i in range(self.n):
                self.mod_x.append(self.mod.addVar(name="xplus_%d" % (i + 1), lb=-1e100, obj=0.))
            for i in range(self.m):
                self.mod_x.append(self.mod.addVar(name="uplus_%d" % (i + 1), lb=-1e100, obj=0.))
            self.mod_x.append(self.mod.addVar(name="alpha", lb=-1e100, obj=0.))
            self.mod.update()

            # Placeholder for state x when building constraints (use the origin)
            # Constraints will be updated for particular xhat when optimizer is called
            xhat = np.zeros((self.n,), dtype=float)
            uhat = np.zeros((self.m,), dtype=float)

            # Add dynamics constraint
            fxx = self.s.f_x_x(xhat)
            fux = self.s.f_u_x(xhat)
            self.mod_c = []
            for i in range(self.n):
                self.mod_c.append(self.mod.addConstr(self.mod_x[i] == fxx[i] + np.dot(fux[i, :],
                                                                                      uhat),
                                                     name="dyn_%d" % (i+1)))

            # Add state-input constraints
            self.mod_c_c = []
            for i in range(self.s.D.shape[0]):
                self.mod_c_c.append(self.mod.addConstr(
                    gu.LinExpr([(self.s.D[i, p], self.mod_x[p]) for p in range(self.n)]) +
                    gu.LinExpr([(self.s.E[i, p], self.mod_x[self.n + p]) for p in range(self.m)])
                    <= self.s.h_x(np.zeros((self.n, ), dtype=float))[i],  name="sic_%d" % (i+1)))

            # Add (discounted) stage cost present in all q-lower-bounds gamma * l(xplus, uplus)
            const = self.s.phi_i_x_const[0]
            xle = gu.LinExpr([(self.s.phi_i_x_lin[0][p], self.mod_x[p]) for p in range(self.n)])
            xqe = gu.QuadExpr()
            for p in range(self.n):
                xqe += 0.5 * self.mod_x[p] * gu.LinExpr([(self.s.phi_i_x_quad[0][p, q],
                                                          self.mod_x[q])
                                                         for q in range(self.n)])
            ule = gu.LinExpr([(self.s.ri[0][p], self.mod_x[self.n + p]) for p in range(self.m)])
            uqe = gu.QuadExpr()
            for p in range(self.m):
                uqe += 0.5 * self.mod_x[self.n + p] * gu.LinExpr([(self.s.Ri[0][p, q],
                                                                   self.mod_x[self.n + q])
                                                                  for q in range(self.m)])
            self.mod.setObjective(self.s.gamma * (const + xle + xqe + ule + uqe + self.mod_x[-1]))

            # Create model for evaluating control policy
            self.create_qf_policy_model()

            # Add initial alpha epigraph constraint
            initial_lb = QConstraint(self.n, self.m, 0, 0., None, None, None, None)
            self.mod_alpha_c, self.pmod_alpha_c = [], []
            self.add_alpha_constraint(initial_lb)

            # Update model to reflect constraints
            self.mod.update()
        else:
            print "Solver " + self.solver + " not implemented!"
            raise SystemExit()

    def create_qf_policy_model(self):
        if self.solver == 'gurobi':
            self.pmod = gu.Model("QFA")
            self.pmod.params.outputflag = 0
            # self.pmod.params.qcpdual = 0
            # self.pmod.params.dualreductions = 1
            # self.pmod.params.optimalitytol = 1e-6
            # self.pmod.params.feasibilitytol = 1e-6
            # self.pmod.params.barconvtol = 1e-6
            # self.pmod.params.barqcpconvtol = 1e-6
            # self.pmod.params.numericfocus = 0
            self.pmod_x = []
            for i in range(self.n):
                self.pmod_x.append(self.pmod.addVar(name="x_%d" % (i + 1), lb=-1e100, obj=0.))
            for i in range(self.m):
                self.pmod_x.append(self.pmod.addVar(name="u_%d" % (i + 1), lb=-1e100, obj=0.))
            self.pmod_x.append(self.pmod.addVar(name="alpha", lb=-1e100, obj=0.))
            self.pmod.update()

            # Placeholder for state x when building constraints (use the origin)
            # Constraints will be updated for particular xhat when optimizer is called
            xhat = np.zeros((self.n,), dtype=float)

            # Fix x variable to xhat
            self.pmod_c = []
            for i in range(self.n):
                self.pmod_c.append(self.pmod.addConstr(self.pmod_x[i] == xhat[i],
                                                       name="fix_x_%d" % (i+1)))

            # Add state-input constraints
            for i in range(self.s.D.shape[0]):
                self.pmod.addConstr(gu.LinExpr([(self.s.D[i, p], self.pmod_x[p]) for p in range(self.n)]) +
                                    gu.LinExpr([(self.s.E[i, p], self.pmod_x[self.n + p])
                                                for p in range(self.m)])
                                    <= self.s.h_x(np.zeros((self.n, ), dtype=float))[i],
                                    name="sic_%d" % (i+1))

            # Add stage cost present in all q-lower-bounds gamma * l(xplus, uplus)
            const = self.s.phi_i_x_const[0]
            xle = gu.LinExpr([(self.s.phi_i_x_lin[0][p], self.pmod_x[p]) for p in range(self.n)])
            xqe = gu.QuadExpr()
            for p in range(self.n):
                xqe += 0.5 * self.pmod_x[p] * gu.LinExpr([(self.s.phi_i_x_quad[0][p, q], self.pmod_x[q])
                                                for q in range(self.n)])
            ule = gu.LinExpr([(self.s.ri[0][p], self.pmod_x[self.n + p]) for p in range(self.m)])
            uqe = gu.QuadExpr()
            for p in range(self.m):
                uqe += 0.5 * self.pmod_x[self.n + p] * gu.LinExpr([(self.s.Ri[0][p, q], self.pmod_x[self.n + q])
                                                         for q in range(self.m)])
            self.pmod.setObjective(const + xle + xqe + ule + uqe + self.pmod_x[-1])

            # Update model to reflect constraints
            self.pmod.update()
        else:
            print "Solver " + self.solver + " not implemented!"
            raise SystemExit()

    def update_constraints_for_xhatuhat(self, xhat, uhat, xplus_constraint):
        """Update constraint of optimization model to account for the current state, which enters
        into the one-stage optimization as a fixed parameter.

        :param xhat: Fixed state parameter for the problem
        :param uhat: Fixed input parameter for the problem
        :param xplus_constraint: Add a constraint that the successor state should take a particular
        value
        :return: nothing
        """

        fxx = self.s.f_x_x(xhat)
        fux = self.s.f_u_x(xhat)
        if self.solver == 'gurobi':
            # Update dynamics
            for i in range(self.n):
                self.mod_c[i].setAttr('RHS', fxx[i] + np.dot(fux[i, :], uhat))
            # No need to update state-input constraints: invariant to xhat, uhat
            # Update cost to include stage cost incurred at (xhat, uhat)
            self.mod.objcon = self.s.quad_stage_cost(xhat, uhat)
            self.mod.update()
            if xplus_constraint is not None:
                print "xplus_constraint not implemented for QF approximator. Exiting."
                raise SystemExit()
                # Remove any existing x_plus constraints
                # c = [l for l in self.mod.getConstrs() if "forced_xplus_" in l.ConstrName]
                # for l in c:
                #     self.mod.remove(l)
                # assert xplus_constraint.shape == (self.n,), "Forced x_plus has wrong dimension"
                #
                # # n_subspace = xplus_constraint.shape[1]
                # # print "n_subspace:", n_subspace
                # for j in range(self.n):
                #     # b_power_A = np.dot(xplus_constraint[:, j],
                #     #                    -np.linalg.matrix_power(self.s.A_mat, self.n - n_subspace))
                #     # print "    ", j, b_power_A
                #     # self.mod.addConstr(gu.LinExpr([(b_power_A[i], x[self.m + i])
                #     #                                for i in range(self.n)]) <= 1e-2,
                #     #                    name='forced_xplus_u_%d' % (j + 1))
                #     # self.mod.addConstr(gu.LinExpr([(b_power_A[i], x[self.m + i])
                #     #                                for i in range(self.n)]) >= -1e-2,
                #     #                    name='forced_xplus_l_%d' % (j + 1))
                #     self.mod.addConstr(x[self.m + j] == xplus_constraint[j],
                #                        name='forced_xplus_l_%d' % (j + 1))
            # else:
            #     c = [l for l in self.mod.getConstrs() if "forced_xplus_" in l.ConstrName]
            #     for l in c:
            #         self.mod.remove(l)
        else:
            print "Solver " + self.solver + " not implemented!"
            raise SystemExit()

    def approximate(self, strategy_in=None, audit_in=None, outputs_in=None):
        """
        Create an iterative approximation of the Q function for the system self.s

        :param strategy_in: Dictionary of parameters describing the solution strategy
        :param audit_in: Dict of parameters describing how progress should be tracked during sol'n
        :param outputs_in: Dict of parameters determining outputs produced
        :return: Convergence data structure
        """

        strategy = self.default_strategy
        audit = self.default_audit
        outputs = self.default_outputs
        if strategy_in is not None and isinstance(strategy_in, dict):
            for (k, v) in strategy_in.iteritems():
                strategy[k] = v
        if audit_in is not None and isinstance(audit_in, dict):
            for (k, v) in audit_in.iteritems():
                audit[k] = v
        if outputs_in is not None and isinstance(outputs_in, dict):
            for (k, v) in outputs_in.iteritems():
                outputs[k] = v

        j_max, n_z_points = strategy['max_iter'], strategy['n_z_points']
        on_policy = strategy['on_policy']
        rand_seed = strategy['rand_seed']
        focus_on_origin = strategy['focus_on_origin']
        sol_strategy = strategy['sol_strategy']
        conv_tol, stop_on_conv = strategy['conv_tol'], strategy['stop_on_convergence']
        remove_red, removal_freq = strategy['remove_redundant'], strategy['removal_freq']
        removal_res = int(strategy['removal_resolution'])
        consolidate_constraints = strategy['consolidate_constraints']
        consolidation_freq = strategy['consolidation_freq']
        q_function_limit = strategy['q_function_limit']

        eval_convergence = audit['eval_convergence']
        eval_convergence_freq = audit['eval_convergence_freq']
        n_ind_x_points = audit['n_independent_x']
        eval_ub, eval_ub_freq = audit['eval_ub'], audit['eval_ub_freq']
        eval_ub_final = audit['eval_ub_final']
        eval_ind_bellman, eval_ind_bellman_freq = audit['eval_ind_bellman'], audit['eval_ind_bellman_freq']
        eval_ind_integral, eval_ind_integral_freq = audit['eval_ind_integral'], audit['eval_ind_integral_freq']        

        cl_plot_j, cl_plot_final = outputs['cl_plot_j'], outputs['cl_plot_final']
        cl_plot_freq, cl_plot_n_steps = outputs['cl_plot_freq'], outputs['cl_plot_n_steps']
        qfa_plot_j, qfa_plot_final = outputs['qfa_plot_j'], outputs['qfa_plot_final']
        qfa_plot_freq = outputs['qfa_plot_freq']
        policy_plot_j, policy_plot_final = outputs['policy_plot_j'], outputs['policy_plot_final']
        policy_plot_freq = outputs['policy_plot_freq']
        suppress_outputs = outputs['suppress all']

        self.on_policy = on_policy

        # Create M samples for QF fitting
        np.random.seed(rand_seed)
        distr = 'uniform'
        x_unif_lower = [x_unif_lower_default] * self.n
        x_unif_upper = [x_unif_upper_default] * self.n
        if focus_on_origin:
            if distr == 'normal':
                xalg_list = [self.s.state_mean + np.dot(np.sqrt(self.s.state_var),
                                                        np.random.randn(self.n))
                             for k in range(n_z_points / 2)] + \
                            [self.s.state_mean + np.dot(np.sqrt(self.s.state_var) / 5.,
                                                        np.random.randn(self.n))
                             for k in range(n_z_points / 2)]
            elif distr == 'laplace':
                laplace_rv = [laplace(self.s.state_mean[m], np.diag(np.sqrt(self.s.state_var))[m])
                              for m in range(self.n)]
                laplace_rv2 = [laplace(self.s.state_mean[m], np.diag(np.sqrt(self.s.state_var))[m] / 5.)
                               for m in range(self.n)]
                xalg_list = [np.array([laplace_rv[m].rvs() for m in range(self.n)])
                             for k in range(n_z_points / 2)] + \
                            [np.array([laplace_rv2[m].rvs() for m in range(self.n)])
                             for k in range(n_z_points / 2)]
            else:
                print "Can only use 'focus_on_origin' with Laplace or Normal distributions of X_alg"
                raise SystemExit()
        else:
            if distr == 'normal':
                xalg_list = [self.s.state_mean + np.dot(np.sqrt(self.s.state_var),
                                                        np.random.randn(self.n))
                             for k in range(n_z_points)]
            elif distr == 'laplace':
                laplace_rv = [laplace(self.s.state_mean[m], np.diag(np.sqrt(self.s.state_var))[m])
                              for m in range(self.n)]
                xalg_list = [np.array([laplace_rv[m].rvs() for m in range(self.n)])
                             for k in range(n_z_points)]
            elif distr == 'uniform':
                assert len(x_unif_lower) == len(x_unif_upper) == self.n
                xalg_per_dim_list = []
                for dim in range(self.n):
                    xalg_per_dim_list.append(np.random.uniform(low=x_unif_lower[dim],
                                                               high=x_unif_upper[dim],
                                                               size=(n_z_points,)))
                xalg_list = [np.array([xalg_per_dim_list[dim][i] for dim in range(self.n)])
                             for i in range(n_z_points)]
            else:
                print "Unrecognised distribtuion '" + distr + "' when generating X_alg! Exiting."
                raise SystemExit()

        # Random u inputs associated with each x (these are not used if "on policy" inputs active)
        u_unif_lower, u_unif_upper = u_unif_lower_default * self.m, u_unif_upper_default * self.m
        ualg_per_dim_list = []
        for dim in range(self.m):
            ualg_per_dim_list.append(np.random.uniform(low=u_unif_lower[dim],
                                                       high=u_unif_upper[dim],
                                                       size=(n_z_points,)))
        ualg_list = [np.array([ualg_per_dim_list[dim][i] for dim in range(self.m)])
                     for i in range(n_z_points)]

        # self.create_audit_samples(n_ind_x_points, seed=rand_seed)  # Create ind't audit samples

        gamma = self.s.gamma
        self.xalg_list = xalg_list
        self.ualg_list = ualg_list
        self.total_solver_time = 0.
        self.b_gap_eval_time = 0.
        self.v_integral_eval_time = 0.
        self.lb_computation_time = 0.
        t1 = time.time()

        # Results for the states visited by the algorithm
        lb_constr_count = np.zeros((j_max,), dtype=int)
        xalgualg_integral_results = []
        xalgualg_bellman_results = []
        xalgualg_cl_ub_results = []
        qf_changed = True  # Flag to indicate Q function changed wrt last iteration

        # Results for independently-generated states
        ind_integral_results = []
        ind_bellman_results = []
        ind_cl_ub_results = []

        # Convergence data
        samples_converged = False
        convergence_j = None
        self.x_visited, self.u_visited = [], []

        for j in range(j_max):

            print "Iteration %d" % j

            lb_constr_count[j] = len(self.alpha_c_in_model)
            if consolidate_constraints and divmod(j, consolidation_freq)[1] == 0:
                self.consolidate_bounds()
            # Eval Bellman error convergence
            j_earliest_likely = 0  # Manipulate to hack iteration #s where convergence is checked
            if eval_convergence \
                    and (divmod(j, eval_convergence_freq)[1] == 0
                         and (j == 0 or j >= j_earliest_likely)) \
                    and qf_changed:
                print "  Measuring QF integral and BE for M=%d elements of ZAlg... " % n_z_points,
                if j > 0 and sol_strategy == 'biggest_gap':
                    old_largest_bellman_error = xalgualg_tq_of_xu[k] - xalgualg_q_of_xu[k]
                else:
                    old_largest_bellman_error = 0.
                xalgualg_q_of_xu, xalgualg_tq_of_xu = [], []
                for m, x in enumerate(xalg_list):
                    if on_policy:
                        u_to_test = self.q_policy(x)
                    else:
                        u_to_test = ualg_list[m]
                    xalgualg_q_of_xu.append(self.eval_qfa(x, u_to_test))
                    xalgualg_tq_of_xu.append(self.solve_for_xhatuhat(x, u_to_test,
                                                                     extract_constr=False)[2])
                    if xalgualg_tq_of_xu[-1] < xalgualg_q_of_xu[-1] - 1e-4:
                        print "  Negative QBE found at element %d of ZAlg: %.5f" % \
                              (len(xalgualg_q_of_xu), xalgualg_tq_of_xu[-1] - xalgualg_q_of_xu[-1])
                xalgualg_q_of_xu = np.array(xalgualg_q_of_xu)
                xalgualg_tq_of_xu = np.array(xalgualg_tq_of_xu)
                xalgualg_integral_results.append((copy.copy(j), np.mean(xalgualg_q_of_xu)))
                xalgualg_bellman_results.append((copy.copy(j),
                                                 np.mean(xalgualg_tq_of_xu - xalgualg_q_of_xu),
                                                 np.mean((xalgualg_tq_of_xu - xalgualg_q_of_xu)
                                                         / xalgualg_q_of_xu),
                                                 np.max(xalgualg_tq_of_xu - xalgualg_q_of_xu),
                                                 np.max((xalgualg_tq_of_xu - xalgualg_q_of_xu)
                                                        / xalgualg_q_of_xu)))
                print "mean = %.6f, max = %.6f" % (np.mean(xalgualg_tq_of_xu - xalgualg_q_of_xu),
                                                   np.max(xalgualg_tq_of_xu - xalgualg_q_of_xu))

            # Evaluate Q function upper bound (NOTE: not changed from V-function implementation!)
            # x_data, x_upper = None, None
            # if eval_ub and divmod(j, eval_ub_freq)[1] == 0:
            #     print "  Evaluating closed-loop UB for M=%d elements of XAlg..." % n_z_points
            #     if not (eval_convergence and divmod(j, eval_convergence_freq)[1] == 0):
            #         # Generate V(x) for points in XAlg if not done above
            #         xalgualg_q_of_xu = np.array([self.eval_vfa(x) for x in xalg_list])
            #     xalg_ubs, xalg_fi = self.measure_many_cl_upper_bounds(xalg_list,
            #                                                           n_steps=cl_plot_n_steps,
            #                                                           j=j)
            #     xalg_ubs_nou, xalg_fi_nou = self.measure_many_cl_upper_bounds(xalg_list,
            #                                                                   n_steps=cl_plot_n_steps,
            #                                                                   j=j,
            #                                                                   no_input=True)
            #     if len(xalg_fi) > 0:
            #         xalgualg_cl_ub_results.append((copy.copy(j),
            #                                    np.sum(xalg_ubs[xalg_fi]) /
            #                                    np.sum(xalgualg_q_of_xu[xalg_fi]) - 1.0,
            #                                    np.sum(xalg_ubs_nou[xalg_fi_nou]) /
            #                                    np.sum(xalgualg_q_of_xu[xalg_fi_nou]) - 1.0))
            #     else:
            #         xalgualg_cl_ub_results.append((copy.copy(j), np.inf, np.inf))
            # Measure integral of value function approximation over x samples
            # if eval_ind_integral and divmod(j, eval_ind_integral_freq)[1] == 0:
            #     print "  Measuring VF integral for %d independent samples..." % n_ind_x_points
            #     ind_v_of_x = np.array([self.eval_qfa(x) for x in self.v_integral_x_list])
            #     t1 = time.time()
            #     ind_integral_results.append((copy.copy(j), np.mean(ind_v_of_x)))
            #     t2 = time.time()
            #     self.v_integral_eval_time += t2 - t1
            # Measure Bellman gap TV(x) - V(x) over x samples
            # if eval_ind_bellman and divmod(j, eval_ind_bellman_freq)[1] == 0:
            #     print "  Measuring Bellman error for %d independent samples..." % n_ind_x_points
            #     avg_gap, avg_rel_gap, max_gap, max_rel_gap = self.measure_ind_bellman_gap()
            #     ind_bellman_results.append((copy.copy(j), avg_gap, avg_rel_gap, max_gap, max_rel_gap))
            #     print "  Measuring closed-loop UB for %d independent samples..." % n_ind_x_points
            #     if not (eval_ind_integral and divmod(j, eval_ind_integral_freq)[1] == 0):
            #         ind_v_of_x = np.array([self.eval_vfa(x) for x in self.v_integral_x_list])
            #     ind_ubs, ind_fi = self.measure_many_cl_upper_bounds(self.v_integral_x_list,
            #                                                         n_steps=cl_plot_n_steps,
            #                                                         j=j,
            #                                                         no_input=False)
            #     ind_ubs_nou, ind_fi_nou = self.measure_many_cl_upper_bounds(self.v_integral_x_list,
            #                                                                 n_steps=cl_plot_n_steps,
            #                                                                 j=j,
            #                                                                 no_input=True)
            #     if len(ind_fi) > 0:
            #         ind_cl_ub_results.append((copy.copy(j),
            #                                   np.sum(ind_ubs[ind_fi]) /
            #                                   np.sum(ind_v_of_x[ind_fi]) - 1.0,
            #                                   np.sum(ind_ubs_nou[ind_fi_nou]) /
            #                                   np.sum(ind_v_of_x[ind_fi_nou]) - 1.0))
            #     else:
            #         ind_cl_ub_results.append((copy.copy(j), np.inf, np.inf))
            # Remove redundant lower bounding functions (based on samples)
            # if remove_red and divmod(j, removal_freq)[1] == 0:
            #     print "  Removing redundant lower-bounding constraints..."
            #     self.remove_redundant_lbs(n_samples=removal_res, threshold=0.00, sigma_in=np.pi)

            # Plot closed-loop trajectories

            if cl_plot_j and divmod(j, cl_plot_freq)[1] == 0 and not suppress_outputs:
                print "  Plotting a closed-loop trajectory..."
                x0 = np.ones((self.n,), dtype=float)
                self.s.simulate(x0, self, n_steps=cl_plot_n_steps, iter_no=j, save_plot=True)
                # Save timing info
                pd.DataFrame([('LB computation time', self.lb_computation_time)]).to_csv(
                    'output/' + self.s.name + '/comp_time_%d.csv' % j,
                    header=['Category', 'Time'], index=None, float_format='%.4f')

            # Plot QF approximation
            if qfa_plot_j and (divmod(j, qfa_plot_freq)[1] == 0 or j < 10) and not suppress_outputs:
                print "  Plotting value function approximation..."
                self.plot_qfa(iter_no=j, save=True, output_dir='output/' + self.s.name)
            # Plot policy
            if policy_plot_j and divmod(j, policy_plot_freq)[1] == 0 and not suppress_outputs:
                print "  Plotting control policy..."
                self.plot_policy(iter_no=j, save=True, output_dir='output/' + self.s.name)

            # Pick the next x
            if sol_strategy == 'random':
                # Choose an x sample index at random and generate a new LB constraint from it
                qfa_too_large = True
                while qfa_too_large:
                    # Keep testing different samples k until one with an open gap found
                    k = np.random.randint(0, n_z_points)
                    x_picked = xalg_list[k]
                    if on_policy:
                        u_picked = self.q_policy(x_picked)
                    else:
                        u_picked = ualg_list[k]
                    if self.eval_qfa(x_picked, u_picked) <= q_function_limit:
                        qfa_too_large = False
            elif sol_strategy == 'biggest_gap':
                old_k = copy.copy(k)
                if not xalgualg_bellman_results[-1][0] == j:
                    # Evaluate Bellman error at all points in XAlg if conv. check not done at it. j
                    old_largest_bellman_error = xalgualg_tq_of_xu[k] - xalgualg_q_of_xu[k]
                    for i, x in enumerate(xalg_list):
                        xalgualg_q_of_xu[i] = self.eval_vfa(x)
                        _, opt_cost, _ = self.solve_for_xhat(x, iter_no=j, extract_constr=False)
                        xalgualg_tq_of_xu[i] = opt_cost

                k = np.argsort(xalgualg_tq_of_xu - xalgualg_q_of_xu)[-1]  # Pick largest B. error
                print "  Largest BE is at sample %d/%d: %.5f" % (k+1, n_z_points,
                                                                 xalgualg_tq_of_xu[k] -
                                                                 xalgualg_q_of_xu[k])
                x_picked = xalg_list[k]
                if k == old_k and old_largest_bellman_error == xalgualg_tq_of_xu[k] - \
                        xalgualg_q_of_xu[k]:
                    print "Index and value of largest Bellman error didn't change!"
            else:
                print "Unrecognized solution strategy:", sol_strategy
                raise SystemExit()

            # Extract new lower-bounding constraint for the x, u sample picked
            print "  x_%d = [" % (k + 1) + ", ".join(["%.3f" % x_el for x_el in x_picked]) + "]" + \
                  "  u_%d = [" % (k + 1) + ", ".join(["%.3f" % u_el for u_el in u_picked]) + "]"
            t_start_lb = time.time()
            new_constr, sol = self.solve_for_xhatuhat(x_picked, u_picked, extract_constr=True)
            t_end_lb = time.time()
            self.lb_computation_time += t_end_lb - t_start_lb
            self.x_visited.append(x_picked)
            self.u_visited.append(u_picked)
            xplus, uplus, st_cost, alpha = sol
            alpha += self.s.quad_stage_cost(xplus, uplus)
            primal_optimal_value = st_cost + gamma * alpha
            qfa_xu_picked = self.eval_qfa(x_picked, u_picked)
            gap_found = primal_optimal_value - qfa_xu_picked

            # Modify the QF approximation
            qf_changed = False  # Only set back to true if a new LB is added
            if gap_found >= 1e-5:  # Bellman error is sufficiently large to add a new LB
                if new_constr.dodgy:
                    print "  New LB constr. at sample %d unreliable. Not adding to model." % k
                elif new_constr.worthless:
                    print "  New LB constr. doesn't increase VFA due to duality gap. Not added."
                else:
                    self.add_alpha_constraint(new_constr)
                    qf_changed = True
                    # new_constr.plot_function(output_dir='output/'+self.s.name, iter_no=j)
                    print "  New LB constraint added to model. TQ(x,u) - Q(x,u) = %.7f" % gap_found
            elif gap_found <= -1e-3:  # Bellman error is sufficiently negative to be suspicious
                print "  Negative Bellman error found for sample %d: %.7f!" % (k + 1, gap_found)
                print "x_picked:", x_picked, "u_picked:", u_picked
                # print "u, xplus, beta, alpha:", u, xplus, beta, alpha
                print "Alpha constraints:"
                for c in self.alpha_c_in_model:
                    print "Constraint %d:" % c.id + " V(x_picked) - g_i(x_picked):", \
                        qfa_xu_picked - c.eval_at(x_picked, u_picked)
                    if qfa_xu_picked - c.eval_at(x_picked, u_picked) == 0:
                        print "Removing constraint from the model."
                        self.remove_alpha_constraint(c)
                        qf_changed = True
            else:
                print "  No new constraint added for sample %d: QBE of %.7f reached." % \
                      (k + 1, gap_found)

            if eval_convergence and divmod(j, eval_convergence_freq)[1] == 0 and \
                    xalgualg_bellman_results[-1][3] <= conv_tol and not samples_converged:
                samples_converged, convergence_j = True, j
                print "  Bellman convergence detected to within", str(conv_tol)
                if stop_on_conv:
                    break

        if eval_convergence and convergence_j is None:
            print "Completed %d iterations without convergence." % j_max
            print "Tolerance reached was %.5f." % xalgualg_bellman_results[-1][3]
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
        print "  QF integral for M=%d elements of ZAlg..." % n_z_points
        if on_policy:
            xalgualg_q_of_xu = np.array([self.eval_qfa(x, self.q_policy(x)) for x in xalg_list])
        else:
            xalgualg_q_of_xu = np.array([self.eval_qfa(x, ualg_list[m]) for m, x in enumerate(xalg_list)])
        xalgualg_integral_results.append((self.final_j, np.mean(xalgualg_q_of_xu)))
        print "  Bellman error for M=%d elements of ZAlg..." % n_z_points
        if on_policy:
            xalgualg_tq_of_xu = np.array(
                [self.solve_for_xhatuhat(x, self.q_policy(x), extract_constr=False)[2]
                 for x in xalg_list])
        else:
            xalgualg_tq_of_xu = np.array(
                [self.solve_for_xhatuhat(x, ualg_list[m], extract_constr=False)[2]
                 for m, x in enumerate(xalg_list)])
        xalg_negative_bellman_list = []
        xalg_negative_bellman_errors = False
        for i, qxu in enumerate(xalgualg_q_of_xu):
            if xalgualg_tq_of_xu[i] < qxu - 1e-2:
                print i, qxu, xalgualg_tq_of_xu[i]
                xalg_negative_bellman_list.append(i + 1)
                xalg_negative_bellman_errors = True
        if xalg_negative_bellman_errors:
            print "Negative Bellman errors found at samples", xalg_negative_bellman_list

        # print "  VF integral for %d independent samples..." % n_ind_x_points
        # t1 = time.time()
        # ind_v_of_x = np.array([self.eval_vfa(x) for x in self.v_integral_x_list])
        # t2 = time.time()
        # self.v_integral_eval_time += t2 - t1
        # ind_integral_results.append((self.final_j, np.mean(ind_v_of_x)))
        # print "  Bellman error for %d independent samples..." % n_ind_x_points
        # avg_gap, avg_rel_gap, max_gap, max_rel_gap = self.measure_ind_bellman_gap()
        # ind_bellman_results.append((self.final_j, avg_gap, avg_rel_gap, max_gap, max_rel_gap))
        # if eval_ub_final:
        #     print "  Closed-loop UB for M=%d elements of XAlg..." % n_z_points
        #     xalg_ubs, xalg_fi = self.measure_many_cl_upper_bounds(xalg_list,
        #                                                           n_steps=cl_plot_n_steps,
        #                                                           j=self.final_j)
        #     xalg_ubs_nou, xalg_fi_nou = self.measure_many_cl_upper_bounds(xalg_list,
        #                                                                   n_steps=cl_plot_n_steps,
        #                                                                   j=self.final_j,
        #                                                                   no_input=True)
        #     if len(xalg_fi) > 0:
        #         xalgualg_cl_ub_results.append((copy.copy(self.final_j),
        #                                    np.sum(xalg_ubs[xalg_fi]) / np.sum(xalgualg_q_of_xu[xalg_fi]) - 1.0,
        #                                    np.sum(xalg_ubs_nou[xalg_fi_nou]) / np.sum(xalgualg_q_of_xu[xalg_fi_nou]) - 1.0))
        #     else:
        #         xalgualg_cl_ub_results.append((copy.copy(self.final_j), np.inf, np.inf))
        #     print "  Closed-loop UB for %d independent samples..." % n_ind_x_points
        #     ind_ubs, ind_fi = self.measure_many_cl_upper_bounds(self.v_integral_x_list,
        #                                                         n_steps=cl_plot_n_steps,
        #                                                         j=j)
        #     ind_ubs_nou, ind_fi_nou = self.measure_many_cl_upper_bounds(self.v_integral_x_list,
        #                                                                 n_steps=cl_plot_n_steps,
        #                                                                 j=j,
        #                                                                 no_input=True)
        #     if len(ind_fi) > 0:
        #         ind_cl_ub_results.append((self.final_j,
        #                                   np.sum(ind_ubs[ind_fi]) / np.sum(ind_v_of_x[ind_fi]) - 1.0,
        #                                   np.sum(ind_ubs_nou[ind_fi_nou]) / np.sum(ind_v_of_x[ind_fi_nou]) - 1.0))
        #     else:
        #         ind_cl_ub_results.append((self.final_j, np.inf, np.inf))

        self.plot_xalg_be(xalgualg_q_of_xu, xalgualg_tq_of_xu)

        xalgualg_bellman_results.append((self.final_j,
                                         np.mean(xalgualg_tq_of_xu - xalgualg_q_of_xu),
                                         np.mean((xalgualg_tq_of_xu - xalgualg_q_of_xu)/xalgualg_q_of_xu),
                                         np.max(xalgualg_tq_of_xu - xalgualg_q_of_xu),
                                         np.max((xalgualg_tq_of_xu - xalgualg_q_of_xu)/xalgualg_q_of_xu)))

        # Output convergence results
        convergence_data = (xalgualg_integral_results, xalgualg_bellman_results,
                            xalgualg_cl_ub_results, ind_integral_results, ind_bellman_results,
                            ind_cl_ub_results, lb_constr_count, self.final_j, iter_time,
                            copy.copy(self.one_stage_problems_solved))

        self.output_convergence_results(convergence_data, n_z_points, conv_tol,
                                        'output/' + self.s.name)

        # Plot closed-loop trajectories
        if cl_plot_final and not suppress_outputs:
            print "  Plotting final closed-loop trajectory..."
            x0 = np.ones((self.n,), dtype=float)
            self.s.simulate(x0, self, cl_plot_n_steps, iter_no=self.final_j, save_plot=True)
        # Plot VF approximation
        if qfa_plot_final and not suppress_outputs:
            print "  Plotting final value function approximation..."
            self.plot_qfa(iter_no=self.final_j, save=True, output_dir='output/' + self.s.name)
        # Plot policy
        if policy_plot_final and not suppress_outputs:
            print "  Plotting final control policy..."
            self.plot_policy(iter_no=self.final_j, save=True, output_dir='output/' + self.s.name)

        total_time = time.time() - t1

        print "Done in %.1f seconds." % total_time
        # Time breakdown commented out, because it doesn't always add up properly.
        # print "  %.4f s spent computing lower bounds," % self.lb_computation_time
        # print "  %.4f s spent measuring V integral," % self.v_integral_eval_time
        # print "  %.4f s spent auditing Bellman gap," % self.b_gap_eval_time
        # print "  and %.1f s elsewhere." % (total_time - self.v_integral_eval_time -
        #                                    self.b_gap_eval_time - self.lb_computation_time)

        return convergence_data

    def solve_for_xhatuhat(self, xhat, uhat, print_sol=False, extract_constr=True,
                           print_errors=False, xplus_constraint=None):
        """ Solve one-stage optimization problem, optionally returning a lower-bounding function
        on the optimal value function.

        :param xhat: Fixed state parameter to solve the one stage problem for
        :param uhat: Fixed input parameter to solve the one stage problem for
        :param print_sol: Boolean, print solution details
        :param extract_constr: Boolean, extract constraint (returns different values if so)
        :param print_errors: Boolean, print error information
        :param xplus_constraint: e-element array or None, Force successor state to this value
        :return: If extract_constr: new lower bound, (u*, x+*, beta*, alpha*)
                 Otherwise: u*, optimal cost, stage cost
        """

        self.update_constraints_for_xhatuhat(xhat, uhat, xplus_constraint)
        check_lambda_a = True

        if self.solver == 'gurobi':
            ts1 = time.time()

            self.mod.optimize()
            self.total_solver_time += time.time() - ts1
            self.one_stage_problems_solved += 1

            if self.mod.status in [2, 13]:
                if self.mod.status == 13:
                    print "  Warning: Optimization status 13 (suboptimal) encountered!"
                opt_cost = self.mod.getObjective().getValue()
                sol_vec = [v.x for v in self.mod.getVars()]
                xplus = np.array(sol_vec[:self.n])
                uplus = np.array(sol_vec[self.n:self.n+self.m])
                stage_cost = opt_cost - self.s.gamma * self.eval_qfa(xplus, uplus)
                assert np.abs(stage_cost - self.s.quad_stage_cost(xhat, uhat)) < 1e-4
                # Can just return the optimal input and cost if we are only interested in that
                if not extract_constr:
                    return sol_vec[:self.n], sol_vec[self.n:self.n + self.m], opt_cost, stage_cost

                dodgy_bound = False
                pi = np.zeros(self.n, dtype=float)
                for i in range(self.n):
                    try:
                        pi[i] = self.mod_c[i].Pi
                    except Exception as e:
                        dodgy_bound = True
                        if print_errors:
                            print "Can't extract multiplier for constraint " + \
                                  self.mod_c[i].getAttr("ConstrName") + "! Using 0."
                            print e
                pi = np.array(pi)

                lambda_c = np.zeros(self.s.D.shape[0])
                for i in range(self.s.D.shape[0]):
                    c_name = 'sic_%d' % (i + 1)
                    try:
                        lambda_c[i] = self.mod_c_c[i].Pi
                    except Exception as e:
                        dodgy_bound = True
                        if print_errors:
                            print "Can't extract multiplier for constraint " + c_name + "! Using 0."
                            print e

                # lambda_a is only needed for audit purposes and can therefore be commented out
                lambda_a = []
                if check_lambda_a:
                    for c in self.mod_alpha_c:
                        try:
                            lambda_a.append(c.Pi)
                        except gu.GurobiError:
                            print c.getAttr('ConstrName')
                        except AttributeError:
                            # If the constraint doesn't have the attribute Pi, it could be a Gurobi
                            # quadratic constraint, which has a corresponding attribute QCPi.
                            try:
                                lambda_a.append(c.QCPi)
                            except gu.GurobiError:
                                print c.getAttr('QCName')
                    lambda_a = np.array(lambda_a)

            else:
                print "Optimization problem did not solve: status %d" % self.mod.status
                raise SystemExit()
        else:
            print "Solver " + self.solver + " not implemented!"
            raise SystemExit()

        if print_sol:
            print "Optimal cost:", opt_cost
            print "x+:", [round(p, 2) for p in sol_vec[:self.n]], \
                "u+:", [round(p, 2) for p in sol_vec[self.n:self.n + self.m]], \
                "alpha:", round(sol_vec[-1], 2)
            print "pi:", [round(p, 2) for p in pi], \
                "lambda_c:", [round(p, 5) for p in lambda_c], \
                "lambda_alpha:", [round(p, 5) for p in lambda_a]

        if check_lambda_a:
            if np.abs(np.sum(lambda_a) - self.s.gamma) >= 1e-3:
                print "  Warning: Incorrect lambda_a sum: %.5f. To fix." % np.sum(lambda_a)
            assert np.all([a > -1e-4 for a in lambda_a]), \
                "Not all lambda_a multipliers positive! " + repr(lambda_a)

        f_x_const, f_x_lin = self.s.f_x_x(None)
        f_u_lin = self.s.f_u_x(None)

        # Excluding stage cost (because all bounds q_i(.,.) have the stage cost in, this gets
        # added later when the Gurobi model is built):
        xhat_uhat_stage_cost = self.s.quad_stage_cost(xhat, uhat)
        bound_out_const = opt_cost - xhat_uhat_stage_cost - np.dot(pi, xplus)
        bound_out_x_lin = np.dot(pi, f_x_lin)
        bound_out_u_lin = np.dot(pi, f_u_lin)
        bound_out_x_quad = None
        bound_out_u_quad = None
        new_bound = QConstraint(n=self.n, m=self.m, id_in=len(self.alpha_c_list),
                                const_in=bound_out_const, x_lin_in=bound_out_x_lin,
                                x_hessian_in=bound_out_x_quad, u_lin_in=bound_out_u_lin,
                                u_hessian_in=bound_out_u_quad, sys_in=self.s)

        if print_sol:
            print "s", bound_out_const, "p", bound_out_x_lin, "P", bound_out_x_quad

        # At equilibrium (assumed to be x=0, u=0), value function should be 0. The following lines
        # check whether the the lower-bounding function takes a value > 0 there.
        bound_value_at_eq = new_bound.eval_at(np.array([0.] * self.n), np.array([0.] * self.m))
        if bound_value_at_eq > 1e-3:
            print "  New bound takes value %.3f at origin! Not adding to model." % bound_value_at_eq
            new_bound.dodgy = True

        if dodgy_bound:
            new_bound.dodgy = True

        return new_bound, (sol_vec[:self.n], sol_vec[self.n:self.n+self.m],
                           stage_cost, sol_vec[self.n+self.m])

    def solve_for_xhat(self, xhat, print_sol=False, extract_constr=True, print_errors=False,
                       iter_no=None, alt_bound=False, xplus_constraint=None):
        u = self.q_policy(xhat)
        return u, None, 0

    def q_policy(self, xhat):
        if self.solver == 'gurobi':
            # Update model constraints to fix the function we are minimizing to Q(xhat, u)
            for i in range(self.n):
                self.pmod_c[i].setAttr('RHS', xhat[i])
            self.pmod.update()
            # Minimize Q(xhat, u) over u
            self.pmod.optimize()
            if self.pmod.status in [2, 13]:
                if self.pmod.status == 13:
                    print "  Warning: Optimization status 13 (suboptimal) encountered!"
                sol_vec = [v.x for v in self.pmod.getVars()]
                xopt = np.array(sol_vec[:self.n])
                assert np.linalg.norm(xhat - xopt) < 1e-5
                uopt = np.array(sol_vec[self.n:self.n + self.m])
                return uopt
            else:
                print "Policy evaluation at", xhat, "returned status %d!" % self.pmod.status
                raise SystemExit()
        else:
            print "Solver " + self.solver + " not implemented! Exiting."
            raise SystemExit()

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
        const, x_lin, x_hess = constr_in.const, constr_in.x_lin, constr_in.x_hessian
        u_lin, u_hess = constr_in.u_lin, constr_in.u_hessian

        if temp_add:
            c_name = str(constr_id)
        else:
            c_name = "alpha_%d" % constr_id
        if self.solver == 'gurobi':
            if const is None:
                const = 0.
            if x_lin is not None:
                xle = gu.LinExpr([(x_lin[p], self.mod_x[p]) for p in range(self.n)])
                pxle = gu.LinExpr([(x_lin[p], self.pmod_x[p]) for p in range(self.n)])
            else:
                xle, pxle = 0., 0.
            if x_hess is not None:
                xqe = gu.QuadExpr()
                pxqe = gu.QuadExpr()
                for p in range(self.n):
                    xqe += 0.5 * self.mod_x[p] * gu.LinExpr([(x_hess[p, q], self.mod_x[q])
                                                             for q in range(self.n)])
                    pxqe += 0.5 * self.pmod_x[p] * gu.LinExpr([(x_hess[p, q], self.pmod_x[q])
                                                               for q in range(self.n)])
            else:
                xqe, pxqe = 0., 0.
            if u_lin is not None:
                ule = gu.LinExpr([(u_lin[p], self.mod_x[self.n + p]) for p in range(self.m)])
                pule = gu.LinExpr([(u_lin[p], self.pmod_x[self.n + p]) for p in range(self.m)])
            else:
                ule, pule = 0., 0.
            if u_hess is not None:
                uqe = gu.QuadExpr()
                puqe = gu.QuadExpr()
                for p in range(self.m):
                    uqe += 0.5 * self.mod_x[self.n + p] * gu.LinExpr(
                        [(u_hess[p, q], self.mod_x[self.n + q]) for q in range(self.m)])
                    puqe += 0.5 * self.pmod_x[self.n + p] * gu.LinExpr(
                        [(u_hess[p, q], self.pmod_x[self.n + q]) for q in range(self.m)])
            else:
                uqe, puqe = 0., 0.

            self.mod_alpha_c.append(
                self.mod.addConstr(self.mod_x[-1] >= const + xle + xqe + ule + uqe, name=c_name))
            self.pmod_alpha_c.append(
                self.pmod.addConstr(self.pmod_x[-1] >= const + pxle + pxqe + pule + puqe,
                                    name=c_name))
            self.mod.update()
            self.pmod.update()
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

    def eval_qfa(self, x_in, u_in):
        """Evaluates lower bound (max of all LB functions) at x_in by evaluating all LBs"""

        # if self.s.name == 'Inverted pendulum':
        #     x_in[0] = divmod(x_in[0] + np.pi, 2 * np.pi)[1] - np.pi
        f_out = -np.inf
        for c in self.alpha_c_in_model:
            f_out = max(f_out, c.eval_at(x_in, u_in))
        f_out += self.s.quad_stage_cost(x_in, u_in)
        return f_out

    def plot_qfa(self, iter_no=None, save=False, output_dir=None):
        """Plot value function approximation in 1 or 2 dimensions. Plot bounds and resolution are
        currently hard-coded inside the function.

        :param iter_no: Integer, iteration number
        :param save: Boolean, save value function plot. If false, show on screen instead
        :param output_dir: Output directory for plots
        :return: nothing
        """
        # Extract value function lower bounds from model constraints, then plot
        plt.figure(figsize=(5, 3))
        if save:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
        if self.n == 1 and self.m == 1:
            x_min, x_max, x1_res = default_x1_min, default_x1_max, default_x1_res
            u_min, u_max, u1_res = default_u1_min, default_u1_max, default_u1_res
            x1 = np.arange(x_min, x_max + x1_res, x1_res).tolist()
            x2 = np.arange(u_min, u_max + u1_res, u1_res).tolist()
            mesh = np.zeros((len(x1), len(x2)), dtype=float)
            for ind1, x1_plot in enumerate(x1):
                for ind2, x2_plot in enumerate(x2):
                    mesh[ind1, ind2] = self.eval_qfa(np.array([x1_plot]), np.array([x2_plot]))
            X, Y = np.meshgrid(x1, x2)
            # if self.x_visited is not None:
            #     savemat(output_dir + '/qfa_%d.mat' % iter_no, {'X': X, 'Y': Y, 'mesh': mesh,
            #                                                    'visited': self.x_visited})
            # else:
            #     savemat(output_dir + '/qfa_%d.mat' % iter_no, {'X': X, 'Y': Y, 'mesh': mesh})

            plot_visited, plot_xalgualg = True, True

            ax = plt.subplot()
            plt.imshow(mesh.T, aspect=0.8, origin='lower', cmap='bone',
                       extent=[x_min, x_max, u_min, u_max],
                       vmin=default_n1m1_vmin, vmax=default_n1m1_vmax)
            plt.xlim([x_min, x_max])
            plt.ylim([u_min, u_max])
            plt.xlabel('$x_1$')
            plt.ylabel('$u_1$')
            if plot_visited and self.x_visited is not None:
                ax.scatter([x[0] for x in self.x_visited], [u[0] for u in self.u_visited],
                           s=30, c='g', marker='x', linewidths=0)
            if plot_xalgualg:
                ax.scatter([x[0] for x in self.xalg_list],
                           [self.q_policy(x)[0] for x in self.xalg_list],
                           s=15, c='w', linewidths=0)
            # on_policy_string = "on-policy" if self.on_policy else "off-policy"
            on_policy_string = "Variant B" if self.on_policy else "Variant A"
            if iter_no is not None:
                plt.title('$Q_{%d}(x,u)$, ' % iter_no + on_policy_string)
            else:
                plt.title('Approximate $Q$-function, ' + on_policy_string)
        else:
            print "    Cannot plot Q-function approximation for n = %d, m = %d" % (self.n, self.m)
            return

        # Save or plot QFA
        if save:
            # plt.tight_layout(pad=0.0)
            ax.set_position([0.05, 0.18, 1.00, 0.72])
            plt.savefig(output_dir + '/qfa_%d.pdf' % iter_no)
        else:
            plt.show()
        plt.close()

    def plot_zalg_qbe(self, xalgualg_q_of_xu, xalgualg_tq_of_xu):
        """Plots the Q-Bellman error at all elements of the set XAlg (i.e. the sample points used in
        the algorithm)

        :param xalgualg_q_of_xu: List of Q(x, u) with length equal to no. of elements in XAlg
        :param xalgualg_tq_of_xu: List of TQ(x, u) with length equal to no. of elements in XAlg
        :return: nothing
        """

        n_x_points = xalgualg_q_of_xu.shape[0]

        if not os.path.exists('output/' + self.s.name):
            os.makedirs('output/' + self.s.name)
        plt.figure(figsize=(16, 8))
        plt.subplot(211)
        plt.bar(np.array(range(1, n_x_points + 1)) - 0.225, xalgualg_q_of_xu,
                width=0.45, color='b', edgecolor=None)
        plt.bar(np.array(range(1, n_x_points + 1)) + 0.225, xalgualg_tq_of_xu,
                width=0.45, color='r', edgecolor=None)
        plt.xlim([0, n_x_points + 1])
        plt.ylabel('$Q(x, u)$, $T_Q Q(x, u)$')
        plt.subplot(212)
        plt.bar(np.array(range(1, n_x_points + 1)) - 0.4,
                xalgualg_tq_of_xu - xalgualg_q_of_xu, edgecolor=None)
        plt.xlim([0, n_x_points + 1])
        plt.ylabel('$(T_Q Q(x, u) - Q(x, u))$')
        plt.ylim([min(-1., np.min(xalgualg_tq_of_xu - xalgualg_q_of_xu) * 1.05),
                  np.max(xalgualg_tq_of_xu - xalgualg_q_of_xu) * 1.05])
        plt.savefig('output/' + self.s.name + '/xalgualg_QBE.pdf')
        plt.close()

        plt.figure()
        plt.plot(xalgualg_tq_of_xu, 100. * (xalgualg_tq_of_xu - xalgualg_q_of_xu) / xalgualg_q_of_xu,
                 linewidth=0., marker='x')
        plt.xlabel('$Q(x, u)$')
        plt.ylabel('$(T_Q Q(x, u) - Q(x, u))/Q(x, u)$ (%)')
        plt.ylim([min(-1., np.min(100. * (xalgualg_tq_of_xu - xalgualg_q_of_xu) / xalgualg_q_of_xu) * 1.05),
                  min(100., np.max(100. * (xalgualg_tq_of_xu - xalgualg_q_of_xu) / xalgualg_q_of_xu) * 1.05)])
        plt.savefig('output/' + self.s.name + '/xalgualg_QBE_by_norm.pdf')
        plt.close()

    def save_function_approximation(self):
        # Saves the lower-bounding functions that describe the value function approximation as a
        # .mat file. The constant, linear, and quadratic parts are stored in separate lists.
        # The functions take the form q_i(x,u) = c.const + (c.lin)'x + 0.5 * x'(c.hessian)x.
        # As the functions are stored without the stage cost, this has to be added back here.
        const_list, x_lin_list, x_quad_list = [], [], []
        u_lin_list, u_quad_list = [], []
        for c in self.alpha_c_in_model:
            const_list.append(c.const + self.s.ti[0])
            x_lin_list.append(c.x_lin + self.s.phi_i_x_lin[0])
            mat_to_append = c.x_hessian if c.x_hessian is not None else np.zeros((self.n, self.n))
            mat_to_append += self.s.phi_i_x_quad[0]
            x_quad_list.append(mat_to_append)
            u_lin_list.append(c.u_lin + self.s.ri[0])
            mat_to_append = c.u_hessian if c.u_hessian is not None else np.zeros((self.m, self.m))
            mat_to_append += self.s.Ri[0]
            u_quad_list.append(mat_to_append)
        savemat('output/' + self.s.name + '/qfa_description.mat', {'q_const': const_list,
                                                                   'q_x_lin': x_lin_list,
                                                                   'q_x_quad': x_quad_list,
                                                                   'q_u_lin': u_lin_list,
                                                                   'q_u_quad': u_quad_list},
                oned_as='column')

    def output_convergence_results(self, data_in, nx, bell_tol_in, output_dir):
        """Save CSV files and graphs in PDF form regarding convergence of the algorithm

        :param data_in: Data structure containing quantities measured by iteration
        :param nx: Number of elements in XAlg
        :param output_dir: Output directory to save results to
        :return: nothing
        """
        (xalgualg_integral_results, xalgualg_bellman_results, xalgualg_cl_ub_results,
         ind_integral_results, ind_bellman_results, ind_cl_ub_results,
         lb_constr_count, convergence_j, iter_time, osp) = data_in

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        j_max = convergence_j if convergence_j is not None \
            else max([a[0] for a in xalgualg_integral_results])

        # Calculate DARE mean for plotting comparison
        # if self.s.dare_sol is not None:
        #     ind_dare_mean = np.mean([0.5 * np.dot(np.dot(x, self.s.dare_sol), x)
        #                              for x in self.v_integral_x_list])
        #     print "  DARE average cost across audit samples is %.2f" % ind_dare_mean
        # else:
        #     ind_dare_mean = None

        # CSV of XAlg VF approximation mean
        xalg_integral_csv_array = np.array(xalgualg_integral_results)
        try:
            np.savetxt(output_dir + '/xalg_lb_M' + str(nx) + '.csv', xalg_integral_csv_array,
                       fmt='%d,%.5f', delimiter=',', header='Iteration,Mean LB',
                       comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/xalg_lb_M' + str(nx) + '.csv'
        # CSV of XAlg Bellman converge nce
        xalg_bellman_csv_array = np.array(xalgualg_bellman_results)
        try:
            np.savetxt(output_dir + '/xalg_bellman_error_M' + str(nx) + '.csv',
                       xalg_bellman_csv_array,
                       fmt='%d,%.5f,%.5f,%.5f,%.5f', delimiter=',',
                       header='Iteration,Mean TV(x)-V(x),Mean (TV(x)-V(x))/V(x),Max TV(x)-V(x),Max (TV(x)-V(x))/V(x)',
                       comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/xalg_bellman_error_M' + str(nx) + '.csv'

        # # CSV of XAlg closed-loop cost upper bound
        # xalg_cl_ub_csv_array = np.array(xalgualg_cl_ub_results)
        # try:
        #     np.savetxt(output_dir + '/xalg_cl_ub_M' + str(nx) + '.csv', xalg_cl_ub_csv_array,
        #                fmt='%d,%.5f,%.5f', delimiter=',', header='Iteration,Mean UB,Mean UB u=0',
        #                comments='')
        # except Exception as e:
        #     print "Could not save " + output_dir + '/xalg_cl_ub_M' + str(nx) + '.csv'
        # ind_integral_csv_array = np.array(ind_integral_results)
        # try:
        #     np.savetxt(output_dir + '/ind_lb_M' + str(nx) + '.csv', ind_integral_csv_array,
        #                fmt='%d,%.5f', delimiter=',', header='Iteration,Mean LB', comments='')
        # except Exception as e:
        #     print "Could not save " + output_dir + '/xalg_lb_M' + str(nx) + '.csv'
        # # CSV of independent Bellman convergence
        # ind_bellman_csv_array = np.array(ind_bellman_results)
        # try:
        #     np.savetxt(output_dir + '/ind_bellman_error_M' + str(nx) + '.csv', ind_bellman_csv_array,
        #                fmt='%d,%.5f,%.5f,%.5f,%.5f', delimiter=',',
        #                header='Iteration,Mean TV(x)-V(x),Mean (TV(x)-V(x))/V(x),Max TV(x)-V(x),Max (TV(x)-V(x))/V(x)', comments='')
        # except Exception as e:
        #     print "Could not save " + output_dir + '/ind_bellman_error_M' + str(nx) + '.csv'
        # # CSV of independent closed-loop cost upper bound
        # ind_cl_ub_csv_array = np.array(ind_cl_ub_results)
        # try:
        #     np.savetxt(output_dir + '/ind_cl_ub_M' + str(nx) + '.csv', ind_cl_ub_csv_array,
        #                fmt='%d,%.5f,%.5f', delimiter=',', header='Iteration,Mean UB,Mean UB u=0',
        #                comments='')
        # except Exception as e:
        #     print "Could not save " + output_dir + '/ind_cl_ub_M' + str(nx) + '.csv'

        # Save timing info
        pd.DataFrame([('LB computation time', self.lb_computation_time),
                      ('Total iteration time', iter_time),
                      ('One-stage problems solved', osp)]).to_csv(output_dir + '/comp_time.csv',
                                                                  header=['Category', 'Time'],
                                                                  index=None, float_format='%.4f')

        # # Plot LB and UB convergence
        # plt.figure(figsize=(8, 12))
        # plt.subplots(3, 1)
        # plt.subplot(211)
        # plt.plot([a[0] for a in xalgualg_integral_results], [a[1] for a in xalgualg_integral_results],
        #          label='LB, $M=%d$ samples' % nx)
        # # plt.plot([a[0] for a in ind_integral_results], [a[1] for a in ind_integral_results],
        # #          label='LB, ind. samples')
        # # plt.plot([a[0] for a in xalgualg_cl_ub_results], [a[1] for a in xalgualg_cl_ub_results],
        # #          label='UB, $M=%d$ samples' % nx)
        # # plt.plot([a[0] for a in ind_cl_ub_results], [a[1] for a in ind_cl_ub_results],
        # #          label='UB, ind. samples')
        # # plt.plot([a[0] for a in xalgualg_cl_ub_results], [a[2] for a in xalgualg_cl_ub_results],
        # #          label='UB, u=0, $M=%d$ s' % nx)
        # # plt.plot([a[0] for a in ind_cl_ub_results], [a[2] for a in ind_cl_ub_results],
        # #          label='UB, u=0, ind. s')
        #
        # # if ind_dare_mean is not None:
        # #     plt.plot(range(j_max+1), [ind_dare_mean] * (j_max+1), color='r',
        # #              label='Unconstr. DARE sol')
        # plt.ylabel('Mean LB and UB')
        # plt.title('Mean LB and UB for $Q(x,u)$, %d sampled points' % nx)
        # plt.xlim([0, np.max([a[0] for a in xalgualg_integral_results])])
        # plt.ylim([-0.001, 2.0 * np.max([a[1] for a in xalgualg_integral_results])])
        # plt.legend(loc=4)
        # plt.grid()

        plt.figure(figsize=(8, 3))
        plt.subplot(111)
        plt.plot([a[0] for a in xalgualg_bellman_results],
                 [np.log10(a[1]) for a in xalgualg_bellman_results], label='Mean BE, $M=%d$ samples' % nx)
        plt.plot([a[0] for a in xalgualg_bellman_results],
                 [np.log10(a[3]) for a in xalgualg_bellman_results], label='Max BE, $M=%d$ samples' % nx)
        # plt.plot([a[0] for a in ind_bellman_results],
        #          [np.log10(a[1]) for a in ind_bellman_results], label='Mean BE, ind. samples')
        # plt.plot([a[0] for a in ind_bellman_results],
        #          [np.log10(a[2]) for a in ind_bellman_results], label='Max BE, ind. samples')
        # if len(xalgualg_cl_ub_results) == len(xalgualg_bellman_results):
        #     plt.plot([a[0] for a in xalgualg_integral_results],
        #              [np.log10(xalgualg_cl_ub_results[i][1] - a[1]) for i, a in enumerate(xalgualg_integral_results)],
        #              label='Mean UB-LB, $M=%d$ samples' % nx)
        # if len(ind_cl_ub_results) == len(ind_integral_results) and len(ind_integral_results) > 1:
        #     plt.plot([a[0] for a in ind_bellman_results],
        #              [np.log10(ind_cl_ub_results[i][1] - a[1]) for i, a in enumerate(ind_integral_results)],
        #              label='Mean UB-LB, ind. samples')

        plt.xlabel('Iteration number')
        plt.ylabel('$\log_{10}$ value')
        # on_policy_string = "on-policy" if self.on_policy else "off-policy"
        on_policy_string = "Variant B" if self.on_policy else "Variant A"
        plt.title('Convergence behaviour, ' + on_policy_string)
        fix_xlim, x_upper_value = False, 250
        if fix_xlim:
            plt.plot([0, x_upper_value],
                     [np.log10(bell_tol_in), np.log10(bell_tol_in)], 'k--', label='Conv. tolerance')
            plt.xlim([0, x_upper_value])
        else:
            plt.plot([0, np.max([a[0] for a in xalgualg_integral_results])],
                     [np.log10(bell_tol_in), np.log10(bell_tol_in)], 'k--', label='Conv. tolerance')
            plt.xlim([0, np.max([a[0] for a in xalgualg_integral_results])])
        # plt.ylim([-0.001, 10.0 * np.mean([a[1] for a in xalgualg_bellman_results])])
        plt.legend(loc=1)
        plt.grid()

        # plt.subplot(313)
        # # plt.plot([a[0] for a in xalgualg_bellman_results],
        # #          [np.log10(a[1]) for a in xalgualg_bellman_results],
        # #          label='Mean BE, $M=%d$ samples' % nx)
        # # plt.plot([a[0] for a in xalgualg_bellman_results],
        # #          [np.log10(a[2]) for a in xalgualg_bellman_results],
        # #          label='Max BE, $M=%d$ samples' % nx)
        # # plt.plot([a[0] for a in ind_bellman_results],
        # #          [np.log10(a[1]) for a in ind_bellman_results], label='Mean BE, ind. samples')
        # # plt.plot([a[0] for a in ind_bellman_results],
        # #          [np.log10(a[2]) for a in ind_bellman_results], label='Max BE, ind. samples')
        # if len(xalgualg_cl_ub_results) == len(xalgualg_bellman_results):
        #     plt.plot([a[0] for a in xalgualg_integral_results],
        #              [np.log10(cl_ub[1]) for cl_ub in xalgualg_cl_ub_results],
        #              label='Mean CL subopt., $M=%d$ samples' % nx)
        #     plt.plot([a[0] for a in xalgualg_integral_results],
        #              [np.log10(cl_ub[2]) for cl_ub in xalgualg_cl_ub_results],
        #              label='Mean CL subopt., u=0, $M=%d$ s.' % nx)
        # if len(ind_cl_ub_results) == len(ind_integral_results) and len(ind_integral_results) > 1:
        #     plt.plot([a[0] for a in ind_bellman_results],
        #              [np.log10(cl_ub[1]) for cl_ub in ind_cl_ub_results],
        #              label='Mean CL cubopt., ind. samples')
        #     plt.plot([a[0] for a in ind_bellman_results],
        #              [np.log10(cl_ub[2]) for cl_ub in ind_cl_ub_results],
        #              label='Mean CL subopt., u=0, ind. s.')
        # plt.xlabel('Iteration number')
        # plt.ylabel('$\log_{10}$ value')
        # plt.title('Relative convergence errors')
        # # plt.ylim([-0.001, 10.0 * np.mean([a[1] for a in xalgualg_bellman_results])])
        # plt.legend(loc=1)
        # plt.grid()
        plt.tight_layout()
        plt.savefig(output_dir + '/conv_M' + str(nx) + '.pdf')
        plt.close()

        # CSV LB constraint count
        lbc_csv_array = np.array([range(j_max), lb_constr_count[:j_max]])
        try:
            np.savetxt(output_dir + '/constr_count_M' + str(nx) + '.csv', lbc_csv_array.T,
                       fmt='%d,%d', delimiter=',', header='Iteration,No. of constraints',
                       comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/constr_count_M' + str(nx) + '.csv'
        plt.figure()
        plt.plot(range(j_max), lb_constr_count[:j_max])
        plt.xlim([0, j_max])
        plt.xlabel('Iteration number')
        plt.ylabel('No. of constraints')
        plt.title('Number of lower-bounding constraints $q_i(\cdot,\cdot)$')
        plt.grid()
        plt.savefig(output_dir + '/constr_count_M' + str(nx) + '.pdf')
        plt.close()

    def create_vfa_model(self):
        print "Objects of class" + self.__class__.__name__ + " cannot create VFA model. Exiting."
        raise SystemExit()
