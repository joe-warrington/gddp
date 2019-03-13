import numpy as np
import time
import gurobipy as gu
import copy
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.io import savemat, loadmat
from scipy.stats import laplace
import os
from gddp import VConstraint

mpl.rcParams['font.family'] = 'serif'
np.seterr(divide='ignore')

# Guroabi solution status codes
gu_status_codes = {1: "Loaded", 2: "Optimal", 3: "Infeasible", 4: "Infeasible or unbounded",
                   5: "Unbounded", 6: "Cutoff", 7: "Iteration limit", 8: "Node limit",
                   9: "Time limit", 10: "Solution limit", 11: "Interrupted", 12: "Numerical issues",
                   13: "Suboptimal", 14: "In progress", 15: "User objective limit"}

# Default value function approximation parameters
default_approx_strategy = {'max_iter': 100,
                           'n_x_points': 100, 'rand_seed': 1,
                           'sol_strategy': 'random', 'conv_tol': 1e-4,
                           'stop_on_convergence': False,
                           'remove_redundant': False, 'removal_freq': 10,
                           'removal_resolution': 100000,
                           'focus_on_origin': False, 'consolidate_constraints': False,
                           'consolidation_freq': False,
                           'value_function_limit': 10000}
default_approx_audit = {'eval_ub': False, 'eval_ub_freq': 5, 'eval_ub_final': False,
                        'eval_bellman': False, 'eval_bellman_freq': 5,
                        'eval_integral': False, 'eval_integral_freq': 5,
                        'n_independent_x': 100,
                        'eval_convergence': False, 'eval_convergence_freq': 5}
default_approx_outputs = {'cl_plot_j': False, 'cl_plot_freq': 20, 'cl_plot_final': False,
                          'cl_plot_n_steps': 50,
                          'vfa_plot_j': False, 'vfa_plot_freq': 1, 'vfa_plot_ub': False,
                          'vfa_plot_final': True,
                          'policy_plot_j': False, 'policy_plot_freq': 5,
                          'policy_plot_final': True,
                          'suppress all': False}

# Plotting ranges for 2D value functions
default_x1_min, default_x1_max, default_res1 = -3, 3, 0.1
default_x2_min, default_x2_max, default_res2 = -3, 3, 0.1
default_2d_vmin, default_2d_vmax = None, None


class VFApproximator(object):
    """VFApproximator generates approximate value functions for continuous state and action
    problems. Initialised with the system in question, which includes cost function, discount rate
    etc.
    """

    def __init__(self, system, solver='gurobi'):
        """Initialize VFApproximator object.

        :param system: System object on which the value function approximator will act
        :param solver: Solver to use ('gurobi' or None)
        """

        # Get properties from system
        self.m = system.m
        self.n = system.n
        self.n_beta = system.n_beta
        self.s = system

        # Create placeholders for optimization model
        self.mod = None
        self.solver = solver

        # Performance logging
        self.total_solver_time = None
        self.one_stage_problems_solved = 0
        self.b_gap_eval_time = None
        self.v_integral_eval_time = None
        self.lb_computation_time = None

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

        # Default approximation parameters
        self.default_strategy = default_approx_strategy
        self.default_audit = default_approx_audit
        self.default_outputs = default_approx_outputs

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
            initial_lb = VConstraint(self.n, 0, 0., None, None)
            self.add_alpha_constraint(initial_lb)

            # Update model to reflect constraints
            self.mod.update()
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
        else:
            print "Solver " + self.solver + " not implemented!"
            raise SystemExit()

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
                print "Optimization problem did not solve: status %d, '" % self.mod.status + \
                      gu_status_codes[self.mod.status] + "'"
                raise SystemExit()
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
            stacked_li = np.hstack(tuple(l.reshape((-1, 1)) for l in self.s.li))
            assert np.linalg.norm(np.dot(stacked_li, lambda_b)
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

        new_bound = VConstraint(n=self.n, id_in=len(self.alpha_c_list),
                                const_in=bound_out_const, lin_in=bound_out_lin,
                                hessian_in=bound_out_quad)

        if print_sol:
            print "s", bound_out_const, "p", bound_out_lin, "P", bound_out_quad

        new_bound_value_at_eq = new_bound.eval_at(np.array([0.] * self.n))
        if new_bound_value_at_eq > 1e-3:
            print "  New bound takes value %.3f at origin! Not adding to model." % new_bound_value_at_eq
            new_bound.dodgy = True

        if dodgy_bound:
            new_bound.dodgy = True

        return new_bound, (sol_vec[:self.m], sol_vec[self.m:self.m+self.n],
                           sol_vec[self.m+self.n:-1], sol_vec[-1])

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
        :param was_temp_added: Boolean, whether Constraint was temporarily added
        :param temp_remove: Boolean, whether to mark Constraint as temporarily removed
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
        else:
            print "Solver " + self.solver + " not implemented!"
            raise SystemExit()
        assert successfully_removed, "Didn't manage to remove constraint " + c_name + "!"
        if not was_temp_added and not temp_remove:
            constr_in.removed_from_model()
        if temp_remove:
            print "Temporarily removed constraint " + c_name + "."
        elif was_temp_added:
            print "Removed temporary virtual constraint " + c_name + "."
        else:
            print "Permanently removed temporary constraint " + c_name + "."
        self.alpha_c_in_model = [c for c in self.alpha_c_list if c.in_model]

    def reset_model_constraints(self):
        """Remove all constraints from the model and start with a zero lower-bounding function"""
        for c in self.alpha_c_in_model:
            self.remove_alpha_constraint(c)
        # Add initial alpha epigraph constraint
        initial_lb = VConstraint(self.n, 0, 0., None, None)
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

    def audit_gurobi_constraint_info(self):
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
            plt.plot(plot_range, v_max, linewidth=1.5, color='k', label='$\hat{V}(x)$')

            if self.s.dare_sol is not None and self.s.pure_lqr:
                plt.plot(plot_range, [0.5 * x ** 2 * self.s.dare_sol[0, 0] for x in plot_range],
                         linewidth=1.0, color='r', label='$V^\star (x)$ (unconstrained LQR)')

            plt.legend()
            plt.xlim([x_min, x_max])
            plt.xlabel('$x$')
            plt.ylim([-0.05, 4.00])
            if self.s.dare_sol is not None:
                dare_cost_at_bound = 0.5 * self.s.dare_sol[0, 0] * x_max * x_max
                plt.ylim([-0.05, 1.5 * dare_cost_at_bound])
            plt.ylabel('$\hat{V}_I(x)$')
            if iter_no is not None:
                plt.title('Value function approximation, %d iterations' % iter_no)
            else:
                plt.title('Value function approximation')
            plt.grid()

        elif self.n == 2 or self.n == 4:
            x1_min, x1_max, res1 = default_x1_min, default_x1_max, default_res1
            x2_min, x2_max, res2 = default_x2_min, default_x2_max, default_res2
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
                       extent=[x1_min, x1_max, x2_min, x2_max],
                       vmin=default_2d_vmin, vmax=default_2d_vmax)
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

    def print_function_approximation(self, save=False):
        # Outputs a string representation of the function approximation, optionally also to file.
        if self.__class__.__name__ == "QFApproximator":
            string_out = "Approximate QF is max_i{q_i(x,u)}, where:\n"
            per_constraint_string_prefix = "  q_"
            per_constraint_string_suffix = "(x,u) = "
            filename = "qfa_description.txt"
        else:
            string_out = "Approximate VF is max_i{g_i(x)}, where:\n"
            per_constraint_string_prefix = "  g_"
            per_constraint_string_suffix = "(x) = "
            filename = "vfa_description.txt"
        for c in self.alpha_c_in_model:
            function_string = c.print_constraint_function()
            string_out += per_constraint_string_prefix + str(c.id) + \
                per_constraint_string_suffix + function_string + ",\n"
        string_out = string_out[:-2]  # Cut off final comma and carriage return ",\n"
        print string_out
        if save:
            try:
                with open('output/' + self.s.name + '/' + filename, 'w') as text_file:
                    text_file.write(string_out)
            except IOError as e:
                print "Could not write QF approximation description to 'output/" + \
                      self.s.name + "/" + filename + "'."
                print e

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
                        for ind2, x2_plot in enumerate(x2):
                            u_mesh[ind1, ind2] = self.solve_for_xhat(np.array([x1_plot, x2_plot]),
                                                                     extract_constr=False)[0][0]
                elif self.n == 4:
                    for ind1, x1_plot in enumerate(x1):
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

        #
        # plt.figure()
        # plt.plot(xalg_tv_of_x, 100. * (xalg_tv_of_x - xalg_v_of_x) / xalg_v_of_x,
        #          linewidth=0., marker='x')
        #
        # if self.__class__.__name__ == "QFApproximator":
        #     plt.ylabel('$(T_QQ(x,u) - Q(x,u))/Q(x,u)$ (%)')
        #     plt.xlabel('$Q(x,u)$')
        # else:
        #     plt.ylabel('$(TV(x) - V(x))/V(x)$ (%)')
        #     plt.xlabel('$V(x)$')
        # plt.ylim([min(-1., np.min(100. * (xalg_tv_of_x - xalg_v_of_x) / xalg_v_of_x) * 1.05),
        #           min(100., np.max(100. * (xalg_tv_of_x - xalg_v_of_x) / xalg_v_of_x) * 1.05)])
        # plt.savefig('output/' + self.s.name + '/xalg_BE_by_norm.pdf')
        # plt.close()

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

        # CSV of XAlg VF approximation mean
        xalg_integral_csv_array = np.array(xalg_integral_results)
        try:
            np.savetxt(output_dir + '/xalg_lb_M' + str(nx) + '.csv', xalg_integral_csv_array,
                       fmt='%d,%.5f', delimiter=',', header='Iteration,Mean LB',
                       comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/xalg_lb_M' + str(nx) + '.csv'
            print e
        # CSV of XAlg Bellman converge nce
        xalg_bellman_csv_array = np.array(xalg_bellman_results)
        try:
            np.savetxt(output_dir + '/xalg_bellman_error_M' + str(nx) + '.csv',
                       xalg_bellman_csv_array,
                       fmt='%d,%.5f,%.5f,%.5f,%.5f', delimiter=',',
                       header='Iteration,Mean TV(x)-V(x),Mean (TV(x)-V(x))/V(x),Max TV(x)-V(x),Max (TV(x)-V(x))/V(x)', comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/xalg_bellman_error_M' + str(nx) + '.csv'
            print e
        # CSV of XAlg closed-loop cost upper bound
        xalg_cl_ub_csv_array = np.array(xalg_cl_ub_results)
        if xalg_cl_ub_csv_array.shape != (0,):
            try:
                np.savetxt(output_dir + '/xalg_cl_ub_M' + str(nx) + '.csv', xalg_cl_ub_csv_array,
                           fmt='%d,%.5f,%.5f', delimiter=',',
                           header='Iteration,Mean UB,Mean UB u=0', comments='')
            except Exception as e:
                print "Could not save " + output_dir + '/xalg_cl_ub_M' + str(nx) + '.csv'
                print e

        ind_integral_csv_array = np.array(ind_integral_results)
        try:
            np.savetxt(output_dir + '/ind_lb_M' + str(nx) + '.csv', ind_integral_csv_array,
                       fmt='%d,%.5f', delimiter=',', header='Iteration,Mean LB', comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/xalg_lb_M' + str(nx) + '.csv'
            print e
        # CSV of independent Bellman convergence
        ind_bellman_csv_array = np.array(ind_bellman_results)
        try:
            np.savetxt(output_dir + '/ind_bellman_error_M' + str(nx) + '.csv', ind_bellman_csv_array,
                       fmt='%d,%.5f,%.5f,%.5f,%.5f', delimiter=',',
                       header='Iteration,Mean TV(x)-V(x),Mean (TV(x)-V(x))/V(x),Max TV(x)-V(x),Max (TV(x)-V(x))/V(x)', comments='')
        except Exception as e:
            print "Could not save " + output_dir + '/ind_bellman_error_M' + str(nx) + '.csv'
            print e
        # CSV of independent closed-loop cost upper bound
        ind_cl_ub_csv_array = np.array(ind_cl_ub_results)
        if ind_cl_ub_csv_array.shape != (0,):
            try:
                np.savetxt(output_dir + '/ind_cl_ub_M' + str(nx) + '.csv', ind_cl_ub_csv_array,
                           fmt='%d,%.5f,%.5f', delimiter=',',
                           header='Iteration,Mean UB,Mean UB u=0', comments='')
            except Exception as e:
                print "Could not save " + output_dir + '/ind_cl_ub_M' + str(nx) + '.csv'
                print e

        # Save timing info
        pd.DataFrame([('LB computation time', self.lb_computation_time),
                      ('Total iteration time', iter_time),
                      ('One-stage problems solved', osp)]).to_csv(output_dir + '/comp_time.csv',
                                                                  header=['Category', 'Time'],
                                                                  index=None, float_format='%.4f')

        # Plot LB and UB convergence
        plt.figure(figsize=(8, 16))
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
        plt.xlabel('Iteration number $I$')
        plt.ylabel('No. of constraints')
        plt.title('Number of $\\alpha$ lower-bounding constraints $g_i(\cdot)$')
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

    def approximate(self, strategy_in=None, audit_in=None, outputs_in=None):
        """
        Create an iterative approximation of the value function for the system self.s

        :param strategy_in: Dictionary of parameters describing the solution strategy
        :param audit_in: Dict of parameters describing how progress should be tracked during solution
        :param outputs_in: Dict of parameters determining outputs produced
        :return: Convergence data structure
        """

        # Update approximation parameters to reflect user-specified settings
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

        print "Done in %.1f seconds." % total_time
        # Time breakdown commented out, because it doesn't always add up properly.
        # print "  %.1f s spent computing lower bounds," % self.lb_computation_time
        # print "  %.1f s spent measuring V integral," % self.v_integral_eval_time
        # print "  %.1f s spent auditing Bellman gap," % self.b_gap_eval_time
        # print "  and %.1f s elsewhere." % (total_time - self.v_integral_eval_time -
        #                                    self.b_gap_eval_time - self.lb_computation_time)

        return convergence_data
