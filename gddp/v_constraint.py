import numpy as np
import copy
import os
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.optimize import minimize
import gurobipy as gu


class Constraint(object):
    """Lower-bounding constraint for value function under-approximation"""

    def __init__(self, n, id_in, const_in=None, lin_in=None, hessian_in=None,
                 sys_in=None, duals_in=None, constrs_in=None, xhat_in=None, xplus_opt_in=None,
                 gridded_values=None):
        """Initialize Constraint object. This can be initialized in one of several forms:
        - Constant, affine, or quadratic, in which case const_in, lin_in and/or hessian_in should
            be defined
        - Dual variable definition, in which case duals_in and constrs_in should be defined so that
            all components of the dual-variable constraint can be calculated
        - Gridded, in which case the gridded_values object should be supplied

        Two properties of the constraint affect whether it will be used in value function approx-
        imation:
        - dodgy: Constraint definition is based on a one-stage optimization that didn't return a
        clean dual result
        - worthless: Constraint does not increase the value function estimate by the required
        tolerance, and is therefore not deemed worth adding to the model

        :param n: Integer state dimension
        :param id_in: Integer constraint ID
        :param const_in: Constant component of constraint function
        :param lin_in: Linear component of constraint function
        :param hessian_in: Hessian of constraint function (if quadratic)
        :param sys_in: System object to which constraint relates
        :param duals_in: Dual variables by which constraint is defined
        :param constrs_in: Other constraints by which the value of constant zeta_1 can be computed
        :param xhat_in: State parameter at which constraint was extracted from the one-stage problem
        :param xplus_opt_in: The optimal xplus from the one-stage problem
        :param gridded_values: scipy.interpolate.RectBivariateSpline object: Gridded constraint vals
        """
        self.n = n
        self.m = sys_in.m if sys_in is not None else 1
        self.id = id_in
        self.in_model = False

        self.const = const_in if const_in is not None else 0.
        self.xplus_opt_in = xplus_opt_in

        if lin_in is not None:
            assert lin_in.shape[0] == self.n
            self.lin = lin_in
        else:
            self.lin = np.zeros(self.n, dtype=float)
        if hessian_in is not None:
            assert hessian_in.shape == (self.n, self.n)
            self.hessian = hessian_in
            self.no_hessian = False
        else:
            self.hessian = np.zeros((self.n, self.n), dtype=float)
            self.no_hessian = True

        self.s = sys_in
        if duals_in is not None and constrs_in is not None:
            self.lambda_a_hat, self.lambda_b_hat, self.lambda_c_hat, self.pi_hat = duals_in
            # self.lambda_a_hat, self.lambda_b_hat, self.lambda_c_hat, self.pi_hat = self.fix_duals(duals_in)
            assert len(self.lambda_a_hat) == len(constrs_in)
            self.lambdas_and_cs = [(la, constrs_in[i])
                                   for i, la in enumerate(self.lambda_a_hat) if la > 0]
            self.xhat = copy.copy(xhat_in)
            self.grid = None
            self.qp_compatible = False
        elif gridded_values is not None:
            self.lambda_a_hat, self.lambda_b_hat, self.lambda_c_hat, self.pi_hat = None, None, None, None
            self.lambdas_and_cs = None
            self.xhat = None
            self.grid = gridded_values  # scipy.interpolate.RectBivariateSpline object
            self.qp_compatible = False
        else:
            self.lambda_a_hat, self.lambda_b_hat, self.lambda_c_hat, self.pi_hat = None, None, None, None
            self.lambdas_and_cs = None
            self.xhat = None
            self.grid = None
            self.qp_compatible = True

        self.region_membership, self.region_A, self.region_b = None, None, None

        self.zeta_1_value = None
        # Set the below to True if defined using erroneous dual variables
        self.dodgy = False
        # Set the below to True if the new constraint doesn't increase the VF estimate when made
        self.worthless = False

    def assign_regions(self, region_id, A=None, b=None):
        self.region_membership = region_id  # Should be in form [region_1, region_2, ...]
        self.region_A = A
        self.region_b = b

    def fix_duals(self, duals_in):
        # Minimize (lambda - lambda_raw)^2 subject to feasibility constraints "fixing" the solution.

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
        m_a.setObjective(qe + le)
        m_a.update()
        m_a.addConstr(gu.LinExpr([(1.0, var) for var in x]) == self.s.gamma)
        m_a.optimize()
        if m_a.status == 2:
            sol_vec = [v.x for v in m_a.getVars()]
            lambda_a_fixed = np.array(sol_vec)
        else:
            print "Fixing duals in constraint " + self.id + "failed! Exiting."
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
            print "Fixing duals in constraint " + self.id + "failed! Exiting."
            raise SystemExit()

        assert np.linalg.norm(lambda_a_fixed - lambda_a_raw) <= 1e-4, str(lambda_a_fixed) + str(lambda_a_raw)
        assert np.linalg.norm(lambda_b_fixed - lambda_b_raw) <= 1e-4, str(lambda_b_fixed) + str(lambda_b_raw)

        return lambda_a_fixed, lambda_b_fixed, lambda_c_raw, pi_raw

    def update_coeffs(self, const_in=None, lin_in=None, hessian_in=None):
        """For const + linear + quadratic models, update one or more of these terms"""
        if lin_in is not None:
            assert lin_in.shape[0] == self.n
        if hessian_in is not None:
            assert hessian_in.shape == (self.n, self.n)
        self.const = const_in
        self.lin = lin_in
        self.hessian = hessian_in

    def x_plus_lagrangian(self, x_plus):
        """Return the component of the Lagrangian of the one-stage problem that relates to x_plus.
        Minimizing this component over R^n yields the constant term zeta_1 of the constraint
        definition. The function depends on the dual variables pi_hat and lambda_hat.

        :param x_plus: x_plus to be used in evaluating the Lagrangian
        :return: Value of this component of the Lagrangian at x_plus
        """
        return (-np.dot(self.pi_hat, x_plus) +
                np.sum([lac[0] * lac[1].eval_at(x_plus) for lac in self.lambdas_and_cs]))

    def x_plus_lagrangian_grad(self, x_plus):
        """Return the gradient of the component of the Lagrangian of the one-stage problem that
        relates to x_plus

        :param x_plus: x_plus to be used in the evaluation
        :return: Gradient of this component of the Lagrangian at x_plus
        """
        return -np.array(self.pi_hat) + np.sum([lac[0] * lac[1].grad_at(x_plus)
                                                for lac in self.lambdas_and_cs])

    def zeta_1(self, x_plus_initial_guess, x_dependent=False, value_to_beat=None):
        """Evaluate the zeta_1 component of the constraint definition, which comes from minimizing
        a part of the Lagrangian of the one-stage problem over x_plus.

        :param x_plus_initial_guess: Initial guess of optimal x_plus
        :param x_dependent: Whether zeta_1 should be x-dependent (in which case it has to be
            evaluated for every x where its value needs to be known)
        :param value_to_beat: Value below which constraint will not be worth adding (other
            components of the constraint value have already been calculated), since it will not
            exceed the existing value function approximation at xhat.
        :return: zeta_1
        """
        if not x_dependent:
            if self.zeta_1_value is None:
                if self.s.name == 'Bead on beam':
                    bounds = [(-3.2, 3.2), (-2.0, 2.0), (-1.6, 1.6), (-2.0, 2.0)]
                else:
                    bounds = [(None, None)] * self.n
                print "  Calculating zeta_1 for constraint " + str(self.id) + \
                      ". %d active constraints:" % len(self.lambdas_and_cs)
                for lac in self.lambdas_and_cs:
                    print "    Constraint " + str(lac[1].id) + ": alpha = %.5f" % lac[0]
                # delta_1_with_x_dependence = self.delta_1(x, pi, lambda_a, x_dependent=True)
                x_first_guess = self.xplus_opt_in if self.xplus_opt_in is not None \
                    else x_plus_initial_guess
                min_result = minimize(fun=self.x_plus_lagrangian,
                                      x0=x_first_guess,
                                      jac=self.x_plus_lagrangian_grad,
                                      method='L-BFGS-B',
                                      bounds=bounds)
                                      # options={'maxiter': 200})
                self.zeta_1_value = min_result.fun

                n = 0  # Try number for local search
                try_limit = 100
                scale_factor = 1.0
                while n < try_limit * scale_factor and self.zeta_1_value > -1e10 and n <= 1000:
                    n += 1
                    # x_rand_init = self.s.state_mean + np.dot(np.sqrt(self.s.state_var * 2),
                    #                                          np.random.randn(self.n))
                    # x_rand_init = np.clip(x_rand_init, [b[0] for b in bounds],
                    #                       [b[1] for b in bounds])
                    x_rand_init = np.random.uniform(low=[b[0] for b in bounds],
                                                    high=[b[1] for b in bounds], size=(self.n,))
                    min_result = minimize(fun=self.x_plus_lagrangian,
                                          x0=x_rand_init,
                                          jac=self.x_plus_lagrangian_grad,
                                          method='L-BFGS-B',
                                          bounds=bounds)
                                          # options={'maxiter': 200})
                    # print "    zeta_1 evaluation success report:", min_result.success
                    # print "    zeta_1 evaluation:", min_result.message

                    if min_result.fun < self.zeta_1_value - 1e-5:
                        self.zeta_1_value = min_result.fun
                        print "    zeta_1 try %d: Found a better value, %.5f" % (n, self.zeta_1_value)
                        if n > 0.5 * try_limit * scale_factor:
                            # Extend number of global search starts if it found a late better value
                            # Also increase variance of search start locations ("expand domain")
                            scale_factor *= 2
                    if value_to_beat is not None and self.zeta_1_value <= value_to_beat - 1e-3:
                        print "    Value of zeta_1 is too low to beat existing VFA at this x."
                        break
                if n >= 1000:
                    self.zeta_1_value = -np.inf  # Set to -infinity if too many bounds were tested
                print "    zeta_1 value without x dependence: %.5f" % self.zeta_1_value
                # print "    zeta_1 value with x dependence: %.5f" % delta_1_with_x_dependence

            return self.zeta_1_value
        else:
            if self.m == 1:
                u_min = -self.s.h[0]
                u_max = self.s.h[1]

                bisection_tol = 1e-2
                a_n = copy.copy(u_min)
                b_n = copy.copy(u_max)
                rho = 0.5 * (np.sqrt(5.) + 1)

                c_n = b_n - (b_n - a_n) / rho
                d_n = a_n + (b_n - a_n) / rho
                while abs(c_n - d_n) > bisection_tol:
                    x_plus_c = self.s.get_x_plus(x_plus_initial_guess, np.array([c_n]))
                    x_plus_d = self.s.get_x_plus(x_plus_initial_guess, np.array([d_n]))
                    c_cost = (-np.dot(self.pi_hat, x_plus_c) +
                              np.sum([lac[0] * lac[1].eval_at(x_plus_c)
                                      for lac in self.lambdas_and_cs]))
                    d_cost = (-np.dot(self.pi_hat, x_plus_d) +
                              np.sum([lac[0] * lac[1].eval_at(x_plus_d)
                                      for lac in self.lambdas_and_cs]))
                    if c_cost < d_cost:
                        b_n = d_n
                    else:
                        a_n = c_n
                    c_n = b_n - (b_n - a_n) / rho
                    d_n = a_n + (b_n - a_n) / rho
                u_opt = np.array([(a_n + b_n) / 2.])
                x_plus_opt = self.s.get_x_plus(x_plus_initial_guess, u_opt)
                return (-np.dot(self.pi_hat, x_plus_opt) +
                        np.sum([lac[0] * lac[1].eval_at(x_plus_opt) for lac in self.lambdas_and_cs]))

                # grid_res = 0.1
                # u_range = np.arange(u_min,
                #                     u_max + grid_res,
                #                     grid_res)
                # n_u_grid = len(u_range)
                # delta_1_grid = np.array([-np.inf] * n_u_grid, dtype=float)
                #
                # for i, u in enumerate(u_range):
                #     x_plus = self.s.get_x_plus(x, np.array([u]))
                #     delta_1_grid[i] = (-np.dot(pi, x_plus) +
                #                        np.sum([lac[0] * lac[1].eval_at(x_plus) for lac in
                #                                self.lambdas_and_cs]))
                #
                # min_u_index = np.argmin(delta_1_grid)
                #
                # # plt.figure()
                # # plt.plot(u_range, delta_1_grid)
                # # plt.plot(u_range[min_u_index], delta_1_grid[min_u_index], 'ro')
                # # plt.ylim([np.min(delta_1_grid) - 0.02, np.max(delta_1_grid) + 0.02])
                # # plt.xlabel('$u$')
                # # plt.ylabel('UB on $\delta_1(\\pi, \\lambda_\\alpha$)')
                # # plt.title('Minimization for $\\delta_1(\\pi, \\lambda_\\alpha$), iter. %d' % iter_no)
                # # plt.savefig(output_dir + '/delta_1_min_%d.pdf' % iter_no)
                # # plt.close()
                #
                # return delta_1_grid[min_u_index]
            else:
                print "Cannot evaluate zeta_1(pi, alpha) for input dimension greater than 1!"

    def zeta_2(self, x, pi, lambda_c, lambda_b):
        """Evaluate zeta_2 in closed analytic form for the case where input cost Hessian R > 0."""
        vec = np.dot(pi, self.s.f_u_x(x)) + \
              np.dot(lambda_c, self.s.E) + \
              np.sum(lambda_b[i] * self.s.ri[i] for i in range(self.s.n_beta_c))
        assert vec.shape == (self.s.ri[0].shape[0],)
        mat = np.sum([l_b * self.s.Ri[i] for i, l_b in enumerate(lambda_b)], keepdims=True)
        if len(mat.shape) == 3:
            mat = mat[0]
        assert len(mat.shape) == 2, "Matrix in delta_2 computation has wrong shape: " + str(mat)
        return -0.5 * np.dot(vec, np.dot(np.linalg.inv(mat), vec))

    def eval_at(self, x_in, exclude_zeta_1=False):
        # Evaluate the function at a particular value of x
        # assert x_in.shape == (self.n,)

        if self.region_membership is not None:
            if np.any(np.dot(self.region_A, x_in) - self.region_b > 1e-8):
                return 0.

        if not self.qp_compatible:
            if self.grid is not None:
                return self.grid(x_in, method='linear')[0]
            else:
                if exclude_zeta_1:
                    return (self.const
                            + np.sum([self.s.phi_i_x(l, x_in) * self.lambda_b_hat[l]
                                      for l in range(self.s.n_beta_c)])
                            - np.dot(self.s.h_x(x_in), self.lambda_c_hat)
                            + np.dot(self.s.f_x_x(x_in), self.pi_hat)
                            + self.zeta_2(x_in, self.pi_hat, self.lambda_c_hat, self.lambda_b_hat))
                else:
                    return (self.const
                            + np.sum([self.s.phi_i_x(l, x_in) * self.lambda_b_hat[l]
                                      for l in range(self.s.n_beta_c)])
                            - np.dot(self.s.h_x(x_in), self.lambda_c_hat)
                            + np.dot(self.s.f_x_x(x_in), self.pi_hat)
                            + self.zeta_1(x_in)
                            + self.zeta_2(x_in, self.pi_hat, self.lambda_c_hat, self.lambda_b_hat))
        else:
            if self.no_hessian:
                return self.const + np.dot(self.lin, x_in)
            else:
                return (self.const + np.dot(self.lin, x_in) +
                        0.5 * np.dot(x_in, np.dot(self.hessian, x_in)))

    def grad_at(self, x_in, dx=0.0001):
        """Evaluate the gradient of the constraint by evaluating at x_in and at points offset from
        x_in by dx in each coordinate

        :param x_in: n-element vector where constraint gradient should be evaluated
        :param dx: Scalar << 1 determining how far from x_in the constraint should be evaluated
        :return: n-element vector of constraint gradient
        """
        assert x_in.shape == (self.n,)
        # direct = None
        if self.s is not None and self.s.name == 'Bead on beam':
            # Produce analytical gradient if we are considering the bead-on-beam example
            m, dt, g, J = self.s.physics['M'], self.s.physics['dt'], self.s.physics['g'], self.s.physics['J']
            x1, x2, x3, x4 = x_in[0], x_in[1], x_in[2], x_in[3]
            pi1, pi2, pi3, pi4 = self.pi_hat[0], self.pi_hat[1], self.pi_hat[2], self.pi_hat[3]
            q11, q22, q33, q44 = self.s.phi_i_x_quad[0][0, 0], self.s.phi_i_x_quad[0][1, 1], self.s.phi_i_x_quad[0][2, 2], self.s.phi_i_x_quad[0][3, 3]
            r = self.s.Ri[0][0, 0]
            den = m * x1 * x1 + J
            return np.array([q11 * x1
                             + pi1 + pi2 * x4 * x4 * dt
                             - pi4 * dt * (-2 * m * x1 * (2 * m * x1 * x2 + m * g * x1 * np.cos(x3))
                                           + (2 * m * x2 + m * g * np.cos(x3)) * den) / (den ** 2)
                             + 2 * r * pi4 * m * x1 * dt * np.power(den, -2) *
                             (pi4 * dt / den + np.dot(self.s.E.T, self.lambda_c_hat)[0]),
                             q22 * x2 + pi1 * dt + pi2 - pi4 * (2 * m * x1 * dt) / den,
                             q33 * x3 - pi2 * g * np.cos(x3) * dt
                             + pi3 + (pi4 * m * g * x1 * np.sin(x3) * dt) / den,
                             q44 * x4 + 2 * pi2 * x1 * x4 * dt + pi3 * dt + pi4])

        # Otherwise evaluate the gradient by perturbing in each coordinate
        grad_out = np.zeros((self.n,), dtype=float)
        gx = self.eval_at(x_in)
        for i in range(self.n):
            delta_x = np.zeros((self.n,), dtype=float)
            delta_x[i] = dx
            grad_out[i] = (self.eval_at(x_in + delta_x) - gx) / dx

        # Reconcile analytical and perturbation results in case of bead-on-beam
        # if np.random.rand() < 0.001:
        #     print "direct:", direct
        #     print "grad_out:", grad_out
        #     if self.s is not None:
        #         print "direct x1 components:"
        #         print q11 * x1, pi1, pi2 * x4 * x4 * dt
        #         print -pi4 * dt * (-2 * m * x1 * (2 * m * x1 * x2 + m * g * x1 * np.cos(x3)) +
        #                                (2 * m * x2 + m * g * np.cos(x3)) * den) / (den ** 2)
        #         print 2 * r * pi4 * m * x1 * dt * np.power(den, -2) * (pi4 * dt / den + np.dot(self.s.E.T, self.lambda_c_hat)[0])
        #         print self.lambda_c_hat
        #     raw_input('')

        return grad_out

    def plot_function(self, output_dir, iter_no):
        """Plot constraint function in two dimensions. Coordinate ranges and discretization step
        are currently hard-coded.

        :param output_dir: Output directory
        :param iter_no: Iteration number for labelling the constraint plot
        :return: nothing
        """
        if self.n == 2:
            x1_min, x1_max, res1 = -3.2, 3.2, 0.2
            x2_min, x2_max, res2 = -3.2, 3.2, 0.2
            x1 = np.arange(x1_min, x1_max + res1, res1).tolist()
            x2 = np.arange(x2_min, x2_max + res2, res2).tolist()
            mesh = np.zeros((len(x1), len(x2)), dtype=float)

            for ind1, x1_plot in enumerate(x1):
                for ind2, x2_plot in enumerate(x2):
                    mesh[ind1, ind2] = self.eval_at(np.array([x1_plot, x2_plot]))

            X, Y = np.meshgrid(x1, x2)
            savemat(output_dir + '/gx_iter%d_id%d.pdf' % (iter_no, self.id),
                    {'X': X, 'Y': Y, 'mesh': mesh})

            # Axes3D.plot_wireframe(X, Y, mesh)
            plt.imshow(mesh.T, aspect='equal', origin='lower',
                       extent=[x1_min, x1_max, x2_min, x2_max], vmin=0, vmax=15)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            if iter_no is not None:
                plt.title('Value function lower bound $g_{%d}(x)$, iter. %d ' % (self.id, iter_no))
            else:
                plt.title('Value function approximation')
        else:
            print "Can only plot lower-bounding constraint for n = 2!"

        # Save plot
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '/gx_iter%d_id%d.pdf' % (iter_no, self.id))
        plt.close()

    def added_to_model(self):
        # Mark the constraint as added to the optimization model
        assert self.in_model is False, "Constraint " + str(self.id) + " already in model!"
        self.in_model = True

    def removed_from_model(self):
        # Mark the constraint as removed from the optimization model
        assert self.in_model is True, "Constraint " + str(self.id) + " not in model!"
        self.in_model = False
