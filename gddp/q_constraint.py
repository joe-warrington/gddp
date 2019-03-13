import numpy as np
import copy
import os
from matplotlib import pyplot as plt
from scipy.io import savemat
import gurobipy as gu

x1_min_plot_default, x1_max_plot_default, res1_plot_default = -3, 3, 0.2
u1_min_plot_default, u1_max_plot_default, res2_plot_default = -3, 3, 0.2


class QConstraint(object):
    """Lower-bounding constraint for value function under-approximation"""

    def __init__(self, n, m, id_in, const_in=None, x_lin_in=None, x_hessian_in=None,
                 u_lin_in=None, u_hessian_in=None,
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
        :param x_lin_in: Linear component of constraint function
        :param x_hessian_in: Hessian of constraint function (if quadratic)
        :param sys_in: System object to which constraint relates
        :param duals_in: Dual variables by which constraint is defined
        :param constrs_in: Other constraints by which the value of constant zeta_1 can be computed
        :param xhat_in: State parameter at which constraint was extracted from the one-stage problem
        :param xplus_opt_in: The optimal xplus from the one-stage problem
        :param gridded_values: scipy.interpolate.RectBivariateSpline object: Gridded constraint vals
        """
        self.n = n
        self.m = m
        self.id = id_in
        self.in_model = False

        self.const = const_in if const_in is not None else 0.
        self.xplus_opt_in = xplus_opt_in

        if x_lin_in is not None:
            assert x_lin_in.shape == (self.n, )
            self.x_lin = x_lin_in
        else:
            self.x_lin = np.zeros(self.n, dtype=float)
        if u_lin_in is not None:
            assert u_lin_in.shape == (self.m, ), "Shape of u_lin_in is " + repr(u_lin_in.shape)
            self.u_lin = u_lin_in
        else:
            self.u_lin = np.zeros(self.m, dtype=float)
        if x_hessian_in is not None:
            assert x_hessian_in.shape == (self.n, self.n)
            assert u_hessian_in.shape == (self.m, self.m)
            self.x_hessian = x_hessian_in
            self.u_hessian = u_hessian_in
            self.no_hessian = False
        else:
            self.x_hessian = None
            self.u_hessian = None
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

    def print_constraint_function(self):
        """Prints constraint function, bearing in mind that q-functions are stored *without* the
        stage cost for efficiency. Therefore the print function adds the stage cost back in.

        :return: Nothing
        """
        string_out = ""
        if self.const is not None and np.abs(self.const) > 1e-6:
            string_out += "%.4f + " % self.const
        if self.x_lin is not None or (self.s is not None and self.s.phi_i_x_lin is not None):
            for i in range(self.n):
                coeff = 0
                if self.x_lin is not None:
                    coeff += self.x_lin[i]
                if self.s is not None and self.s.phi_i_x_lin is not None:
                    coeff += self.s.phi_i_x_lin[0][i]
                if np.abs(coeff) > 1e-6:
                    string_out += "%.4f x%d + " % (coeff, i+1)
        if self.x_hessian is not None or (self.s is not None and isinstance(self.s.phi_i_x_quad, tuple)):
            for i in range(self.n):
                for j in range(i, self.n):
                    coeff = 0
                    if self.x_hessian is not None:
                        coeff += self.x_hessian[i, j]
                    if self.s is not None and isinstance(self.s.phi_i_x_quad, tuple):
                        coeff += self.s.phi_i_x_quad[0][i, j]
                    if i == j and coeff != 0:
                        string_out += "%.4f x%d^2 + " % (0.5 * coeff, i+1)
                    elif coeff != 0.0:
                        # Combine the separate terms in, for example, x1x2 and x2x1, which means
                        # doubling the (i, j) entry to get H(i, j) + H(j, i)
                        string_out += "%.4f x%dx%d + " % (0.5 * (2 * coeff), i+1, j+1)
        if self.u_lin is not None or (self.s is not None and self.s.ri is not None):
            for i in range(self.m):
                coeff = 0
                if self.u_lin is not None:
                    coeff += self.u_lin[i]
                if self.s is not None and self.s.ri is not None:
                    coeff += self.s.ri[0][i]
                if np.abs(coeff) > 1e-6:
                    string_out += "%.4f u%d + " % (coeff, i+1)
        if self.u_hessian is not None or (self.s is not None and isinstance(self.s.Ri, tuple)):
            for i in range(self.m):
                for j in range(i, self.m):
                    coeff = 0
                    if self.u_hessian is not None:
                        coeff += self.u_hessian[i, j]
                    if self.s is not None and isinstance(self.s.Ri, tuple):
                        coeff += self.s.Ri[0][i, j]
                    if i == j and coeff != 0:
                        string_out += "%.4f u%d^2 + " % (0.5 * coeff, i+1)
                    elif coeff != 0.0:
                        # Combine the separate terms in, for example, x1x2 and x2x1, which means
                        # doubling the (i, j) entry to get H(i, j) + H(j, i)
                        string_out += "%.4f u%dx%d + " % (0.5 * (2 * coeff), i+1, j+1)
        if len(string_out) >= 3 and string_out[-3:] == " + ":
            string_out = string_out[:-3]  # Trim trailing + sign and spaces
        if string_out == "":
            string_out = "0"
        return string_out

    def repair_duals(self, duals_in):
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

    def eval_at(self, x_in, u_in):
        # Evaluate the function at a particular value of (x, u)
        assert x_in.shape == (self.n,)
        assert u_in.shape == (self.m,)

        if self.no_hessian:
            return self.const + np.dot(self.x_lin, x_in) + np.dot(self.u_lin, u_in)
        else:
            return (self.const + np.dot(self.x_lin, x_in) + np.dot(self.u_lin, u_in) +
                    0.5 * np.dot(x_in, np.dot(self.x_hessian, x_in)) +
                    0.5 * np.dot(u_in, np.dot(self.u_hessian, u_in)))

    def plot_function(self, output_dir, iter_no):
        """Plot constraint function in two dimensions. Coordinate ranges and discretization step
        are currently hard-coded.

        :param output_dir: Output directory
        :param iter_no: Iteration number for labelling the constraint plot
        :return: nothing
        """
        if self.n == 1 and self.m == 1:
            x1_min, x1_max, res1 = x1_min_plot_default, x1_max_plot_default, res1_plot_default
            u1_min, u1_max, res2 = u1_min_plot_default, u1_max_plot_default, res2_plot_default
            x1 = np.arange(x1_min, x1_max + res1, res1).tolist()
            u1 = np.arange(u1_min, u1_max + res2, res2).tolist()
            mesh = np.zeros((len(x1), len(u1)), dtype=float)

            for ind1, x_plot in enumerate(x1):
                for ind2, u_plot in enumerate(u1):
                    # Fill mesh point with QConstraint evaluated at x,u plus system stage cost,
                    # which for efficiency is not stored as part of the constraint function.
                    mesh[ind1, ind2] = self.eval_at(np.array([x_plot]), np.array([u_plot])) + \
                        self.s.quad_stage_cost(np.array([x_plot]), np.array([u_plot]))

            X, Y = np.meshgrid(x1, x2)
            savemat(output_dir + '/gx_iter%d_id%d.pdf' % (iter_no, self.id),
                    {'X': X, 'Y': Y, 'mesh': mesh})

            # Axes3D.plot_wireframe(X, Y, mesh)
            plt.imshow(mesh.T, aspect='equal', origin='lower',
                       extent=[x1_min, x1_max, u1_min, u1_max], vmin=0, vmax=15)
            plt.xlabel('$x_1$')
            plt.ylabel('$u_1$')
            if iter_no is not None:
                plt.title('Q-function lower bound $q_{%d}(x)$, iter. %d ' % (self.id, iter_no))
            else:
                plt.title('Approximate Q-function')
        else:
            print "Can only plot lower-bounding constraint for n = 1, m = 1!"

        # Save plot
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '/qxu_iter%d_id%d.pdf' % (iter_no, self.id))
        plt.close()

    def added_to_model(self):
        # Mark the constraint as added to the optimization model
        assert self.in_model is False, "Constraint " + str(self.id) + " already in model!"
        self.in_model = True

    def removed_from_model(self):
        # Mark the constraint as removed from the optimization model
        assert self.in_model is True, "Constraint " + str(self.id) + " not in model!"
        self.in_model = False
