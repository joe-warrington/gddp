import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.io import savemat
import gurobipy as gu

x1_min_plot_default, x1_max_plot_default, res1_plot_default = -3, 3, 0.2
x2_min_plot_default, x2_max_plot_default, res2_plot_default = -3, 3, 0.2


class VConstraint(object):
    """Lower-bounding constraint for value function under-approximation"""

    def __init__(self, n, id_in, const_in=None, lin_in=None, hessian_in=None,
                 sys_in=None, duals_in=None):
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
        """

        self.n = n
        self.m = sys_in.m if sys_in is not None else 1
        self.id = id_in
        self.in_model = False

        self.const = const_in if const_in is not None else 0.

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
        if duals_in is not None:
            print "Can't create constraints from dual variables alone! Exiting."
            raise SystemExit()
        else:
            self.lambda_a_hat, self.lambda_b_hat, self.lambda_c_hat, self.pi_hat = None, None, None, None
            self.lambdas_and_cs = None
            self.xhat = None
            self.grid = None
            self.qp_compatible = True

        # Set the below to True if defined using erroneous dual variables
        self.dodgy = False
        # Set the below to True if the new constraint doesn't increase the VF estimate when made
        self.worthless = False

    def print_constraint_function(self):
        string_out = ""
        if self.const is not None and np.abs(self.const) > 1e-6:
            string_out += "%.4f + " % self.const
        if self.lin is not None and np.any(self.lin != np.zeros((self.n,))):
            for i in range(self.n):
                if np.abs(self.lin[i]) > 1e-6:
                    string_out += "%.4f x%d + " % (self.lin[i], i+1)
        if self.hessian is not None:
            for i in range(self.n):
                for j in range(i, self.n):
                    if i == j and self.hessian[i, i] != 0:
                        string_out += "%.4f x%d^2 + " % (0.5 * (self.hessian[i, i]), i+1)
                    elif self.hessian[i, j] + self.hessian[j, i] != 0.0:
                        # Combine the separate terms in, for example, x1x2 and x2x1
                        string_out += "%.4f x%dx%d + " % (0.5 * (self.hessian[i, j] +
                                                                 self.hessian[j, i]), (i+1), (j+1))
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

    def update_coeffs(self, const_in=None, lin_in=None, hessian_in=None):
        """For const + linear + quadratic models, update one or more of these terms"""
        if lin_in is not None:
            assert lin_in.shape[0] == self.n
        if hessian_in is not None:
            assert hessian_in.shape == (self.n, self.n)
        self.const = const_in
        self.lin = lin_in
        self.hessian = hessian_in

    def eval_at(self, x_in):
        # Evaluate the function at a particular value of x
        assert x_in.shape == (self.n,)

        if self.qp_compatible:
            if self.no_hessian:
                return self.const + np.dot(self.lin, x_in)
            else:
                return (self.const + np.dot(self.lin, x_in) +
                        0.5 * np.dot(x_in, np.dot(self.hessian, x_in)))
        else:
            print "Cannot evaluate constraint function for constraints not expressed as quadratic."
            raise SystemExit()

    def plot_function(self, output_dir, iter_no):
        """Plot constraint function in two dimensions. Coordinate ranges and discretization step
        are currently hard-coded.

        :param output_dir: Output directory
        :param iter_no: Iteration number for labelling the constraint plot
        :return: nothing
        """
        if self.n == 2:
            x1_min, x1_max, res1 = x1_min_plot_default, x1_max_plot_default, res1_plot_default
            x2_min, x2_max, res2 = x2_min_plot_default, x2_max_plot_default, res2_plot_default
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
