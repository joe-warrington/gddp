import numpy as np
import copy
from matplotlib import pyplot as plt
from scipy.io import savemat, loadmat
from scipy.linalg import solve_discrete_are, qr
from copy import deepcopy
import os

np.set_printoptions(precision=5)


class System(object):
    """System specification. Can also simulate system with a given controller (based on a value
    function approximation)
    """

    def __init__(self, name=None, preset_system=None, model_dict=None):
        """Initialize the system by specifying its dynamics, costs, and flags determining how the
        one-stage problems for this system should be solved

        :param name: System name. Determines the name of the output directory created
        :param model_dict: Dictionary of model parameters; needed if the system name is not one of
        the particular cases named below.
        """
        if name is None and preset_system is None and model_dict is None:
            print "No system name, preset model, or parameter dictionary specified!"
            raise SystemExit()

        # Infer name of system, either from explicit name or inherited from preset_system
        if isinstance(name, str):
            self.name = name
        elif name is None and isinstance(preset_system, str):
            self.name = preset_system
        else:
            print "System not given a name! Exiting."
            raise SystemExit()

        if preset_system == "1D":
            print "Loading preset model '1D' and disregarding model_dict"
            self.m = 1  # Input dimension
            self.n = 1  # State dimension
            self.gamma = 1.00  # Discount factor
            self.A_mat = np.array([[0.9]], dtype=float)  # A matrix if linear system
            self.B_mat = np.array([[1.]], dtype=float)  # B matrix if linear system

            # State-input constraint of the form Dx + Eu <= h
            self.D = np.array([[0.], [0.]], dtype=float)
            self.E = np.array([[-1.], [1.]], dtype=float)
            max_u = 5  # Maximum control input magnitude
            self.h = np.array([max_u, max_u], dtype=float)

            self.n_beta = 1  # Length of beta vector for epigraph of stage costs
            self.li = (np.array([1.], dtype=float), )  # selector row matrix for cost epigraph
            self.phi_i_x_const = (0.,)  # Const component of state cost
            self.phi_i_x_lin = (np.array([0.]), )  # Linear component of state cost
            self.phi_i_x_quad = (np.array([[1.]]), )  # 2 * Hessian of state cost
            self.Ri = (np.array([[1.]], dtype=float),)  # 2 * input cost Hessian
            self.ri = (np.array([0.], dtype=float), )  # input cost linear component
            self.ti = (0., )  # constant cost component
            self.state_mean = np.array([0.], dtype=float)  # Mean of random state samples
            self.state_var = np.array([[1.]], dtype=float)  # Variance matrix of state samples
            self.pure_lqr = True  # Whether a pure LQR solution is available
            self.brute_force_solve = False
        elif preset_system == "Double integrator":
            print "Loading preset model 'Double integrator' and disregarding model_dict"
            self.m = 1
            self.n = 2
            self.gamma = 0.95
            tau = 0.1
            self.A_mat = np.array([[1.0, 1.0 * tau],
                                   [0.0, 1.0, ]], dtype=float)
            self.B_mat = np.array([[0.], [1.]], dtype=float)
            self.D = np.array([[0., 0.], [0., 0.]], dtype=float)  # Dx + Eu <= h
            self.E = np.array([[-1.], [1.]], dtype=float)
            min_u, max_u = -1.0, 1.0
            self.h = np.array([-min_u, max_u], dtype=float)
            self.n_beta = 1
            self.li = (np.array([1.], dtype=float),
                       np.array([1.], dtype=float))  # selector row matrix for cost epigraph
            self.phi_i_x_const = (0., -10.)
            self.phi_i_x_lin = (np.array([0., 0.]), np.array([0., 0.]))
            self.phi_i_x_quad = (np.array([[1., 0.], [0., 0.]]),
                                 np.array([[10., 0.], [0., 0.]]))
            self.Ri = (np.array([[0.5]], dtype=float),
                       np.array([[0.5]], dtype=float),)  # 2 * input cost Hessian
            self.ri = (np.array([0.], dtype=float),
                       np.array([0.], dtype=float))  # input cost linear component
            self.ti = (0., 0.)  # constant cost component
            self.state_mean = np.zeros((self.n,), dtype=float)
            self.state_var = 4.0 * np.eye(self.n, dtype=float)
            self.pure_lqr = False
            self.brute_force_solve = False
        elif preset_system == "8 states 3 inputs":
            print "Loading preset model '8states3inputs' and disregarding model_dict"
            self.m = 3
            self.n = 8
            self.gamma = 0.95
            # A and B matrices randomly generated in Matlab and pasted in.
            self.A_mat = np.array([[-0.12596404, -0.23946429, -0.01788977, -0.07828756, -0.08912436,
                                    0.1012664,  0.39073598,  0.28687528],
                                   [-0.14099204,  0.03634655, -0.05263492, -0.0632178, -0.31189514,
                                    0.03166916, -0.08866424,  0.43571461],
                                   [ 0.01629553,  0.04207279, -0.19266241,  0.09719722,  0.19258326,
                                     -0.17236181, -0.19187486,  0.24177364],
                                   [-0.32311836, -0.27003105,  0.2608717, -0.02492495,  0.08358762,
                                    0.10022042, -0.15297461,  0.08774322],
                                   [ 0.29537144, -0.09006863, -0.10020494,  0.33958726, -0.24019225,
                                     -0.07569654, -0.03908319,  0.11942865],
                                   [-0.0573139,  0.02683016, -0.02152383, -0.22080171,  0.2626398 ,
                                    -0.47796454,  0.11247055,  0.30683287],
                                   [ 0.23436397, -0.14955325, -0.06261889, -0.3300021,  0.07542506,
                                     0.20284236, -0.36264723,  0.06238351],
                                   [ 0.21669514,  0.39925849,  0.29661485,  0.03099017,  0.13517511,
                                     0.36209333,  0.07688017, -0.1951162]])

            self.B_mat = np.array([[ 0.18733102, -0.54452893, -0.83958875],
                                   [-0.08249443,  0.30352079,  1.35459433],
                                   [-1.93302292, -0.60032656, -1.07215529],
                                   [-0.        ,  0.48996532,  0.        ],
                                   [-1.79467884,  0.73936312,  0.        ],
                                   [ 0.84037553,  1.71188778,  1.43669662],
                                   [-0.88803208, -0.19412354, -1.9609    ],
                                   [ 0.        , -2.13835527, -0.19769823]])

            # self.D = np.zeros((6, self.n), dtype=float)  # Dx + Eu <= h
            self.D = np.zeros((0, self.n), dtype=float)  # Dx + Eu <= h
            self.E = np.array(([[-1., 0., 0.], [1., 0., 0.],
                                [0., -1., 0.], [0., 1., 0.],
                                [0., 0., -1.], [0., 0., 1.]]))
            self.h = np.array(([[1.], [1.], [1.], [1.], [1.], [1.]]))
            self.n_beta = 1
            self.li = (np.array([1.], dtype=float),)  # selector row matrix for cost epigraph
            self.phi_i_x_const = (0.,)
            self.phi_i_x_lin = (np.zeros((self.n,), dtype=float),)
            self.phi_i_x_quad = (np.eye(self.n, dtype=float),)
            input_cost = 1.
            self.Ri = (input_cost * np.eye(self.m, dtype=float),)  # 2 * input cost Hessian
            self.ri = (np.zeros((self.m,), dtype=float),)  # input cost linear component
            self.ti = (0.,)  # constant cost component
            self.state_mean = np.zeros((self.n,), dtype=float)
            self.state_var = np.eye(self.n, dtype=float)
            self.pure_lqr = True
            self.brute_force_solve = False
        elif model_dict is not None:
            # Load model parameters from an input dictionary
            for (k, v) in model_dict.iteritems():
                exec('self.' + k + ' = v')
        else:
            print "Unrecognized system model!"
            raise SystemExit()

        # Extra attributes generated for linear systems (i.e. where system has (A, B) matrices)
        if self.A_mat is not None and self.B_mat is not None:
            # Calculate controllability steps (i.e. steps to reach arbitrary state without
            # state-input constraints)
            self.ctrb_steps = self.calc_ctrb_steps(self.A_mat, self.B_mat, eig_tol=1e-4)
            assert self.ctrb_steps < np.inf, "System is not controllable! Exiting."
            print "Unconstrained system is controllable to any state in %d steps." % self.ctrb_steps

            self.dare_sol = None
            # If quadratic state and input costs used, evaluate a Lyapunov function for comparison
            # Note this is only the optimal value function if there are no constraints or other cost
            if np.any([rmat > 0 for rmat in self.Ri]):
                try:
                    self.dare_sol = solve_discrete_are(self.A_mat * np.sqrt(self.gamma),
                                                       self.B_mat * np.sqrt(self.gamma),
                                                       self.phi_i_x_quad[0], self.Ri[0])
                    print "DARE solution for LQR control found."
                except Exception as e:
                    print "Failed to solve discrete algebraic Riccati equation for " + self.name
                    print e
        else:
            print "System " + self.name + " is nonlinear; no LQR solution available."
            self.ctrb_steps = None
            self.dare_sol = None

        self.n_beta_c = len(self.li)  # Number of constraints in epigraph form of stage cost
        assert len(self.li) == len(self.Ri) == len(self.ri) == len(self.ti)
        # assert len(self.phi_i_x_const) == len(self.phi_i_x_lin) == len(self.phi_i_x_quad) \
        #     == self.n_beta_c
        assert np.all([l.shape == (self.n_beta,) for l in self.li])
        assert np.all([r.shape == (self.m,) for r in self.ri])

    def set_attribute(self, attr_name, attr_value):
        """Setter for attributes.

        :param attr_name: String; name of attribute
        :param attr_value: Value of attribute
        :return: Nothing
        """
        modifiable_attributes = ['name', 'gamma', 'A_mat', 'B_mat', 'D', 'E', 'h',
                                 'n_beta', 'li', 'phi_i_x_const', 'phi_i_x_lin', 'phi_i_x_quad',
                                 'Ri', 'ri', 'ti', 'state_mean', 'state_var']
        numpy_array_attributes = ['A_mat', 'B_mat', 'D', 'E', 'h']
        if attr_name in modifiable_attributes:
            if attr_name in numpy_array_attributes and isinstance(attr_value, list):
                # Make a numpy array from list input, if entered as a list
                attr_value_to_enter = np.array(attr_value, dtype=float)
            else:
                attr_value_to_enter = attr_value
            setattr(self, attr_name, attr_value_to_enter)
            print "Attribute '" + attr_name + "' changed to "
            print " ", getattr(self, attr_name)
        else:
            raise SystemError("Cannot modify attribute '" + attr_name + "'! Exiting.")

    def get_attribute(self, attr_name):
        return getattr(self, attr_name)

    def copy(self):
        return deepcopy(self)

    def f_x_x(self, x=None):
        # Evaluate f_x(x) (system dynamics are of the form x_plus = f_x(x) + F_u(x)u )
        if x is not None:
            if self.A_mat is not None:
                return np.dot(self.A_mat, x)  # Linear dynamics
            else:
                print "Cannot return f_x(x) for system " + self.name + "!"
                raise SystemExit()
        else:
            # If no state x is supplied, return linear coefficient, i.e. the A matrix if present.
            return 0., self.A_mat

    def f_u_x(self, x):
        # Evaluate input coefficient matrix F_u(x) in the dynamics for given x
        # (system dynamics are of the form x_plus = f_x(x) + F_u(x)u )
        if x is not None:
            if self.B_mat is not None:
                return self.B_mat
            else:
                print "Cannot return F_u(x) for system " + self.name + "!"
                raise SystemExit()
        else:
            return self.B_mat  # Just linear coefficient of u if no x is specified

    def h_x(self, x):
        # State-input constraints right-hand side as a function of x (constraint is Eu <= h(x))
        if self.name == "2d_obs":
            u_max = 0.4
            if x[0] <= -2 and -1 < x[1] < 1:
                u1_max_here = min(u_max, -2 - x[0])
            else:
                u1_max_here = u_max
            return np.array([u1_max_here, u_max, u_max, u_max])
        else:
            return self.h - np.dot(self.D, x)

    def phi_i_x(self, i, x=None):
        # Stage cost associated with state for stage cost epigraph constraint i, at state x
        if x is not None:
            # Return phi_i(x) for the particular value of x passed to the function
            return self.phi_i_x_const[i] + \
                   np.dot(self.phi_i_x_lin[i], x) + \
                   0.5 * np.dot(x, np.dot(self.phi_i_x_quad[i], x))
        else:
            # Return only the parameters (constant, linear term, Hessian) of phi
            return self.phi_i_x_const[i], \
                   self.phi_i_x_lin[i], \
                   self.phi_i_x_quad[i]  # Hessian will be multiplied by 0.5 later

    def beta_vec(self, x, u):
        # Return vector of epigraph variables beta characterizing the stage cost, by looping through
        # all epigraph constraints and seeing if they apply and are active, for each element of beta
        beta_vec = -np.inf * np.ones((self.n_beta,), dtype=float)
        for l in range(self.n_beta):
            # For each element of the beta vector
            for m in range(self.n_beta_c):
                # Identify constraints that relate to this element and find max
                if self.li[m][l] == 1:
                    # Constraint m relates to element l of the beta vector
                    cost_candidate = (self.phi_i_x(m, x) +
                                      np.dot(self.ri[m], u) +
                                      0.5 * np.dot(u, np.dot(self.Ri[m], u)) +
                                      self.ti[m])
                    beta_vec[l] = max(beta_vec[l], cost_candidate)

        # assert np.all(beta_vec > -np.inf), "Not all beta elements are > -infinity!"
        return beta_vec

    def quad_stage_cost(self, x, u):
        """Returns simple cost of the form 0.5 * x'Qx + 0.5 * u'Ru

        :param x: state, numpy vector of length self.n
        :param u: input, numpy vector of length self.m
        :return: stage cost, scalar
        """
        assert self.n_beta == 1
        return 0.5 * (np.dot(x, np.dot(self.phi_i_x_quad[0], x)) + np.dot(u, np.dot(self.Ri[0], u)))

    def get_x_plus(self, x, u):
        # Evaluate successor state as a function of current state x and input u
        return self.f_x_x(x) + np.dot(self.f_u_x(x), u)

    def manual_policy(self, x_in):
        # Manually specify a u = pi(x_in) by hard-coding within this function.
        u = np.min(1, np.max(-0.5376666 * x_in, -1))

        # Make sure you specified a legal policy that returns an array u of the right size:
        assert isinstance(u, np.ndarray)
        assert u.shape == (self.n, )
        return u

    def simulate(self, x0, fa_in, n_steps, iter_no=None,
                 save_plot=False, no_input=False, return_ub=False, manual_policy=False):
        """ Simulate the system for n_steps steps using the supplied approximate value function

        :param x0: Initial state
        :param fa_in: Function approximator: VFApproximator or QFApproximator object
        :param n_steps: Number of steps
        :param iter_no: Iteration number of algorithm (or other label) to label results with
        :param save_plot: Boolean, generate and save plot of trajectory
        :param no_input: Boolean, use zero input (i.e., simulate autonomous system transient)
        :param return_ub: Boolean, return upper bound on value function by measuring costs along
            the trajectory
        :param manual_policy: Boolean, use the policy hard-coded in self.manual_policy()
        :return: If return_ub: Scalar cost upper bound. Else: List containing (x_t, u_t) pairs
        """
        assert x0.shape == (self.n,), "Simulation started from an x0 with the wrong dimension.\n" \
                                      "Dimension required: %d, " % self.n + "x0: " + str(x0)
        if return_ub:
            assert n_steps > self.n, "Probably not enough steps to force regulation to the origin!"
        output = []
        cost_ub = 0.
        u_end_seq = None

        for t in range(n_steps):
            steps_to_go = n_steps - t  # including the current one
            if t == 0:
                x_t = copy.copy(x0)
            else:
                x_t = self.get_x_plus(output[t-1][0], output[t-1][1])  # x(t-1), u(t-1)
            if no_input:
                u_t = np.zeros((self.m,), dtype=float)
                cost_ub += (self.gamma ** t) * np.sum([self.phi_i_x(i, x_t)
                                                       for i in range(self.n_beta_c)])
            else:
                if manual_policy:
                    u_t = self.manual_policy(x_t)
                    stage_cost = self.quad_stage_cost(x_t, u_t)
                else:
                    u_t, _, stage_cost = fa_in.solve_for_xhat(x_t, extract_constr=False)
                cost_ub += (self.gamma ** t) * stage_cost
                # if return_ub and steps_to_go <= self.ctrb_steps and \
                #         (self.A_mat is not None and self.B_mat is not None):
                #     # Impose constraints before the end of the horizon so that state enters
                #     # successive subspaces from which origin can be reached.
                #
                #     if u_end_seq is None:
                #         # Calculate sequence to origin assuming no input constraints
                #         # print "  Planning feasible inputs from x = " + str(x_t) + "..."
                #         x_end_seq, u_end_seq = self.calc_seq_to_origin(x_t, self.ctrb_steps)
                #         # print "x seq: ", str(x_end_seq)
                #
                #     # Inputs are returned in reverse order (i.e. the input that takes system from
                #     # the second-to-last state to the origin is the first m elements of the vector)
                #     u_to_apply = u_end_seq[(steps_to_go - 1)*self.m:
                #                            steps_to_go*self.m]
                #
                #     xplus_constr = self.get_x_plus(x_t, u_to_apply)
                #     # if steps_to_go == 1:
                #     #     assert np.linalg.norm(xplus_constr) <= 1e-6, \
                #     #         "Not steering to origin, but to " + str(xplus_constr) + \
                #     #         "\nInput: " + str(u_to_apply) + ", t = %d, steps_to_go = %d" % (t, steps_to_go)
                #
                #     # print "  u(%d): " % t + str(u_to_apply)
                #
                #     # ctrb, ctrb_nullspace = self.ctrb_nullspace(self.A_mat, self.B_mat, n_steps-t-1)
                #     # print "t = %d, n_steps = %d" % (t, n_steps)
                #     # print "A =", self.A_mat, "B =", self.B_mat
                #     # print "Nullspace:", ctrb_nullspace, ctrb_nullspace.dtype
                #     # if ctrb is not None:
                #     #     print "Matrix product:"
                #     #     print [np.dot(ctrb.T, ctrb_nullspace)
                #     try:
                #         u_t_alt, _, stage_cost_alt = \
                #             fa_in.solve_for_xhat(x_t, extract_constr=False,
                #                                  xplus_constraint=xplus_constr)
                #         assert np.linalg.norm(u_t_alt - u_to_apply) <= 1e-4, \
                #             "Wrong input: " + str(u_to_apply) + ", " + str(u_t_alt)
                #         cost_ub += (self.gamma ** t) * (stage_cost_alt - stage_cost)
                #         u_t = u_t_alt
                #     except:
                #         print "  Infeasible input suggested for step %d." % t
                #         print "  State was " + str(x_t)
                #         print "  Input to apply was " + str(u_to_apply)
                #         print "  Successor state would have been " + str(xplus_constr)
                #         cost_ub = np.inf
            output.append((x_t, u_t))

        if save_plot:
            output_dir = 'output/' + self.name
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            plt.subplots(2, 1)
            plt.subplot(211)
            for state_element in range(output[0][0].shape[0]):
                plt.plot(range(n_steps), [xu[0][state_element] for xu in output],
                         label='x%d' % (state_element + 1))
            plt.title('State and input trajectories, ' + str(iter_no))
            plt.ylabel('State $x_t$')
            plt.legend()
            # plt.ylim([-2., 2.])
            plt.subplot(212)
            plt.plot(range(n_steps), [xu[1][0] for xu in output])
            plt.ylabel('Input $u_t$')
            plt.xlabel('Time step $t$')
            plt.ylim([-4, 4])

            if iter_no is None:
                suffix = ''
            else:
                suffix = str(iter_no)
            plt.savefig(output_dir + '/traj_' + suffix + '.pdf')
            plt.close()

            savemat(output_dir + '/traj_' + suffix + '.mat', {'output': output})

        if return_ub:
            return cost_ub
        else:
            return output

    def calc_seq_to_origin(self, x0, ctrb_steps):
        """Calculate a sequence of ctrb_steps state steps that minimize input energy from x0 to the
        origin, assuming no constraints on the input

        :param x0: Initial state
        :param ctrb_steps: Number of steps to take (derived from controllability properties of the
            system)
        :return: List of x in sequence, list of u in sequence
        """
        assert self.A_mat is not None and self.B_mat is not None
        ctrb_k = np.hstack((np.dot(np.linalg.matrix_power(self.A_mat, i), self.B_mat)
                            for i in range(self.ctrb_steps)))
        assert np.linalg.matrix_rank(ctrb_k, tol=1e-6) == self.n, "k-step controllability failed!"
        u_seq = np.dot(np.dot(ctrb_k.T, np.linalg.inv(np.dot(ctrb_k, ctrb_k.T))),
                       -np.dot(np.linalg.matrix_power(self.A_mat, ctrb_steps), x0))
        assert np.linalg.norm(np.dot(np.linalg.matrix_power(self.A_mat, ctrb_steps), x0) +
                              np.dot(ctrb_k, u_seq)) <= 1e-6, "Sequence does not steer to origin!"
        assert u_seq.shape == (ctrb_steps * self.m,), "Sequence has wrong length!"
        x_seq = []
        x = x0
        for t in range(ctrb_steps):
            x_seq.append(self.get_x_plus(x, u_seq[(ctrb_steps - t - 1)*self.m:
                                                  (ctrb_steps - t)*self.m]))
            x = x_seq[-1]
        return x_seq, u_seq

    @staticmethod
    def calc_ctrb_steps(A, B, eig_tol):
        # Return number of steps needed to regulate linear system (A, B) to any desired state.
        # Assuming unconstrained input, this only requires simple controllability analysis.
        n = A.shape[0]
        for t in range(1, n+1):
            # Check if rank of matrix [B AB A^2 B, ..., A^(t-1) B] == n
            if np.linalg.matrix_rank(np.hstack(tuple(np.dot(np.linalg.matrix_power(A, i), B)
                                                     for i in range(t))), tol=eig_tol) == n:
                return t
        # If we get to full controllability matrix and still not rank, system not controllable.
        print n
        print np.hstack((np.dot(np.linalg.matrix_power(A, i), B)
                                                for i in range(n)))
        return np.inf
