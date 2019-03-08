import numpy as np
from gddp import System, VFApproximator

####################################################################################################
# 1D unconstrained input

md1_lqr_uc = {'m': 1, 'n': 1, 'gamma': 1.00,
              'A_mat': np.array([[0.9]], dtype=float),
              'B_mat': np.array([[1.]], dtype=float),
              'D': np.zeros((0, 1), dtype=float),  # Dx + Eu <= h
              'E': None,
              'h': None,
              'n_beta': 1,
              'li': (np.array([1.], dtype=float),),  # selector row matrix for cost epigraph
              'phi_i_x_const': (0.,),
              'phi_i_x_lin': (np.array([0.]),),
              'phi_i_x_quad': (np.array([[1.]]),),
              'Ri': (np.array([[0.5]], dtype=float),),  # 2 * input cost Hessian
              'ri': (np.array([0.], dtype=float),),  # input cost linear component
              'ti': (0.,),  # constant cost component
              'state_mean': np.array([0.], dtype=float),
              'state_var': np.array([[1.]], dtype=float),
              'pure_lqr': True,
              'brute_force_solve': False}

strategy = {'max_iter': 10,
            'n_x_points': 100, 'rand_seed': 1,
            'sol_strategy': 'random', 'conv_tol': 1e-4, 'stop_on_convergence': False,
            'remove_redundant': False, 'removal_freq': 10, 'removal_resolution': 100000,
            'focus_on_origin': False, 'consolidate_constraints': False, 'consolidation_freq': False,
            'value_function_limit': 10000,
            'brute_force_grid_res': 100}
audit = {'eval_ub': False, 'eval_ub_freq': 1, 'eval_ub_final': False,
         'eval_bellman': True, 'eval_bellman_freq': 1,
         'eval_integral': True, 'eval_integral_freq': 1,
         'n_independent_x': 500, 'eval_convergence': False, 'eval_convergence_freq': 100}
outputs = {'cl_plot_j': True, 'cl_plot_freq': 20, 'cl_plot_final': False, 'cl_plot_n_steps': 50,
           'vfa_plot_j': True, 'vfa_plot_freq': 1, 'vfa_plot_ub': False, 'vfa_plot_final': True,
           'policy_plot_j': False, 'policy_plot_freq': 5, 'policy_plot_final': True,
           'suppress all': False}

sys1a = System(name="1D_LQR_uc", model_dict=md1_lqr_uc)
vfa = VFApproximator(sys1a, solver='gurobi')
vfa.create_vfa_model()
vfa.approximate(strategy=strategy, audit=audit, outputs=outputs)
#
# ####################################################################################################
# # 1D constrained input
#
# md1_lqr_c = {'m': 1, 'n': 1, 'gamma': 1.00,
#            'A_mat': np.array([[0.9]], dtype=float),
#            'B_mat': np.array([[1.]], dtype=float),
#            'D': np.array([[0.], [0.]], dtype=float),  # Dx + Eu >= h
#            'E': np.array([[1.], [-1.]], dtype=float),
#            'h': np.array([[-0.2], [-0.2]], dtype=float),
#            'n_beta': 1,
#            'li': (np.array([1.], dtype=float),),  # selector row matrix for cost epigraph
#            'phi_i_x_const': (0.,),
#            'phi_i_x_lin': (np.array([0.]),),
#              'phi_i_x_quad': (np.array([[1.0]]),),
#              'Ri': (np.array([[1.0]], dtype=float),),  # 2 * input cost Hessian
#            'ri': (np.array([0.], dtype=float),),  # input cost linear component
#            'ti': (0.,),  # constant cost component
#            'state_mean': np.array([0.], dtype=float),
#              'state_var': np.array([[1.]], dtype=float),
#              'pure_lqr': True}
#
# strategy = {'max_iter': 50,
#             'n_x_points': 50, 'rand_seed': 1,
#             'sol_strategy': 'random', 'conv_tol': 1e-4, 'stop_on_convergence': True,
#             'remove_redundant': False, 'removal_freq': 10}
# audit = {'eval_ub': False, 'eval_ub_freq': 1, 'eval_ub_final': False,
#          'eval_bellman': False, 'eval_bellman_freq': 10,
#          'eval_integral': False, 'eval_integral_freq': 10}
# outputs = {'cl_plot_j': False, 'cl_plot_freq': 1, 'cl_plot_final': False, 'cl_plot_n_steps': 100,
#            'vfa_plot_j': True, 'vfa_plot_freq': 10, 'vfa_plot_ub': False, 'vfa_plot_final': True,
#            'policy_plot_j': False, 'policy_plot_freq': 1, 'policy_plot_final': False,
#            'suppress all': False}
#
# sys1b = System(name="1D_LQR_c", model_dict=md1_lqr_c)
# vfa = VFApproximator(sys1b, solver='gurobi')
# vfa.create_vfa_model()
# vfa.approximate(strategy=strategy, audit=audit, outputs=outputs)

####################################################################################################
# 2D double integrator

# strategy = {'max_iter': 100,
#             'n_x_points': 100, 'rand_seed': 1,
#             'sol_strategy': 'random', 'conv_tol': 1e-4, 'stop_on_convergence': True,
#             'remove_redundant': False, 'removal_freq': 10}
# audit = {'eval_ub': False, 'eval_ub_freq': 1, 'eval_ub_final': False,
#          'eval_bellman': False, 'eval_bellman_freq': 10,
#          'eval_integral': False, 'eval_integral_freq': 10}
# outputs = {'cl_plot_j': False, 'cl_plot_freq': 1, 'cl_plot_final': False, 'cl_plot_n_steps': 100,
#            'vfa_plot_j': True, 'vfa_plot_freq': 5, 'vfa_plot_ub': False, 'vfa_plot_final': True,
#            'policy_plot_j': True, 'policy_plot_freq': 5, 'policy_plot_final': True,
#            'suppress all': False}
#
# sys2 = System(name="Double integrator")
# vfa = VFApproximator(sys2, solver='gurobi')
# vfa.create_vfa_model()
# vfa.approximate(strategy=strategy, audit=audit, outputs=outputs)

####################################################################################################
# 8D random system

# strategy = {'max_iter': 100,
#             'n_x_points': 100, 'rand_seed': 1,
#             'sol_strategy': 'random', 'conv_tol': 1e-4, 'stop_on_convergence': False,
#             'remove_redundant': False, 'removal_freq': 1, 'removal_resolution': 10000}
# audit = {'eval_ub': False, 'eval_ub_freq': 1, 'eval_ub_final': False,
#          'eval_bellman': True, 'eval_bellman_freq': 10,
#          'eval_integral': True, 'eval_integral_freq': 10,
#          'n_independent_x': 1000, 'eval_convergence_freq': 10}
# outputs = {'cl_plot_j': False, 'cl_plot_freq': 1, 'cl_plot_final': False, 'cl_plot_n_steps': 100,
#            'vfa_plot_j': False, 'vfa_plot_freq': 5, 'vfa_plot_ub': False, 'vfa_plot_final': False,
#            'policy_plot_j': False, 'policy_plot_freq': 5, 'policy_plot_final': False,
#            'suppress all': False}
#
# sys3 = System(name="8states3inputs")
# vfa = VFApproximator(sys3, solver='gurobi')
# vfa.create_vfa_model()
# vfa.approximate(strategy=strategy, audit=audit, outputs=outputs)
