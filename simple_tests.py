from gddp import System, VFApproximator, QFApproximator

strategy_dict = {'max_iter': 100}
# outputs_dict = {'suppress_outputs': False}
outputs_dict = {'vfa_plot_j': True}

####################################################################################################
# 1D system

sys = System(name='1D constrained VFA', preset_system='1D')
sys.set_attribute('D', [[0.], [0.]])
sys.set_attribute('E', [[1.], [-1.]])
sys.set_attribute('h', [0.2, 0.2])

vfa = VFApproximator(sys)
vfa.create_vfa_model()
vfa.approximate(strategy_in=strategy_dict, outputs_in=outputs_dict)
vfa.print_function_approximation(save=True)

# sys_alt = sys.copy()
# sys_alt.set_attribute('name', '1D unconstrained VFA')
# sys_alt.set_attribute('E', [[0.], [0.]])
#
# vfa = VFApproximator(sys_alt)
# vfa.create_vfa_model()
# vfa.approximate(strategy_in=strategy_dict, outputs_in=outputs_dict)
# vfa.print_function_approximation(save=True)

# qfa = QFApproximator(sys_alt)
# qfa.create_qfa_model()
# qfa.approximate(strategy_in=strategy_dict)
# qfa.print_function_approximation(save=True)

####################################################################################################
# 2D double integrator

# sys2 = System(preset_system="Double integrator")
# vfa = VFApproximator(sys2)
# vfa.create_vfa_model()
# vfa.approximate(strategy_in=strategy_dict)
# vfa.print_function_approximation(save=True)
#
# qfa = QFApproximator(sys2)
# qfa.create_qfa_model()
# qfa.approximate(strategy_in=strategy_dict)
# qfa.print_function_approximation(save=True)

####################################################################################################
# 8D random system

# sys3 = System(preset_system="8 states 3 inputs")
# vfa = VFApproximator(sys3)
# vfa.create_vfa_model()
# vfa.approximate(strategy_in=strategy_dict, outputs_in=outputs_dict)
# vfa.print_function_approximation(save=True)
# vfa.save_function_approximation()
#
# qfa = QFApproximator(sys3)
# qfa.create_qfa_model()
# qfa.approximate(strategy_in=strategy_dict, outputs_in=outputs_dict)
# qfa.print_function_approximation(save=True)
# qfa.save_function_approximation()
