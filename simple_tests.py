from gddp import System, VFApproximator, QFApproximator

####################################################################################################
# 1D system

sys = System(name='1D constrained VFA', preset_system='1D')
sys.set_attribute('D', [[0.], [0.]])
sys.set_attribute('E', [[1.], [-1.]])
sys.set_attribute('h', [0.2, 0.2])
vfa = VFApproximator(sys)
vfa.create_vfa_model()
vfa.approximate()

sys_alt = sys.copy()
sys_alt.set_attribute('name', '1D constrained QFA')

qfa = QFApproximator(sys_alt)
qfa.create_qfa_model()
qfa.approximate()

####################################################################################################
# 2D double integrator

# sys2 = System(preset_system="Double integrator")
# vfa = VFApproximator(sys2)
# vfa.create_vfa_model()
# vfa.approximate()

####################################################################################################
# 8D random system

# sys3 = System(preset_system="8 states 3 inputs")
# vfa = VFApproximator(sys3)
# vfa.create_vfa_model()
# vfa.approximate()
