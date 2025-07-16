# Parameter file
import numpy as np

parameter = {
  # TOPOLOGY PARAMETERS
  'topology': 'E7',              # Self-explanatory
  'LAx': 1,                     # In diameter of LSS
  'LAy': 0,                     # In diameter of LSS
  'LBx': 0,                     # In diameter of LSS
  'LBz': 1,                     # In diameter of LSS
  'LCy': 1,                     # In diameter of LSS

  'do_polarization': False, 


  
  'c_l_accuracy': 0.99,          # Determines what k_max is. Determined by doing the integration up to a k_max
  
  'x0': np.array([0,0,0], dtype=np.float64),

                                  # that gives 'c_l_accuracy' times CAMB c_l ouput
  'l_max': 20,                    # Self-explanatory
}