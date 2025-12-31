# Parameter file
# Specify the initial power spectrum of the non-trivial topology (E18 uses the standard power law)
import numpy as np

power_parameter = {
  # POWERSPECTRUM PARAMETERS
  'PS_mod': False,     # Default: no modification to the initial power spectrum
  'E18_mod': False,    # False/True means a given modification is applied to the non-trivial topology only/also E18 in the KL calculation
  'powerspec': 'powlaw',  
  'amp': 1.0,             
  'width': 1.5,           
  'freq': 10,             
  'x_cutoff': 1,          
}

'''
Available power spectra, their name and free parameters:

Standard power law (default)
  'powlaw'
  As*(k/0.05)**(ns-1)

Wavepacket (made to fit with the shape of the Planck posterior)
  'wavepacket'
  As*(k/0.05)**(ns-1)*(1+ np.sin(x*freq)*amp*np.exp(-(x/xc)))
  parameters:
    'amp'     Amplitude
    'freq'    Frecuency of the oscillation
    'width'   The xc, controls the exponential surpression of the oscillation

Exponential cutoff
  'cutoff'
  As*(k/0.05)**(ns-1)*(1-amp*np.exp(-(x/xc)))
  parameters:
  'x_cutoff'    Position of the cutoff
  'amp'         Larger value means a steeper cut-off 

Enhancement
  'enhance'
  As*(k/0.05)**(ns-1)*(1+amp*np.exp(-(x/xc)))
  parameters:
  'x_cutoff'      Position of the rise
  'amp'           Larger value means a steeper increase

Logarithmic oscillation model
  'logosci'
  As*(k/0.05)**(ns-1)*(1+amp*np.cos(np.log(k/0.05)*freq))
  parameters:
  'amp'     Amplitude
  'freq'    Frequency
    
'''