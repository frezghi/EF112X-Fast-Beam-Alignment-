import numpy as np
from scipy.linalg import dft
import math
from Generate_channel import generate_channel


num_antenna_bs = 100
Rician_factor = 10
Pt_dB = -5 # Transmit power in dB
Pt = 10**(Pt_dB/10)
noise_power = 10**(-90/10 - 3) # Noise power: -90 dBm

channel_bs_user = generate_channel(num_antenna_bs, Rician_factor=Rician_factor)

#code_book = dft(num_antenna_bs)

code_book = np.zeros((num_antenna_bs, num_antenna_bs), dtype=complex)
for i in range(num_antenna_bs):
    angle = 2 * np.pi/num_antenna_bs * i
    code_book[:, i] = np.exp(1j * np.pi * np.cos(angle) * np.arange(num_antenna_bs))

Rate_max = 0
Beam_index_optimal = 0
for i in range(num_antenna_bs):

    beam = code_book[:, i]
    
    w = np.sqrt(Pt)/np.sqrt(num_antenna_bs) * beam
    
    channel_bs_user_T = np.conjugate(channel_bs_user.reshape(1, num_antenna_bs))
    
    h_w= np.dot(channel_bs_user_T, w)
    
    SNR = Pt * (np.abs(np.squeeze(h_w)) **2) / noise_power
    
    Rate = math.log2(1 + SNR)
    
    if Rate > Rate_max:
        Rate_max = Rate
        Beam_index_optimal = i
        
print('Rate_max (bits/s/Hz):', Rate_max)
print('Beam_index_optimal:', Beam_index_optimal)



