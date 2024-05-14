import numpy as np
from scipy.linalg import dft
import math
from Generate_channel import generate_channel

num_antenna_bs = 64
Rician_factor = 10
Pt_dB = -5 # Transmit power in dB
Pt = 10**(Pt_dB/10)
noise_power = 10**(-90/10 - 3) # Noise power: -90 dBm
num_samples = 5000 # Size of dataset
time_slots = 4 # time_slots for pilot transmission

#code_book = dft(num_antenna_bs)

def generate_data(num_antenna = 64, Rician_factor = 10, Pt_dB = -5, noise_power_in_dB = -90, num_samples = 5000, time_slots = 4):
    Pt = 10**(Pt_dB/10)
    noise_power = 10**(noise_power_in_dB/10 - 3)
    code_book = np.zeros((num_antenna_bs, num_antenna_bs), dtype=complex)
    for i in range(num_antenna_bs):
        angle = 2 * np.pi/num_antenna_bs * i
        code_book[:, i] = np.exp(1j * np.pi * np.cos(angle) * np.arange(num_antenna_bs))

    #-----------------------------------------------------------------------------
    data_set_input = []
    data_set_label = []
    data_set_label_transmisson = []

    for ii in range(num_samples):
        
        channel_bs_user = generate_channel(num_antenna_bs, Rician_factor=Rician_factor) # shape: (um_antenna_bs, 1)
        
        #======================== Input: received siganls at BS ==================
        y_bs = np.zeros((time_slots, 1), dtype=complex)
        for t in range(time_slots):
            
            tt = num_antenna_bs//time_slots * t
            
            beam = 1/np.sqrt(num_antenna_bs) * code_book[:, tt]
            
            beam_T = np.conjugate(beam.reshape(1, num_antenna_bs))
            
            noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, 1]) \
                                + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, 1])
            noise = noise * np.sqrt(noise_power)      
            
            P_pilot = 10**(-10/10) # Power of pilots, -10 dB
            yt = np.sqrt(P_pilot) * channel_bs_user + noise
        
            yt = np.dot(beam_T, yt)
            y_bs[t, :] = yt
        
        # convert complex vector into real-valued vector, as neural networks only support real-valued input
        #y_bs = np.concatenate([y_bs.real, y_bs.imag], axis=1) # shape: (time_slots, 2)
        #y_bs = y_bs.reshape(time_slots, 2, 1) # Input to CNN requies 3 dimensions. You can modify it if you use other neural networks
        data_set_input.append(y_bs)
        
        # print(data_set_input[0][0])
        # print(data_set_input[0])
        # exit()

                                
        #=========================== Output: optimal beam index ==================
        Rate_max = 0
        Beam_index_optimal = 0
        sample_transmissions = []
        for i in range(num_antenna_bs):
        
            beam = code_book[:, i]
            
            w = np.sqrt(Pt)/np.sqrt(num_antenna_bs) * beam
            
            channel_bs_user_T = np.conjugate(channel_bs_user.reshape(1, num_antenna_bs))
            
            h_w= np.dot(channel_bs_user_T, w)
            
            SNR = Pt * (np.abs(np.squeeze(h_w)) **2) / noise_power
            
            Rate = math.log2(1 + SNR)

            sample_transmissions.append(Rate)
            
            if Rate > Rate_max:
                Rate_max = Rate
                Beam_index_optimal = i
        
        data_set_label_transmisson.append(sample_transmissions)
        data_set_label.append(Beam_index_optimal)
    
    np.save('data/data_input_complex.npy', np.array(data_set_input))
    np.save('data/data_label_complex.npy', np.array(data_set_label))
    np.save('data/data_label_transmission_complex.npy', np.array(data_set_label_transmisson))
    return

generate_data(num_samples=10000)