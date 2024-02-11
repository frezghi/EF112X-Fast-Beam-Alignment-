import numpy as np


def path_loss_d(d):
    loss = 30 + 22.0 * np.log10(d)
    return loss

def generate_pathloss_aoa_aod(location_user, location_bs):

    d0 = np.linalg.norm(location_user - location_bs)

    aoa_bs = (location_user[0] - location_bs[0]) / d0

    pathloss_bs_user = np.zeros([1, 1])

    pathloss_bs_user = path_loss_d(d0)
    
    return pathloss_bs_user, aoa_bs


def generate_channel(num_antenna_bs, location_bs=np.array([0, 0, 0]), Rician_factor = 10):

    # Get the location of user
    location_user = np.empty([1, 3])
    x = np.random.uniform(-25, 25)
    y = np.random.uniform(100, 150)
    z = 0
    location_user = np.array([x, y, z])

    # Get path loss and AOA
    pathloss_bs_user, aoa_bs = generate_pathloss_aoa_aod(location_user, location_bs)

    pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))

    # tmp:(num_antenna_bs, 1) channel between BS and user
    tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, 1]) \
          + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, 1])
          
    a_bs_user = np.exp(1j * np.pi * (np.arange(num_antenna_bs) * aoa_bs))
    a_bs_user = a_bs_user.reshape(num_antenna_bs, 1)

    tmp = np.sqrt(Rician_factor/(1+Rician_factor))*a_bs_user + np.sqrt(1/(1+Rician_factor))*tmp
    
    channel_bs_user = tmp * pathloss_bs_user
    
    return np.array(channel_bs_user)


def main():
    
    num_antenna_bs = 4
    Rician_factor = 10

    channel_bs_user = generate_channel(num_antenna_bs, Rician_factor=Rician_factor)                                                       

    print('channel_bs_user:\n', channel_bs_user)


if __name__ == '__main__':
    main()
