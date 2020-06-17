import numpy as np


def get_amass_sequence_thetas(which):
    # ['01_08', '01_14', '05_02', '05_04', '05_05', '05_06', '05_07', '86_07']
    seqs = np.load('./visualization/amass_sequences.npy', allow_pickle=True, encoding='latin1').item()
    assert(which in seqs.keys())
    return seqs[which]


def get_specific_shape(which):
    betas = np.zeros(10, dtype=np.float32)
    if which == 'mean':
        return betas
    elif which == 'thin':
        betas[0] = -2.0
        betas[1] = 2.0
    elif which == 'fat':
        betas = -get_specific_shape('thin')
    elif which == 'somethin':
        return get_specific_shape('thin') / 2
    elif which == 'somefat':
        return get_specific_shape('fat') / 2
    elif which == 'tallthin':
        betas[0] = 2.0
        betas[1] = 2.0
    elif which == 'shortfat':
        betas = -get_specific_shape('tallthin')
    else:
        raise AttributeError
    return betas


def get_specific_pose(which):
    """
    `which` index can be 0, 1, 2, ..., 10
    0 - apose
    """
    thetas = np.load('./visualization/some_thetas.npy')
    which = int(which)
    return thetas[which]


def get_specific_style_old_tshirt(which):
    g = np.array([1.5, 0.5, 1.5, 0.0], dtype=np.float32)
    if which == 'mean':
        return g
    elif which == 'big':
        g[0] = 0.
    elif which == 'small':
        g[0] = 2.5
    elif which == 'shortsleeve':
        g[1] = -1.
    elif which == 'longsleeve':
        g[1] = 1.5
    elif which == 'small_shortsleeve':
        g = get_specific_style_old_tshirt('small') + get_specific_style_old_tshirt('shortsleeve') - g
    elif which == 'small_longsleeve':
        g = get_specific_style_old_tshirt('small') + get_specific_style_old_tshirt('longsleeve') - g
    elif which == 'big_shortsleeve':
        g = get_specific_style_old_tshirt('big') + get_specific_style_old_tshirt('shortsleeve') - g
    elif which == 'big_longsleeve':
        g = get_specific_style_old_tshirt('big') + get_specific_style_old_tshirt('longsleeve') - g
    else:
        raise AttributeError
    return g


if __name__ == '__main__':
    get_amass_sequence_thetas('05_04')
    get_specific_pose(0)