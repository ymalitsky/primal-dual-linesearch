# Important only for nice plots

import numpy as np
import matplotlib as mpl


mpl.rc('lines', linewidth=2)
mpl.rcParams.update(
    {'font.size': 12, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
mpl.rcParams['xtick.major.pad'] = 2
mpl.rcParams['ytick.major.pad'] = 2


def find_smallest_energy(answers):
    """
    from all energies find the smallest.
    """
    energy_list = [min(answers[i][1]) for i in range(len(answers))]
    t = min(energy_list)
    ind = energy_list.index(t)
    return t


def cut_time(times, values, t=300):
    """
    Cut down all energies values untill the given time stamp.
    """
    s = np.argmax(np.array(times) > t)
    return times[:s], np.array(values[:s])
