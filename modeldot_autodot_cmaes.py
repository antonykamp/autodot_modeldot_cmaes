from AutoDot.tune import tune_from_dict
import sys
import json
import numpy as np
from math import exp
import matplotlib.pyplot as plt
'''
# Define path to modeldot files
fp = open('parameters.json', 'r')
parameters = json.load(fp)
fp.close()
f_path = parameters['file_path']

sys.path.insert(0, f_path)
'''
from ModelDot import ModelDot
intro = ModelDot.introduction()

device = ModelDot.virtual_device()
main_device = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
plunger_main_device = ['c2', 'c6']
device.getdevice()
device.setdevice(disorder=True)
device.getdisorder()
device.setdisorder()
device.getdisorder()
# Ohmic contacts labelled with capital letters A-F
# Default source-drain path is AF
# Check source-drain
device.getpath()
# Set to another path
device.setpath('B', 'C')
# Check
device.getpath()
# Return to original
device.setpath('A', 'F')
# Check
device.getpath()


def jump(params, inv=False):
    if inv:
        labels = plunger_main_device  # plunger gates
    else:
        labels = main_device  # all gates
    device.setval(labels, params)
    return params


def measure():
    mes = device.do0d(output=True)
    return mes['Current']


def measure_singledot():
    return device.do0d(output=True)['P(Dots)']['1']


def measure_doubledot():
    return device.do0d(output=True)['P(Dots)']['2']


def check(inv=True):
    if inv:
        labels = plunger_main_device  # plunger gates
    else:
        labels = main_device  # all gates

    # function that takes dac key and returns state that channel is in
    dac_state = device.getval(labels, output=True)
    _, b = zip(*dac_state.items())
    return list(b)


def combo1d(jump, measure, anc, deltas, dirs):

    trace = np.zeros(len(deltas))

    for i in range(len(deltas)):
        params_c = anc + dirs*deltas[i]

        jump(params_c)
        trace[i] = measure()

    return trace


def do1dcombo(jump, measure_cur, anchor_vals, configs, **kwags):
    size = configs.get('size', 128)
    direction = np.array(configs.get('direction', [1]*len(anchor_vals)))
    res = configs.get('res', 128)

    delta_volt = np.linspace(0, size, res)

    anchor_vals = np.array(anchor_vals)

    trace = combo1d(jump, measure_singledot,
                    anchor_vals, delta_volt, direction)

    if configs.get('plot', False):
        plt.plot(delta_volt, trace)
        plt.ylim([0.0, 1.0])
        plt.show()
    return trace


def do2d(jump, measure_cur, anchor_vals, configs, **kwags):
    bound = kwags.get('bound', configs['size'])
    res = configs.get('res', 20)
    direction = np.array(configs.get('direction', [1]*len(anchor_vals)))

    iter_vals = [None]*2
    for i in range(2):
        iter_vals[i] = np.linspace(0, bound[i], res)

    iter_deltas = np.array(np.meshgrid(*iter_vals))

    data = np.zeros([res, res])

    for i in range(res):
        for j in range(res):
            params_c = anchor_vals + direction*iter_deltas[:, i, j]
            jump(params_c)

            data[i, j] = measure_doubledot()

    if configs.get('plot', False):
        plt.imshow(data, cmap='bwr', vmin=0.0, vmax=1.0)
        plt.show()

    return data


def any_true(trace, minc, maxc, configs, **kwags):

    passed = np.any(trace > 0.5)

    return passed, passed, None


def score_propabilities(invest_results, config):
    trace_passed, _, _ = invest_results['extra_measure'][3]
    high_trace = [exp(t) for t in trace_passed if t >= 0.5]
    return -(sum(high_trace)/len(high_trace))


def tune_with_modeldot(num_iteration, sampler, score_function="score_nothing", popsize = 11):
    if score_function == "score_propabilities":
        score_function = score_propabilities

    template_config = {
        "plunger_gates": [1, 2],
        "save_dir": "mock_device_demo/",

        "investigation": {
            "measurement_seq": ["diag_trace", "2d_lowres"],
            "cond_meas": [False, False],
            "diag_trace": {
                "func": do1dcombo,
                "condition": any_true,
                "res": 10,
                "size": 200,
                "direction": [-1, -1],
                "plot": False},
            "2d_lowres": {
                "func": do2d,
                "condition": any_true,
                "res": 10,
                "size": [200, 200],
                "direction": [-1, -1],
                "plot": False},
            "2d_highres": {
                "func": "mock_measurement",
                "condition": "check_nothing",
                "pause": 1.0
            },
            "score_func": {
                "func": score_function
            }
        },

        "detector": {
            "d_r": 50,
            "len_after_poff": 20,
            "th_high": 0.2,
            "th_low": 0.01
        },

        "general": {
            "directions": [-1.0]*8,
            "lb_box": [-2000]*8,
            "bound": [-2000]*8,
            "num_samples": num_iteration,
            "origin": [0]*8,
            "ub_box": [0]*8,
            "sampler": sampler
        },

        "cmaes": {
            "popsize": popsize
        },

        "gpc": {
            "gpc_start": 10,
            "d_tooclose": 20.0,
            "gpc_on": True,
            "gpc_list": [True, True, False],
            "configs": {
                "length_prior_mean": 500.0,
                "length_prior_var": 100.0,
                "var_prior_mean": 50.0,
                "var_prior_var": 20.0,
                "kernal": "Matern52"
            }
        },

        "gpr": {
            "restarts": 5,
            "factor_std": 2.0,
            "gpr_start": 10,
            "gpr_on": True,
            "length_prior_mean": 0.4,
            "length_prior_var": 0.1,
            "r_min": 0.0,
            "var_prior_mean_divisor": 4.0,
            "kernal": "Matern52"
        },

        "pruning": {
            "pruning_stop": 30,
            "pruning_on": True,
            "step_back": 100
        },

        "sampling": {
            "max_steps": 100000,
            "n_part": 200,
            "sigma": 25
        },

        "track": [
            "origin",
            "bound",
            "directions",
            "samples",
            "extra_measure",
            "d_vec",
            "poff_vec",
            "meas_each_axis",
            "vols_each_axis",
            "conditional_idx",
            "vols_pinchoff",
            "detected",
            "r_vals",
            "times",
            "poff_traces"
        ],
        "verbose": ["conditional_idx", "vols_pinchoff", "detected", "r_vals"]
    }

    return tune_from_dict(jump, measure, check, template_config)


if __name__ == '__main__':
  # argv[1]: number iteration
  # argv[2]: sampler
  # argv[3]: score func
  # argv[4]: population size
    tune_with_modeldot(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
