from AutoDot.tune import tune_from_dict
import sys


def jump(params, inv=False):
    pass
    return params


def measure():
    pass


def check(inv=True):
    pass


def combo1d(jump, measure, anc, deltas, dirs):
    pass


def do1dcombo(jump, measure_cur, anchor_vals, configs, **kwags):
    pass


def do2d(jump, measure_cur, anchor_vals, configs, **kwags):
    pass


def any_true(trace, minc, maxc, configs, **kwags):
    pass


def tune(num_iteration, sampler, score_function="score_nothing", popsize = 10):

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
    tune(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
