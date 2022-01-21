from AutoDot.tune import tune_with_playground

def tune(num_iteration, sampler, score_function="score_nothing", popsize=5):

    template_config = {
        "playground": {
            "shapes": {
            "Crosstalk_box": {},
            "Leakage": {},
            "Circle": {
                "origin": [-500, 0, -1200],
                "r": 600
            }
            },
            "ndim": 3,
        },

        "plunger_gates": [1, 2],
        "save_dir": "mock_device_demo/",

        "investigation": {
            "measurement_seq": ["diag_trace", "2d_lowres", "2d_highres"],
            "cond_meas": [False, { "quantile": 0.85, "min_thresh": 0.001 }, False],
            "diag_trace": {
            "func": "mock_measurement",
            "condition": "mock_peak_check",
            "a": [0, 0],
            "b": [-1000, -1000],
            "pause": 0,
            "verbose": True
            },
            "2d_lowres": {
            "func": "mock_measurement",
            "condition": "mock_score_func",
            "target": [-500, -250],
            "pause": 0
            },
            "2d_highres": {
            "func": "mock_measurement",
            "condition": "check_nothing",
            "pause": 0
            },
            "score_func": {
            "func": score_function
            }
        },

        "detector": {
            "d_r": 20,
            "len_after_poff": 300,
            "th_high": 0.2,
            "th_low": 0.01
        },

        "general": {
            "directions": [-1.0, -1.0, -1.0],
            "lb_box": [-2000, -2000, -2000],
            "bound": [-2000, -2000, -2000],
            "num_samples": num_iteration,
            "origin": [0, 0, 0],
            "ub_box": [0, 0, 0],
            "sampler": sampler
        },

        "cmaes": {
            "popsize": popsize
        },

        "gpr": {
            "restarts": 5,
            "factor_std": 2.0,
            "gpr_start": 10,
            "gpr_on": False,
            "length_prior_mean": 0.4,
            "length_prior_var": 0.1,
            "r_min": 0.0,
            "var_prior_mean_divisor": 4.0,
            "kernal": "Matern52"
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

        "sampling": {
            "max_steps": 100000,
            "n_part": 200,
            "sigma": 25
        },

        "pruning": {
            "pruning_stop": 30,
            "pruning_on": False,
            "step_back": 100
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

    return tune_with_playground(template_config)