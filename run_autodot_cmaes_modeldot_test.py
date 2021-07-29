import matplotlib.pyplot as plt
from modeldot_autodot_cmaes import tune_with_modeldot
from datetime import date
from statistics import stdev
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')


def run_tests():
    prefix = str(date.today())
    Path(prefix).mkdir(exist_ok=True)
    Path(prefix + "/errors").mkdir(exist_ok=True)
    num_iterations = [40, 60, 80, 100, 120, 140]
    paper, cmaes_stages, cmaes_prop = collect_data(max(num_iterations))
    for num_iter in num_iterations:
        compare_sampler(
            num_iter, select_first_n_values(paper, num_iter),
            select_first_n_values(cmaes_stages, num_iter),
            select_first_n_values(cmaes_prop, num_iter))


def collect_data(num_iterations, popsize=10):
    paper = save_tuning(num_iteration=num_iterations, sampler="Paper_sampler")
    cmaes_stages = save_tuning(num_iteration=num_iterations,
                               sampler="CMAES_sampler",
                               score_function="mock_count_stages")
    cmaes_prop = save_tuning(num_iteration=num_iterations,
                             sampler="CMAES_sampler",
                             score_function="score_propabilities")

    return paper, cmaes_stages, cmaes_prop


def compare_sampler(num_iterations, paper, cmaes_stages, cmaes_prop):
    prefix = str(date.today())

    np.savetxt("{}/paper_{}.csv".format(prefix, num_iterations),
               paper, delimiter=",", fmt="%i")
    np.savetxt("{}/cmaes_stages_{}.csv".format(prefix, num_iterations),
               cmaes_stages, delimiter=",", fmt="%i")
    np.savetxt("{}/cmaes_prop_{}.csv".format(prefix, num_iterations),
               cmaes_prop, delimiter=",", fmt="%i")

    paper_avg = evaluate_average(paper)
    cmaes_stages_avg = evaluate_average(cmaes_stages)
    cmaes_prop_avg = evaluate_average(cmaes_prop)
    plot_data(paper_avg,
              cmaes_stages_avg,
              cmaes_prop_avg,
              "Average number of points",
              "Average number of points [{} iterations]".format(
                  num_iterations),
              "{}/num_points_{}.png".format(prefix,
                                            num_iterations))

    paper_passed_to_eval = evaluate_passed_to_eval_average(evaluate_passed_to_eval(
        paper, "{}/paper_{}_passed_to_eval.csv".format(prefix, num_iterations)))
    cmaes_stages_passed_to_eval = evaluate_passed_to_eval_average(evaluate_passed_to_eval(
        cmaes_stages, "{}/cmaes_stages_{}_passed_to_eval.csv".format(prefix, num_iterations)))
    cmaes_prop_passed_to_eval = evaluate_passed_to_eval_average(evaluate_passed_to_eval(
        cmaes_prop, "{}/cmaes_prop_{}_passed_to_eval.csv".format(prefix, num_iterations)))

    plot_data(paper_passed_to_eval, cmaes_stages_passed_to_eval,
              cmaes_prop_passed_to_eval, "Percentage of passed points",
              "Percentage of passed points [{} iterations]".format(
                  num_iterations),
              "{}/percentage_passed_{}.png".format(prefix, num_iterations))


def save_tuning(**kwargs):
    data = np.array([[]])
    i = -1
    while len(data) != 100:
        i = i + 1
        try:
            res, _ = tune_with_modeldot(**kwargs)
        except Exception as err:
            f = open("errors/{}_{}.txt".format(str(kwargs), i), "w")
            f.write(str(err))
            f.close()
            continue

        if data.size == 0:
            data = np.array([res["conditional_idx"]])
        else:
            data = np.concatenate(
                (data, np.array([res["conditional_idx"]])))
    return data


def select_first_n_values(data, n):
    num_stages = max(data[0])
    result = np.array([[]])
    for conditional_idx in data:
        sorted_idx = [len([idx for idx in conditional_idx[:n] if idx == stage])
                      for stage in range(num_stages + 1)]
        if result.size == 0:
            result = np.array([sorted_idx])
        else:
            result = np.concatenate((result, np.array([sorted_idx])))
    return result


def evaluate_average(data):
    num_stages = len(data[0])
    num_data = len(data)
    sum_elements = [sum([res[stage] for res in data])
                    for stage in range(num_stages)]
    average = [sum_elements[stage] / num_data for stage in range(num_stages)]
    stdevi = [stdev([res[stage] for res in data])
              for stage in range(num_stages)]
    return average, stdevi


def evaluate_passed_to_eval(data, ex_file):
    num_stages = len(data[0])
    passed_to_eval = [[passed_to_evaluated(
        res, stage) for res in data] for stage in range(num_stages)]

    saveable = [[data[res_idx][stage]
                 for stage in range(num_stages)] for res_idx in range(len(data))]
    np.savetxt(ex_file, saveable, delimiter=",", fmt="%f")
    return passed_to_eval


def passed_to_evaluated(res, stage):
    passed_evaluated = num_passed(res, stage) / num_evaluated(res, stage)
    return passed_evaluated


def num_passed(result, stage):
    sum_elements = sum_elements_from_start(result, stage + 1)
    return sum_elements


def num_evaluated(result, stage):
    sum_elements = sum_elements_from_start(result, stage)
    return sum_elements


def sum_elements_from_start(result, start):
    num_stages = len(result)
    return sum([result[start + i] for i in range(num_stages - start)])


def evaluate_passed_to_eval_average(data):
    num_stages = len(data)
    num_data = len(data[0])
    sum_elements = [sum(data[stage]) for stage in range(num_stages)]
    average = [sum_elements[stage] / num_data
               for stage in range(num_stages)]
    stdevi = [stdev(data[stage]) for stage in range(num_stages)]
    return average, stdevi


def plot_data(paper, cmaes_stages, cmaes_prop, y_label, title, ex_file):
    paper_avg, paper_stdev = paper
    cmaes_stages_avg, cmaes_stages_stdev = cmaes_stages
    cmaes_prop_avg, cmaes_prop_stdev = cmaes_prop

    num_stages = len(paper_avg)
    stages = np.arange(num_stages)
    width = 0.2

    fig, ax = plt.subplots()

    ax.bar(stages - width, paper_avg, width, yerr=paper_stdev,
           color="gold", error_kw={"capsize": 4}, label="paper_sampler")
    ax.bar(stages, cmaes_stages_avg,
           width, yerr=cmaes_stages_stdev, color="orange",
           error_kw={"capsize": 4}, label="cmaes_stages")
    ax.bar(stages + width, cmaes_prop_avg,
           width, yerr=cmaes_prop_stdev, color="darkorange",
           error_kw={"capsize": 4}, label="cmaes_prop")

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(stages)
    ax.set_xticklabels(["stage {}".format(i) for i in stages])
    ax.legend()

    fig.tight_layout()
    fig.savefig(ex_file)


if __name__ == "__main__":
    run_tests()
