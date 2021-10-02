import matplotlib.pyplot as plt
from modeldot_autodot_cmaes import tune_with_modeldot
from datetime import date
from statistics import stdev
from pathlib import Path
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')
DATE = str(date.today())
NUM_ITERATION = [40, 60, 80, 100, 120, 140, 160]


def run_tests():
    Path(DATE).mkdir(exist_ok=True)
    Path(DATE + "/errors").mkdir(exist_ok=True)
    paper, cmaes_stages = collect_data(max(NUM_ITERATION))
    comp = [compare_sampler(num_iter,
                            select_first_n_values(paper, num_iter),
                            select_first_n_values(cmaes_stages, num_iter))
            for num_iter in NUM_ITERATION]

    compare_iterations(comp)


def collect_data(num_iterations, popsize=10):
    paper = save_tuning(num_iteration=num_iterations, sampler="Paper_sampler")
    cmaes_stages = save_tuning(num_iteration=int(num_iterations/10),
                               sampler="CMAES_sampler",
                               score_function="mock_count_stages")

    return paper, cmaes_stages


def compare_sampler(num_iterations, paper, cmaes_stages):
    paper_avg = evaluate_average(paper)
    cmaes_stages_avg = evaluate_average(cmaes_stages)
    plot_stage_compare(paper_avg,
                       cmaes_stages_avg,
                       "Average number of points",
                       "Average number of points [{} iterations]".format(
                           num_iterations),
                       "{}/num_points_{}.png".format(DATE,
                                                     num_iterations))

    paper_passed_to_eval = evaluate_passed_to_eval_average(evaluate_passed_to_eval(
        paper, "{}/paper_{}_passed_to_eval.csv".format(DATE, num_iterations)))
    cmaes_stages_passed_to_eval = evaluate_passed_to_eval_average(evaluate_passed_to_eval(
        cmaes_stages, "{}/cmaes_stages_{}_passed_to_eval.csv".format(DATE, num_iterations)))

    plot_stage_compare(paper_passed_to_eval, cmaes_stages_passed_to_eval,
                       "Percentage of passed points",
                       "Percentage of passed points [{} iterations]".format(
                           num_iterations),
                       "{}/percentage_passed_{}.png".format(DATE, num_iterations))

    return {"paper_avg": paper_avg, "paper_passed_to_eval": paper_passed_to_eval, "cmaes_avg": cmaes_stages_avg, "cmaes_passed_to_eval": cmaes_stages_passed_to_eval}


def save_tuning(**kwargs):
    data = np.array([[]])
    filename = "{}/{}{}.csv".format(DATE,
                                    kwargs["sampler"], kwargs.get("score_function", ""))
    if Path(filename).exists():
        data = np.loadtxt(filename, delimiter=';', dtype="i", ndmin=2)
    i = -1
    while len(data) != 100:
        i = i + 1
        print("! ! ! ! ! ! ! ! ! {}-iter{}-collected{}/100 ! ! ! ! ! ! ! !".format(
            kwargs["sampler"], i, len(data)))
        try:
            res, _ = tune_with_modeldot(**kwargs)
        except Exception as err:
            f = open(
                DATE+"/errors/{}_{}.txt".format(str(kwargs).replace(":", ""), i), "w")
            f.write(str(err))
            f.write(str(traceback.format_exc()))
            f.close()
        else:
            if data.size == 0:
                data = np.array([res["conditional_idx"]])
            else:
                data = np.concatenate(
                    (data, np.array([res["conditional_idx"]])))
            # save intermediate status
            np.savetxt(filename, data, delimiter=",", fmt="%i")
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


def plot_stage_compare(paper, cmaes_stages, y_label, title, ex_file):
    num_stages = len(paper[0])
    xticks = range(num_stages)
    xticklabels = ["stage {}".format(s) for s in xticks]
    plot_data(paper, cmaes_stages, y_label,
              title, xticks, xticklabels, ex_file)


def compare_iterations(comp):
    paper_avg = [it["paper_avg"] for it in comp]
    cmaes_avg = [it["cmaes_avg"] for it in comp]
    num_stages = len(paper_avg[0][0])

    for stage in range(num_stages):
        paper = [p[0][stage] for p in paper_avg], [p[1][stage]
                                                   for p in paper_avg]
        cmaes = [c[0][stage] for c in cmaes_avg], [c[1][stage]
                                                   for c in cmaes_avg]
        plot_iteration_compare(paper, cmaes, "Average number of points", "Average number of points (stage {})".format(
            stage), "{}/num_points_{}.png".format(DATE, stage))

    paper_passed_to_eval = [it["paper_passed_to_eval"] for it in comp]
    cmaes_passed_to_eval = [it["cmaes_passed_to_eval"] for it in comp]

    for stage in range(num_stages):
        paper = [p[0][stage] for p in paper_passed_to_eval], [p[1][stage]
                                                   for p in paper_passed_to_eval]
        cmaes = [c[0][stage] for c in cmaes_passed_to_eval], [c[1][stage]
                                                   for c in cmaes_passed_to_eval]
        plot_iteration_compare(paper, cmaes, "Percentage of passed points", "Percentage of passed points (stage {})".format(
            stage), "{}/passed_points_{}.png".format(DATE, stage))


def plot_iteration_compare(paper, cmaes_stages, y_label, title, ex_file):
    num_iter = len(NUM_ITERATION)
    xticks = np.arange(num_iter)
    xticklabels = ["{} iter".format(it) for it in NUM_ITERATION]
    plot_data(paper, cmaes_stages, y_label,
              title, xticks, xticklabels, ex_file)


def plot_data(paper, cmaes_stages, y_label, title, xticks, xticklabels, ex_file):
    paper_avg, paper_stdev = paper
    cmaes_stages_avg, cmaes_stages_stdev = cmaes_stages

    num_stages = len(paper_avg)
    stages = np.arange(num_stages)
    width = 0.2

    fig, ax = plt.subplots()

    ax.bar(stages - width, paper_avg, width, yerr=paper_stdev,
           color="gold", error_kw={"capsize": 4}, label="paper_sampler")
    ax.bar(stages, cmaes_stages_avg,
           width, yerr=cmaes_stages_stdev, color="orange",
           error_kw={"capsize": 4}, label="cmaes_stages")

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.legend()

    fig.tight_layout()
    fig.savefig(ex_file)


if __name__ == "__main__":
    run_tests()
