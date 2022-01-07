import pickle
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt

def get_time_of_first_point(pickle_data):
    times = pickle_data['times']
    conditional_idx = pickle_data['conditional_idx']
    t_sum = 0
    for t, c in zip(times, conditional_idx):
        t_sum += max(t)
        if c == 3:
            return t_sum
    return t_sum


def get_whole_time(pickle_data):
    times = pickle_data['times']
    t_sum = 0
    for t in times:
        t_sum += max(t)
    return t_sum

def plot(avg_t_paper, avg_t_cmaes, avg_mu_paper, avg_mu_cmaes):
    fig, ax = plt.subplots()
    x = np.arange(2)
    width = 0.35

    rects1 = ax.bar(x - width/2, [avg_t_paper, avg_mu_paper], width, label='Paper Sampler')
    rects2 = ax.bar(x + width/2, [avg_t_cmaes, avg_mu_cmaes], width, label='CMAES Sampler')
    
    ax.set_title('Compare runtime')
    ax.set_xticks(x)
    ax.set_ylabel("t in seconds")
    ax.set_xticklabels(["t_160", "mu_t"])
    fig.tight_layout()
    ax.legend()
    plt.show()

def get_pickle_data(pickle_file):
    objects = []
    with (open(pickle_file, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects[0]


def get_t_and_mu(files):
    t = []
    mu = []
    for pickle_file in files:
        pickle_data = get_pickle_data(pickle_file)
        t.append(get_whole_time(pickle_data))
        mu.append(get_time_of_first_point(pickle_data))
    avg_t_paper = sum(t)/len(t)
    avg_mu_paper = sum(mu)/len(mu)
    return avg_t_paper, avg_mu_paper

if __name__ == "__main__":
    date = sys.argv[1]
    files_paper = glob.glob('{}/Paper_sampler_i_*/tuning.pkl'.format(date))
    files_cmaes = glob.glob('{}/CMAES_Sampler_i_*/tuning.pkl'.format(date))
    avg_t_paper, avg_mu_paper = get_t_and_mu(files_paper)
    avg_t_cmaes, avg_mu_cmaes = get_t_and_mu(files_cmaes)
    plot(avg_t_paper, avg_t_cmaes, avg_mu_paper, avg_mu_cmaes)