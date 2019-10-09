from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pickle


if __name__ == '__main__':
    with open("tabulated_errors.pkl", "rb") as pk_file:
        error_values = dict(pickle.load(pk_file))

    nodes = sorted(list(error_values.keys()))
    for node in nodes:
        plot_dates = []
        warn_score = []
        error_score = []
        fatal_score = []
        dates = sorted(list(error_values[node].keys()))
        start_date = datetime(2000, 2, 28, 0, 0, 0)
        end_date = datetime(2020, 3, 17, 0, 0, 0)
        print(node)
        for date in dates:
            if date > start_date and date < end_date:
                plot_dates.append(date)
                error_vector = error_values[node][date]
                if np.count_nonzero(error_vector) > 0:
                    print(date, error_vector)

                warn_score.append(error_vector[0])
                error_score.append(error_vector[1])
                fatal_score.append(error_vector[2])


        #plt.title(node)
        #plt.plot(plot_dates, warn_score, label="warning")
        #plt.plot(plot_dates, error_score, label="error")
        #plt.plot(plot_dates, fatal_score, label="fatal")
        #plt.xticks(plot_dates, plot_dates)
        #plt.legend()
        #plt.show()
