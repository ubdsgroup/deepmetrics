import os
import pickle
import re
from os import path
from datetime import datetime
from datetime import timedelta
from copy import deepcopy


class Error_Parser():
    def __init__(self):
        self.error_values = {}

    """ Combines all log data in node_dir into a single files. If "error" is present
        in log data, file is instead saved as anomalous"""

    def combine_log_data(self, parent_dir, node_dir):
        self.error_values[node_dir] = {}
        for file in sorted(os.listdir(parent_dir + "/" + node_dir)):
            if len(file) < 10:
                continue
            error_values = [0, 0, 0]
            date_bucket = datetime(2000, 1, 1, 0, 0, 0, 0)
            with open(parent_dir + "/" + node_dir + "/" + file, "r+") as log_file:

                data_lines = log_file.readlines()
                for line in data_lines:

                    # Get the date and time of the log line through regex matching
                    date_string = re.search("\D{3}\s*\d*\s\d{2}.\d{2}.\d{2}", line)
                    if date_string is not None:
                        date_string = date_string.group()
                        date = datetime.strptime(date_string, '%b %d %H:%M:%S')
                        date = date.replace(year=2019)
                        # Subtract our bucket start from this date to get a timedelta object
                        difference = date - date_bucket
                        if difference.total_seconds() > 3600:
                            if date_bucket.year == 2000:
                                date_bucket = date

                            else:
                                self.error_values[node_dir][date_bucket] = deepcopy(error_values)
                                date_bucket = date_bucket + timedelta(hours=1)
                                error_values = [0, 0, 0]

                    if "warn" in line or "Warn" in line:
                        error_values[0] += 1
                    if "error" in line or "Error" in line:
                        error_values[1] += 1
                    if "fatal" in line or "Fatal" in line:
                        error_values[2] += 1


if __name__ == '__main__':

    parent_dir = "unparsed_logs/12cpu48ram"
    EP = Error_Parser()
    for node in os.listdir(parent_dir):
        if len(node) < 10:
            continue
        else:
            sub_dir = node
            EP.combine_log_data(parent_dir, sub_dir)

    with open("tabulated_errors.pkl", "wb") as pk_file:
        pickle.dump(EP.error_values, pk_file)
