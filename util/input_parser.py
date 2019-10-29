import pandas as pd
import csv


class InputParser:

    @staticmethod
    def parse_input_zoo_data(filename, header='infer'):

        input_data = pd.read_csv(filename, header=header)

        classes = input_data[17].tolist()
        labels = input_data[0].tolist()

        del input_data[0]
        del input_data[17]

        input_database = {0: input_data.values}

        return input_database, labels, classes

    @staticmethod
    def output_list(data_list, filename):

        with open(filename, 'w') as f:
            wr = csv.writer(f, lineterminator='\n')
            wr.writerow(data_list)
