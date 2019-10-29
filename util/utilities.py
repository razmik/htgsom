import pickle
import datetime, time, sys
import math
import numpy as np
from scipy import spatial
from params import params as Params


class Utilities:

    @staticmethod
    def get_distance(vector_1, vector_2, method, divider=-1):
        if method == Params.DistanceFunction.EUCLIDEAN:
            return np.linalg.norm(vector_2 - vector_1)
        elif method == Params.DistanceFunction.COSINE:
            if np.count_nonzero(vector_2) != 0:
                return spatial.distance.cosine(vector_2, vector_1)
            else:
                return 1.0
        elif method == Params.DistanceFunction.COMBINED:
            cosine_dist = spatial.distance.cosine(vector_2[0:(divider+1)], vector_1[0:(divider+1)])
            euclidean_dist = spatial.distance.euclidean(vector_2[(divider+1):len(vector_2)], vector_1[(divider+1):len(vector_1)])
            distance = (euclidean_dist * 10 + cosine_dist) / 2
            return distance
        else:
            print('ERROR: Undefined distance function -', method)
            sys.exit(-1)

    @staticmethod
    def get_distance_recurrent(global_context, recurrent_weights, alphas):

        gamma_distance = np.linalg.norm(np.dot(alphas.T, (global_context - recurrent_weights)))
        return gamma_distance

    @staticmethod
    def get_max_node_distance_square(node_1, node_2):
        return max(math.pow(node_1.x - node_2.x, 2), math.pow(node_1.y - node_2.y, 2))

    @staticmethod
    def generate_index(x, y):
        return str(x) + ':' + str(y)

    @staticmethod
    def select_winner(nodemap, input_vector, distance_function, distance_divider):

        _, winner = min(nodemap.items(), key=lambda node: Utilities.get_distance(node[1].weights, input_vector,
                                                                                 distance_function, distance_divider))
        return winner

    @staticmethod
    def select_input_to_closest_aggregate_node(aggr_node_list, input_weight, distance_function, distance_divider):
        min_dist = float("inf")

        list_index = -1
        itr = 0
        for aggr_node in aggr_node_list:
            curr_dist = Utilities.get_distance(aggr_node.weights, input_weight.weight,
                                               distance_function, distance_divider)
            if curr_dist < min_dist:
                min_dist = curr_dist
                list_index = itr
            itr += 1

        aggr_node_list[list_index].select_input_vector(input_weight)

    @staticmethod
    def neighbors(nx, ny, neighbour_radius):
        return [(x2, y2) for x2 in range(nx - neighbour_radius, nx + neighbour_radius + 1)
                for y2 in range(ny - neighbour_radius, ny + neighbour_radius + 1)
                if (nx != x2 or ny != y2)]

    @staticmethod
    def increment_node_ages(nodemap):
        for key, node in nodemap.items():
            node.age_increment()

    @staticmethod
    def remove_older_nodes(nodemap, age_threshold):
        for key in list(nodemap.keys()):
            if nodemap[key].age > age_threshold:
                del nodemap[key]

    @staticmethod
    def weight_transformation(nodemap):
        for _, value in nodemap.items():
            value.setup_weights()

    @staticmethod
    def save_object(input_object, filename):
        suffix = '_'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        full_name = filename+suffix
        with open(full_name+'.pickle', 'wb') as handle:
            pickle.dump(input_object, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return full_name

    @staticmethod
    def load_object(filename):
        return pickle.load(open(filename+".pickle", "rb"))
