import time
import sys
sys.path.append('../../')

from util import input_parser as Parser
from util import utilities as Utils
from util import kmeans_cluster as KCluster
from topology_preservation import topographic_error as TE
from topology_preservation import topographic_product as TP
from topology_preservation import davies_bouldin_index as DBI
from topology_preservation import zrehen_measure as ZM


# Config
input_filename = ("data/zoo.txt").replace('\\', '/')
output_save_location = 'output/'
output_save_filename = 'zoo_data'

# Display config
pickle_filename = 'output/general_zoo_data_2018-08-06-11-19-37'

mode = 1  # Algorithm Mode = 1, Display Mode = 2


if __name__ == '__main__':

    input_vector_database, _, _ = Parser.InputParser.parse_input_zoo_data(input_filename, None)
    result_dict = Utils.Utilities.load_object(pickle_filename)

    # Topographic Error e.g., 0.009900990099009901
    # topographic_error = TE.TopographicError('euclidean').calculate_gsom_error(result_dict[0]['gsom'], input_vector_database[0])
    # print('TE =', topographic_error)

    # Topographic Product e.g., -0.005688104320110536
    # topographic_product = TP.TopographicProduct().calculate_gsom_topographic_product(result_dict[0]['gsom'])
    # print('TP =', topographic_product)

    # Davies Bouldin Index e.g.,
    # gsom_list, centroids, labels = KCluster.KMeansSOM().cluster_GSOM(result_dict[0]['gsom'], 5)
    # db_index = DBI.DaviesBouldinIndex().compute_DB_index(gsom_list, labels, centroids)
    # print('DB-Index =', db_index)

    # Zrehen Measure
    zrehen_measure = ZM.ZrehenMeasure().calculate_gsom_zrehen_measure(result_dict[0]['gsom'])
    print('ZM =', zrehen_measure)
