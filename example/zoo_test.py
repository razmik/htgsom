import time
import sys
import os
sys.path.append('../../')

from util import input_parser as Parser
from util import utilities as Utils
from util import display as Display_Utils

from params import params as Params
from core3 import core_controller as Core

# Config
input_filename = ("data/zoo.txt").replace('\\', '/')
output_save_location = 'output/'
output_save_filename = 'zoo_data'

SF = 0.7
temporal_contexts = 2
forget_threshold = 1000
plot_for_itr = 100
plot_output_name = output_save_location + output_save_filename + str(SF) + '_T_' + str(temporal_contexts) + '_mage_' + str(
    forget_threshold) + 'itr'


# Generate output plot location
output_loc = plot_output_name
output_loc_images = output_loc + '/images/'
if not os.path.exists(output_loc):
    os.makedirs(output_loc)
if not os.path.exists(output_loc_images):
    os.makedirs(output_loc_images)

# Display config
pickle_filename = 'zoo_data_2018-08-02-11-30-32'

mode = 1  # Algorithm Mode = 1, Display Mode = 2


if __name__ == '__main__':

    if mode == 1:

        print('Start running GSOM algorithm.')

        # Init GSOM Parameters
        """
        GSOMParameters(spread_factor, learning_itr, smooth_itr, max_neighbourhood_radius=4, start_learning_rate=0.3,
                 smooth_neighbourhood_radius_factor=0.5, smooth_learning_factor=0.5, distance='EUC', fd=0.1,
                 alpha=0.9, r=0.95)

        IKASLParameters(self, gsom_parameters, aggregate_proximity=2, hit_threshold_fraction=0.05, 
                        aggregate_function='AVG', aggregate_inside_hitnode_proximity=True)
        """
        gsom_params = Params.GSOMParameters(SF, 100, 100, distance=Params.DistanceFunction.EUCLIDEAN,
                                            temporal_context_count=temporal_contexts, forget_itr_count=forget_threshold)
        generalise_params = Params.GeneraliseParameters(gsom_params)

        # Process the input files
        input_vector_database, labels, classes = Parser.InputParser.parse_input_zoo_data(input_filename, None)

        # Process the clustering algorithm algorithm
        controller = Core.Controller(generalise_params)
        controller_start = time.time()
        result_dict = controller.run(input_vector_database, plot_for_itr, classes, output_loc_images)
        print('Algorithms completed in', round(time.time() - controller_start, 2), '(s)')
        # saved_name = Utils.Utilities.save_object(result_dict, output_save_location + output_save_filename)

        # Display
        display = Display_Utils.Display(result_dict[0]['gsom'], None)
        display.setup_labels_for_gsom_nodemap(labels, 1, 'Names of animals', output_save_location + 'gsom_names_' + str(SF))
        display.setup_labels_for_gsom_nodemap(classes, 2, 'Categories of animals', output_save_location + 'gsom_categories_' + str(SF))
        # display.display()

        print('Visualisation saved in output folder.')

    elif mode == 2:

        result_dict = Utils.Utilities.load_object(pickle_filename)

        print('File is read.')