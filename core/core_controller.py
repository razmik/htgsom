import time
from core2 import gsom as GSOM_Core
from core2 import generalisation_layer as Core_Generalisation_Layer
from util import utilities as Utils
from core2 import elements as Elements


class Controller:

    def __init__(self, params):
        self.params = params
        self.gsom_nodemap = None
        self.generalisation_layer = None

    def _grow_gsom(self, inputs, dimensions):
        gsom = GSOM_Core.GSOM(self.params.get_gsom_parameters(), inputs, dimensions)
        gsom.grow()
        gsom.smooth()
        gsom.assign_hits()
        self.gsom_nodemap = gsom.evaluate_hits(inputs)

    def _generalise(self, batch_id, inputs, dimensions):
        self.generalisation_layer = Core_Generalisation_Layer.GeneralisationLayer(batch_id,
                                                                                  self.gsom_nodemap, self.params,
                                                                                  len(inputs),
                                                                                  dimensions)
        self.generalisation_layer.generalise()

    def _map_input_to_aggregated_nodes(self, inputs):
        for idx, input_vector in enumerate(inputs):
            Utils.Utilities.select_input_to_closest_aggregate_node(self.generalisation_layer.get_generalised_nodemap(),
                                                                   Elements.InputWeight(input_vector, idx),
                                                                   self.params.get_gsom_parameters().DISTANCE_FUNCTION,
                                                                   self.params.get_gsom_parameters().DISTANCE_DIVIDER)

    def run(self, input_vector_db):

        results = []

        for batch_key, batch_vector_weights in input_vector_db.items():

            batch_id = int(batch_key)

            start_time = time.time()

            # Generate GSOM Node map
            self._grow_gsom(batch_vector_weights, batch_vector_weights.shape[1])
            print('Batch', batch_id, 'Learning Layer built with', len(self.gsom_nodemap), 'nodes')

            # Aggregate the nodes
            self._generalise(batch_id, batch_vector_weights, batch_vector_weights.shape[1])
            print('Sq:', batch_id, 'Generalisation Layer built with',
                  len(self.generalisation_layer.get_generalised_nodemap()), 'nodes')

            # Construct the input_weight_vectors relevant to each aggregate node using distance of input to aggr weight.
            if len(self.generalisation_layer.get_generalised_nodemap()) > 0:
                self._map_input_to_aggregated_nodes(batch_vector_weights)

            print('Generalisation', batch_id, 'completed in', round(time.time() - start_time, 2), '(s)\n')

            results.append({
                'gsom': self.gsom_nodemap,
                'aggregated': self.generalisation_layer.get_generalised_nodemap()
            })

        return results

    def generalise_gsom_map(self, input_vector_db, gsom_nodemap):

        results = []
        self.gsom_nodemap = gsom_nodemap

        for batch_key, batch_vector_weights in input_vector_db.items():

            batch_id = int(batch_key)

            start_time = time.time()

            # Aggregate the nodes
            self._generalise(batch_id, batch_vector_weights, batch_vector_weights.shape[1])
            print('Sq:', batch_id, 'Generalisation Layer built with',
                  len(self.generalisation_layer.get_generalised_nodemap()), 'nodes')

            # Construct the input_weight_vectors relevant to each aggregate node using distance of input to aggr weight.
            if len(self.generalisation_layer.get_generalised_nodemap()) > 0:
                self._map_input_to_aggregated_nodes(batch_vector_weights)

            print('Generalisation', batch_id, 'completed in', round(time.time() - start_time, 2), '(s)\n')

            results.append({
                'gsom': self.gsom_nodemap,
                'aggregated': self.generalisation_layer.get_generalised_nodemap()
            })

        return results
