from core import gsom as GSOM_Core


class Controller:

    def __init__(self, params):
        self.params = params
        self.gsom_nodemap = None

    def _grow_gsom(self, inputs, dimensions):
        gsom = GSOM_Core.GSOM(self.params.get_gsom_parameters(), inputs, dimensions)
        gsom.grow()
        gsom.smooth()
        gsom.assign_hits()
        self.gsom_nodemap = gsom.evaluate_hits(inputs)

    def run(self, input_vector_db):

        results = []

        for batch_key, batch_vector_weights in input_vector_db.items():

            self._grow_gsom(batch_vector_weights, batch_vector_weights.shape[1])

            results.append({
                'gsom': self.gsom_nodemap,
            })

        return results
