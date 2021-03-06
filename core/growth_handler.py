import numpy as np
from util import utilities as Utils
from core import elements as Elements


class GrowthHandler:
    def grow_nodes(self, node_map, winner):
        x = winner.x
        y = winner.y

        self._grow_individual_node(x - 1, y, winner, node_map)
        self._grow_individual_node(x + 1, y, winner, node_map)
        self._grow_individual_node(x, y - 1, winner, node_map)
        self._grow_individual_node(x, y + 1, winner, node_map)

    def _grow_individual_node(self, x, y, winner, node_map):

        new_node_index = Utils.Utilities.generate_index(x, y)

        if new_node_index not in node_map:
            node_map[new_node_index] = Elements.GSOMNode(x, y, self._generate_new_node_weights(node_map, winner, x, y))

    def _generate_new_node_weights(self, node_map, winner, x, y):

        if winner.y == y:

            # W1 is the winner in following cases
            # W1 - W(new)
            if x == winner.x + 1:

                next_node_str = Utils.Utilities.generate_index(x + 1, y)
                other_side_node_str = Utils.Utilities.generate_index(x - 2, y)
                top_node_srt = Utils.Utilities.generate_index(winner.x, y + 1)
                bottom_node_str = Utils.Utilities.generate_index(winner.x, y - 1)

                """
                 * 1. W1 - W(new) - W2
                 * 2. W2 - W1 - W(new)
                 * 3. W2
                 *    |
                 *    W1 - W(new)
                 * 4. W1 - W(new)
                 *    |
                 *    W2
                 * 5. W1 - W(new)
                """
                new_weights = self._get_new_node_weights_in_xy_axis(node_map, winner, next_node_str,
                                                                    other_side_node_str, top_node_srt, bottom_node_str)

            # W(new) - W1
            elif x == winner.x - 1:

                next_node_str = Utils.Utilities.generate_index(x - 1, y)
                other_side_node_str = Utils.Utilities.generate_index(x + 2, y)
                top_node_srt = Utils.Utilities.generate_index(winner.x, y + 1)
                bottom_node_str = Utils.Utilities.generate_index(winner.x, y - 1)

                """
                 * 1. W2 - W(new) - W1
                 * 2. W(new) - W1 - W2
                 * 3.          W2
                 *             |
                 *    W(new) - W1
                 * 4. W(new) - W1
                 *              |
                 *              W2
                 * 5. W(new) - W1
                """

                new_weights = self._get_new_node_weights_in_xy_axis(node_map, winner, next_node_str,
                                                                    other_side_node_str, top_node_srt, bottom_node_str)

        elif winner.x == x:

            """            
            * W(new)
            * |
            * W1
            """
            if y == winner.y + 1:

                next_node_str = Utils.Utilities.generate_index(x, y + 1)
                other_side_node_str = Utils.Utilities.generate_index(x, y - 2)
                left_node_srt = Utils.Utilities.generate_index(x - 1, winner.y)
                right_node_str = Utils.Utilities.generate_index(x + 1, winner.y)

                new_weights = self._get_new_node_weights_in_xy_axis(node_map, winner, next_node_str,
                                                                    other_side_node_str, left_node_srt, right_node_str)

            elif y == winner.y - 1:

                next_node_str = Utils.Utilities.generate_index(x, y - 1)
                other_side_node_str = Utils.Utilities.generate_index(x, y + 2)
                left_node_srt = Utils.Utilities.generate_index(x - 1, winner.y)
                right_node_str = Utils.Utilities.generate_index(x + 1, winner.y)

                new_weights = self._get_new_node_weights_in_xy_axis(node_map, winner, next_node_str,
                                                                    other_side_node_str, left_node_srt, right_node_str)

        # new_weights[new_weights < 0] = 0.0
        # new_weights[new_weights > 1] = 1.0

        return new_weights

    def _get_new_node_weights_in_xy_axis(self, node_map, winner, next_node_str, other_side_node_str,
                                         top_or_left_node_srt, bottom_or_right_node_str):

        if next_node_str in node_map:
            new_weights = self._new_weights_for_new_node_in_middle(node_map, winner, next_node_str)
        elif other_side_node_str in node_map:
            new_weights = self._new_weights_for_new_node_on_one_side(node_map, winner, other_side_node_str)
        elif top_or_left_node_srt in node_map:
            new_weights = self._new_weights_for_new_node_on_one_side(node_map, winner, top_or_left_node_srt)
        elif bottom_or_right_node_str in node_map:
            new_weights = self._new_weights_for_new_node_on_one_side(node_map, winner, bottom_or_right_node_str)
        else:
            new_weights = self._new_weights_for_new_node_one_older_neighbour(winner)

        return new_weights

    def _new_weights_for_new_node_in_middle(self, node_map, winner, next_node_str):
        return (winner.weights + node_map[next_node_str].weights) * 0.5

    def _new_weights_for_new_node_on_one_side(self, node_map, winner, next_node_str):
        return (winner.weights * 2) - node_map[next_node_str].weights

    def _new_weights_for_new_node_one_older_neighbour(self, winner):
        return np.full(len(winner.weights), (max(winner.weights) + min(winner.weights)) / 2)