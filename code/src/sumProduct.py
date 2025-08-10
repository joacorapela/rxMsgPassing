
import itertools
import numpy as np


class VariableNode:
    def __init__(self, name):
        self._name = name
        self._msgs = {}

    @property
    def name(self):
        return self._name

    @property
    def neighbors(self):
        return self._neighbors

    @neighbors.setter
    def neighbors(self, value):
        self._neighbors = value

    def msg(self, factor_name):
        if factor_name in self._msgs.keys():
            return self._msgs[factor_name]

        answer = np.ones(self._neighbors[0].cardinality(self.name))
        for g in self.neighbors():
            if g.name() != factor_name:
                answer *= g.msg(self.name)
        return answer


class FactorNode:
    def __init__(self, name, probabilities, var_names):
        self._name = name
        self._probabilities = probabilities
        self._var_names = var_names
        self._msgs = {}

    @property
    def name(self):
        return self._name

    @property
    def neighbors(self):
        return self._neighbors

    @neighbors.setter
    def neighbors(self, value):
        self._neighbors = value

    def cardinality(self, var_name):
        answer = self._probabilities.shape[self._var_names.index(var_name)]
        return answer

    def msg(self, var_name):
        if var_name in self._msgs.keys():
            return self._msgs[var_name]

        # collect input messages
        in_msgs_contents = []
        in_msgs_names = []
        in_msgs_domains = []
        for neighbor in self.neighbors():
            if neighbor.name != var_name:
                in_msg = neighbor.msg(self.name)
                in_msgs_contents.append(in_msg)
                in_msgs_names.append(neighbor.name)
                in_msgs_domains.append([i for i in range(len(in_msg))])

        # generate output message (challenging due to the nested loop of
        # arbitrary depth)
        out_msg = np.empty(shape=len(self._var_names), dtype=np.double)
        i_var_name = self._var_names.index(var_name)
        var_names_values = np.arange(self._probabilities.shape[i_var_name])
        for i, var_name_value in enumerate(var_names_values):
            if len(in_msgs_domains) > 0:
                total = 0.0
                for xs in itertools.product(*in_msgs_domains):
                    keys = [var_name].append(in_msgs_names)
                    msg_values = [in_msgs_contents[i][x] for i, x in enumerate(xs)]
                    values = [var_name_value].append(msg_values)
                    prod_in_msgs = np.prod(msg_values)
                    factor_value = self._get_prob_value(
                        vars_dict=zip(keys, values))
                    total += factor_value * prod_in_msgs
            else:
                factor_value = self._probabilities[i]
                total = factor_value
            out_msg[i] = total

        self._msgs[var_name] = out_msg
        return out_msg

    def _get_prob_value(self, vars_dict):
        indices = [None] * len(self._var_names)
        for i in range(len(indices)):
            var_name = self._var_names[i]
            indices[i] = vars_dict[var_name]
        prob_value = self._probabilities[tuple(indices)]
        return prob_value
