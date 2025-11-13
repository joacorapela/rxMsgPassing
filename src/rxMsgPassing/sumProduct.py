
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

        out_msg = np.ones(self._neighbors[0].cardinality(self.name),
                          dtype=np.double)
        for g in self.neighbors:
            if g.name != factor_name:
                out_msg *= g.msg(self.name)
        self._msgs[factor_name] = out_msg
        print((f"Computed msg from variable {self._name} "
               f"to factor {factor_name}: {out_msg}"))
        return out_msg

    def marginal(self):
        marginal = np.ones(self._neighbors[0].cardinality(self.name),
                           dtype=np.double)
        for g in self.neighbors:
            marginal *= g.msg(self.name)
        return marginal

class FactorNode:
    def __init__(self, name, probabilities, var_names, conditional=None):
        self._name = name
        self._probabilities = probabilities
        self._var_names = var_names
        self._conditional = conditional
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
        for neighbor in self.neighbors:
            if neighbor.name != var_name:
                in_msg = neighbor.msg(self.name)
                in_msgs_contents.append(in_msg)
                in_msgs_names.append(neighbor.name)
                in_msgs_domains.append([i for i in range(len(in_msg))])

        # generate output message
        # tricky due to the nested loop of arbitrary depth
        out_msg = np.ones(self.cardinality(var_name), dtype=np.double)
        var_names_values = list(range(len(out_msg)))
        dict_keys = [var_name] + in_msgs_names
        for i, var_name_value in enumerate(var_names_values):
            if len(in_msgs_domains) > 0:
                total = 0.0
                for xs in itertools.product(*in_msgs_domains):
                    msg_values = [in_msgs_contents[i][x]
                                  for i, x in enumerate(xs)]
                    prod_in_msgs = np.prod(msg_values)
                    dict_values = [var_name_value] + list(xs)
                    factor_value = self._get_prob_value(
                        vars_dict=dict(zip(dict_keys, dict_values)))
                    total += factor_value * prod_in_msgs
            else:
                factor_value = self._probabilities[i]
                total = factor_value
            out_msg[i] = total

        self._msgs[var_name] = out_msg
        print((f"Computed msg from factor {self._name} "
               f"to variable {var_name}: {out_msg}"))
        return out_msg

    def _get_prob_value(self, vars_dict):
        conditional_factor = 1.0
        indices = [None] * len(self._var_names)
        for i in range(len(indices)):
            var_name = self._var_names[i]
            indices[i] = vars_dict[var_name]
            if self._conditional is not None and var_name in self._conditional:
                if self._conditional[var_name] != vars_dict[var_name]:
                    conditional_factor = 0.0
                    break
        if conditional_factor == 0.0:
            return 0.0
        else:
            prob_value = self._probabilities[tuple(indices)] * conditional_factor
            return prob_value
