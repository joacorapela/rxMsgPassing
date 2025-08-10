
import sumProduct
import utils


def main(argv):
    # create factor nodes
    f1_probabilities = utils.paGb
    f1_varNames = ["a", "b"]
    f1 = sumProduct.FactorNode(name="f1",
                               probabilities=f1_probabilities,
                               var_names=f1_varNames)
    f2_probabilities = utils.pbGcd
    f2_varNames = ["b", "c", "d"]
    f2 = sumProduct.FactorNode(name="f2",
                               probabilities=f2_probabilities,
                               var_names=f2_varNames)
    f3_probabilities = utils.pc
    f3_varNames = ["c"]
    f3 = sumProduct.FactorNode(name="f3",
                               probabilities=f3_probabilities,
                               var_names=f3_varNames)
    f4_probabilities = utils.pd
    f4_varNames = ["d"]
    f4 = sumProduct.FactorNode(name="f4",
                               probabilities=f4_probabilities,
                               var_names=f4_varNames)
    f5_probabilities = utils.peGd
    f5_varNames = ["e", "d"]
    f5 = sumProduct.FactorNode(name="f5",
                               probabilities=f5_probabilities,
                               var_names=f5_varNames)

    # create variable nodes
    va = sumProduct.VariableNode(name="va")
    vb = sumProduct.VariableNode(name="vb")
    vc = sumProduct.VariableNode(name="vc")
    vd = sumProduct.VariableNode(name="vd")
    ve = sumProduct.VariableNode(name="ve")

    # link variable nodes to factor nodes
    f1.neighbors = [va, vb]
    f2.neighbors = [vb, vc, vd]
    f3.neighbors = [vc]
    f4.neighbors = [vd]
    f5.neighbors = [vd, ve]

    # link factor nodes to variable nodes
    va.neighbors = [f1]
    vb.neighbors = [f1, f2]
    vc.neighbors = [f2, f3]
    vd.neighbors = [f2, f4, f5]
    ve.neighbors = [f4]

    # compute and print marginals
    pa = f1.msg(var_name="a")
    print(f"pa={pa}")
