import numpy as np
import rxMsgPassing.sumProduct
import utils


def test_marginals(tol=1e-4):
    # create factor nodes
    f1_probabilities = utils.paGb
    f1_varNames = ["va", "vb"]
    f1 = rxMsgPassing.sumProduct.FactorNode(name="f1",
                               probabilities=f1_probabilities,
                               var_names=f1_varNames)
    f2_probabilities = utils.pbGcd
    f2_varNames = ["vb", "vc", "vd"]
    f2 = rxMsgPassing.sumProduct.FactorNode(name="f2",
                               probabilities=f2_probabilities,
                               var_names=f2_varNames)
    f3_probabilities = utils.pc
    f3_varNames = ["vc"]
    f3 = rxMsgPassing.sumProduct.FactorNode(name="f3",
                               probabilities=f3_probabilities,
                               var_names=f3_varNames)
    f4_probabilities = utils.peGd
    f4_varNames = ["ve", "vd"]
    f4 = rxMsgPassing.sumProduct.FactorNode(name="f4",
                               probabilities=f4_probabilities,
                               var_names=f4_varNames)
    f5_probabilities = utils.pd
    f5_varNames = ["vd"]
    f5 = rxMsgPassing.sumProduct.FactorNode(name="f5",
                               probabilities=f5_probabilities,
                               var_names=f5_varNames)

    # create variable nodes
    va = rxMsgPassing.sumProduct.VariableNode(name="va")
    vb = rxMsgPassing.sumProduct.VariableNode(name="vb")
    vc = rxMsgPassing.sumProduct.VariableNode(name="vc")
    vd = rxMsgPassing.sumProduct.VariableNode(name="vd")
    ve = rxMsgPassing.sumProduct.VariableNode(name="ve")

    # link variable nodes to factor nodes
    f1.neighbors = [va, vb]
    f2.neighbors = [vb, vc, vd]
    f3.neighbors = [vc]
    f4.neighbors = [vd, ve]
    f5.neighbors = [vd]

    # link factor nodes to variable nodes
    va.neighbors = [f1]
    vb.neighbors = [f1, f2]
    vc.neighbors = [f2, f3]
    vd.neighbors = [f2, f4, f5]
    ve.neighbors = [f4]

    # compute marginals
    m_a = va.marginal()
    m_b = vb.marginal()
    m_c = vc.marginal()
    m_d = vd.marginal()
    m_e = ve.marginal()

    # compute marginals by brute force
    domain_a = np.arange(utils.paGb.shape[0])
    domain_b = np.arange(utils.paGb.shape[1])
    domain_c = np.arange(utils.pbGcd.shape[1])
    domain_d = np.arange(utils.pbGcd.shape[2])
    domain_e = np.arange(utils.peGd.shape[0])

    def pabcde(a, b, c, d, e):
        answer = utils.paGb[tuple([a, b])].item() * \
                 utils.pbGcd[tuple([b, c, d])].item() * \
                 utils.pc[c].item() * \
                 utils.pd[d].item() * \
                 utils.peGd[tuple([e, d])].item()
        return answer

    # brute force marginal for a
    bf_m_a = [None] * len(domain_a)
    for i, a in enumerate(domain_a):
        total = 0.0
        for b in domain_b:
            for c in domain_c:
                for d in domain_d:
                    for e in domain_e:
                        total += pabcde(a=a, b=b, c=c, d=d, e=e)
        bf_m_a[i] = total
    # assert equality between message passing and brute force marginals
    for i in range(len(m_a)):
        assert(abs(m_a[i] - bf_m_a[i]) < tol)

    # brute force marginal for b
    bf_m_b = [None] * len(domain_b)
    for i, b in enumerate(domain_b):
        total = 0.0
        for a in domain_a:
            for c in domain_c:
                for d in domain_d:
                    for e in domain_e:
                        total += pabcde(a=a, b=b, c=c, d=d, e=e)
        bf_m_b[i] = total
    # assert equality between message passing and brute force marginals
    for i in range(len(m_b)):
        assert(abs(m_b[i] - bf_m_b[i]) < tol)

    # brute force marginal for c
    bf_m_c = [None] * len(domain_c)
    for i, c in enumerate(domain_c):
        total = 0.0
        for a in domain_a:
            for b in domain_b:
                for d in domain_d:
                    for e in domain_e:
                        total += pabcde(a=a, b=b, c=c, d=d, e=e)
        bf_m_c[i] = total
    # assert equality between message passing and brute force marginals
    for i in range(len(m_c)):
        assert(abs(m_c[i] - bf_m_c[i]) < tol)

    # brute force marginal for d
    bf_m_d = [None] * len(domain_d)
    for i, d in enumerate(domain_d):
        total = 0.0
        for a in domain_a:
            for b in domain_b:
                for c in domain_c:
                    for e in domain_e:
                        total += pabcde(a=a, b=b, c=c, d=d, e=e)
        bf_m_d[i] = total
    # assert equality between message passing and brute force marginals
    for i in range(len(m_d)):
        assert(abs(m_d[i] - bf_m_d[i]) < tol)

    # brute force marginal for e
    bf_m_e = [None] * len(domain_e)
    for i, e in enumerate(domain_e):
        total = 0.0
        for a in domain_a:
            for b in domain_b:
                for c in domain_c:
                    for d in domain_d:
                        total += pabcde(a=a, b=b, c=c, d=d, e=e)
        bf_m_e[i] = total
    # assert equality between message passing and brute force marginals
    for i in range(len(m_e)):
        assert(abs(m_e[i] - bf_m_e[i]) < tol)
