

"""
Example on page 81 of `David Barber's book Bayesian Reasoning and Machine Learning <http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/090310.pdf>`_
==================================================================================================================================================

"""

#%%
# Import required packages
# ^^^^^^^^^^^^^^^^^^^^^^^^

import numpy as np
import rxMsgPassing.sumProduct


#%%
# Define probability tables
# ^^^^^^^^^^^^^^^^^^^^^^^^^

paGb = np.array([[0.4, 0.0],  # p(a|b)
                 [0.2, 0.1],
                 [0.4, 0.2],
                 [0.0, 0.7]])

pbGcd = np.array([[[0.8, 0.7, 0.9],  # p(b|c,d)
                   [0.5, 0.3, 0.2]],
                  [[0.2, 0.3, 0.1],
                   [0.5, 0.7, 0.8]]])

pc = np.array([0.2, 0.8])  # p(c)

pd = np.array([0.1, 0.3, 0.6])  # p(d)

peGd = np.array([[0.1, 0.7, 0.0],  # p(e|d)
                 [0.1, 0.3, 0.0],
                 [0.2, 0.0, 0.0],
                 [0.3, 0.0, 0.0],
                 [0.3, 0.0, 1.0]])
#%%
# Create factor nodes
# ^^^^^^^^^^^^^^^^^^^

f1_probabilities = paGb
f1_varNames = ["va", "vb"]
f1 = rxMsgPassing.sumProduct.FactorNode(name="f1",
                           probabilities=f1_probabilities,
                           var_names=f1_varNames)
f2_probabilities = pbGcd
f2_varNames = ["vb", "vc", "vd"]
f2 = rxMsgPassing.sumProduct.FactorNode(name="f2",
                           probabilities=f2_probabilities,
                           var_names=f2_varNames)
f3_probabilities = pc
f3_varNames = ["vc"]
f3 = rxMsgPassing.sumProduct.FactorNode(name="f3",
                           probabilities=f3_probabilities,
                           var_names=f3_varNames)
f4_probabilities = peGd
f4_varNames = ["ve", "vd"]
f4 = rxMsgPassing.sumProduct.FactorNode(name="f4",
                           probabilities=f4_probabilities,
                           var_names=f4_varNames)
f5_probabilities = pd
f5_varNames = ["vd"]
f5 = rxMsgPassing.sumProduct.FactorNode(name="f5",
                           probabilities=f5_probabilities,
                           var_names=f5_varNames)

#%%
# Create variable nodes
# ^^^^^^^^^^^^^^^^^^^^^

va = rxMsgPassing.sumProduct.VariableNode(name="va")
vb = rxMsgPassing.sumProduct.VariableNode(name="vb")
vc = rxMsgPassing.sumProduct.VariableNode(name="vc")
vd = rxMsgPassing.sumProduct.VariableNode(name="vd")
ve = rxMsgPassing.sumProduct.VariableNode(name="ve")

#%%
# Link variable nodes to factor nodes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

f1.neighbors = [va, vb]
f2.neighbors = [vb, vc, vd]
f3.neighbors = [vc]
f4.neighbors = [vd, ve]
f5.neighbors = [vd]

#%%
# Link factor nodes to variable nodes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

va.neighbors = [f1]
vb.neighbors = [f1, f2]
vc.neighbors = [f2, f3]
vd.neighbors = [f2, f4, f5]
ve.neighbors = [f4]

#%%
# Computer marginal of a by message passing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

m_a = va.marginal()

#%%
# Computer marginal of a by brute force
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

domain_a = np.arange(paGb.shape[0])
domain_b = np.arange(paGb.shape[1])
domain_c = np.arange(pbGcd.shape[1])
domain_d = np.arange(pbGcd.shape[2])
domain_e = np.arange(peGd.shape[0])

def pabcde(a, b, c, d, e):
    answer = paGb[tuple([a, b])].item() * \
             pbGcd[tuple([b, c, d])].item() * \
             pc[c].item() * \
             pd[d].item() * \
             peGd[tuple([e, d])].item()
    return answer

bf_m_a = [None] * len(domain_a)
for i, a in enumerate(domain_a):
    total = 0.0
    for b in domain_b:
        for c in domain_c:
            for d in domain_d:
                for e in domain_e:
                    total += pabcde(a=a, b=b, c=c, d=d, e=e)
    bf_m_a[i] = total

#%%
# Test agreement between message passing and brute force marginals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tol = 1e-6

for i in range(len(m_a)):
    if abs(m_a[i] - bf_m_a[i]) < tol:
        print(f"Agreement in component {i}")
    else:
        print(f"Disagreement in component {i}")

