

"""
Example of calculation of conditional probability mass function. Based on Figure 5.3 of `David Barber's book Bayesian Reasoning and Machine Learning <http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/090310.pdf>`_
====================================================================================================================================================================================================================

**Task.** Compute the marginal PMF :math:`p(a|b=1)` for the model

.. math::

   p(a,b,c,d,e) \\;=\\; p(a\\mid b)\\; p(b\\mid c,d)\\; p(e\\mid d)\\; p(c)\\; p(d).

Use the factor graph in :numref:`fig-brml-53` and the factor tables in
:numref:`tab-var-dim`–:numref:`tab-pd`. Report a normalized :math:`p(a)`.

.. _fig-brml-53:

.. figure:: /images/fig5_3_BRML.png
   :width: 100%
   :alt: Factor graph for variables a, b, c, d, e (BRML Fig. 5.3)

   Factor graph for the model.

.. _tab-var-dim:

.. list-table:: Variable dimensions
   :header-rows: 1
   :align: left

   * - Var
     - Dim
   * - a
     - 4
   * - b
     - 2
   * - c
     - 2
   * - d
     - 3
   * - e
     - 5

.. _tab-pab:

.. list-table:: Conditional PMF :math:`p(a\\mid b)`
   :header-rows: 1
   :align: left

   * - :math:`a\\backslash b`
     - 0
     - 1
   * - 0
     - 0.4
     - 0.0
   * - 1
     - 0.2
     - 0.1
   * - 2
     - 0.4
     - 0.2
   * - 3
     - 0.0
     - 0.7

.. _tab-pbcd-0:

.. list-table:: Conditional PMF :math:`p(b\\mid c,d{=}0)`
   :header-rows: 1
   :align: left

   * - :math:`b\\backslash c`
     - 0
     - 1
   * - 0
     - 0.8
     - 0.5
   * - 1
     - 0.2
     - 0.5

.. _tab-pbcd-1:

.. list-table:: Conditional PMF :math:`p(b\\mid c,d{=}1)`
   :header-rows: 1
   :align: left

   * - :math:`b\\backslash c`
     - 0
     - 1
   * - 0
     - 0.7
     - 0.3
   * - 1
     - 0.3
     - 0.7

.. _tab-pbcd-2:

.. list-table:: Conditional PMF :math:`p(b\\mid c,d{=}2)`
   :header-rows: 1
   :align: left

   * - :math:`b\\backslash c`
     - 0
     - 1
   * - 0
     - 0.9
     - 0.2
   * - 1
     - 0.1
     - 0.9

.. _tab-ped:

.. list-table:: Conditional PMF :math:`p(e\\mid d)`
   :header-rows: 1
   :align: left

   * - :math:`e\\backslash d`
     - 0
     - 1
     - 2
   * - 0
     - 0.1
     - 0.7
     - 0.0
   * - 1
     - 0.1
     - 0.3
     - 0.0
   * - 2
     - 0.2
     - 0.0
     - 0.0
   * - 3
     - 0.3
     - 0.0
     - 0.0
   * - 4
     - 0.3
     - 0.0
     - 1.0

.. _tab-pc:

.. list-table:: PMF :math:`p(c)`
   :header-rows: 1
   :align: left

   * - :math:`c`
     - :math:`p(c)`
   * - 0
     - 0.2
   * - 1
     - 0.8

.. _tab-pd:

.. list-table:: PMF :math:`p(d)`
   :header-rows: 1
   :align: left

   * - :math:`d`
     - :math:`p(d)`
   * - 0
     - 0.1
   * - 1
     - 0.3
   * - 2
     - 0.6
"""

#%%
# Import required packages
# ^^^^^^^^^^^^^^^^^^^^^^^^

import time
import numpy as np
import plotly.graph_objects as go
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
f1 = rxMsgPassing.sumProduct.FactorNode(
    name="f1", probabilities=f1_probabilities, var_names=f1_varNames,
    conditional={"vb": 1})
f2_probabilities = pbGcd
f2_varNames = ["vb", "vc", "vd"]
f2 = rxMsgPassing.sumProduct.FactorNode(
    name="f2", probabilities=f2_probabilities, var_names=f2_varNames,
    conditional={"vb": 1})
f3_probabilities = pc
f3_varNames = ["vc"]
f3 = rxMsgPassing.sumProduct.FactorNode(
    name="f3", probabilities=f3_probabilities, var_names=f3_varNames)
f4_probabilities = peGd
f4_varNames = ["ve", "vd"]
f4 = rxMsgPassing.sumProduct.FactorNode(
    name="f4", probabilities=f4_probabilities, var_names=f4_varNames)
f5_probabilities = pd
f5_varNames = ["vd"]
f5 = rxMsgPassing.sumProduct.FactorNode(
    name="f5", probabilities=f5_probabilities, var_names=f5_varNames)

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
# Compute marginal of a by message passing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

start_time = time.perf_counter()
aux = va.marginal()
mp_m_a_g_b1 = aux / np.sum(aux)
mp_elapsed_time = time.perf_counter() - start_time 
print(f"message passing: p(a|b=1)={mp_m_a_g_b1}")
print(f"message passing: elapsed_time={mp_elapsed_time}")

#%%
# Computer marginal of a by brute force
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

domain_a = np.arange(paGb.shape[0])
domain_b = np.arange(pbGcd.shape[0])
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

start_time = time.perf_counter()

bf_m_a_g_b1 = [None] * len(domain_a)
for i, a in enumerate(domain_a):
    total = 0.0
    for c in domain_c:
        for d in domain_d:
            for e in domain_e:
                total += pabcde(a=a, b=1, c=c, d=d, e=e)
    bf_m_a_g_b1[i] = total
bf_m_a_g_b1 = bf_m_a_g_b1 / np.sum(bf_m_a_g_b1)

bf_elapsed_time = time.perf_counter() - start_time

print(f"brute force: p(a|b=1)={bf_m_a_g_b1}")
print(f"brute force: elapsed_time={bf_elapsed_time}")

#%%
# Plot marginals computed by message passing and brute force
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig = go.Figure()
trace = go.Bar(y=mp_m_a_g_b1, name="Message Passing")
fig.add_trace(trace)
trace = go.Bar(y=bf_m_a_g_b1, name="Brute Force")
fig.add_trace(trace)
fig.update_xaxes(title="x")
fig.update_yaxes(title="p(a=x)")
fig

#%%
# Test agreement between message passing and brute force marginals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tol = 1e-6

for i in range(len(mp_m_a_g_b1)):
    if abs(mp_m_a_g_b1[i] - bf_m_a_g_b1[i]) < tol:
        print(f"Agreement in component {i}")
    else:
        print(f"Disagreement in component {i}")
