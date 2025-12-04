# Simple python project that creates discrete bayesian network using the pgmpy
# Most of the code was reused from another small project in this repo "dbn_titanic.ipynb"
# This project was mostly to mess around with the streamlit library
# can be seen by running "streamlit run st_test.py" in the console and visiting the returned link

import streamlit as st
import pandas as pd
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BDeu
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# -- helper function that converts the choice made into the discretized version -- #
def handle_choice(x):
    if x in ["low", "0", "male", "younger"]:
        return 1
    elif x in ["medium", "1", "female", "middle aged"]:
        return 2
    else:
        return 3

# -- helper function that converts the survived label into a string for printing -- #
def handle_survived(x):
    if x == 1:
        return "perished."
    else:
        return "survived!"

# --------- Setup the streamlit site
st.title("Discrete Bayesian Network on the Titanic dataset");

# --------- Load in and display dataset
data = pd.read_csv("titanic.csv")
cols = ["age","numparentschildren","passengerclass","sex","numsiblings","survived"]
data = data[cols].dropna()
st.header("Titanic dataset:")
st.write(data);

# --- Learn network structure using Hill Climb + BDeu score
hc = HillClimbSearch(data)
score = BDeu(data)
best_model = hc.estimate(scoring_method=score)

# --- Fit the parameters
model = DiscreteBayesianNetwork(best_model.edges())
model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")

# --------- Display the DBN structure on streamlit
st.header("Learned DBN Structure:")

# -- this converts it to a networkx image and displays it
G = nx.DiGraph()
G.add_edges_from(model.edges())

pos = nx.spring_layout(G, k=2, iterations=200, seed=42)

fig, ax = plt.subplots(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_size=2500, font_size=12, arrows=True, arrowstyle='-|>', arrowsize=20, width=2, ax=ax)

st.pyplot(fig)

# --------- Added section to choose attributes and queery the survivability
infer = VariableElimination(model)
attributes = ["age","numparentschildren","passengerclass","sex","numsiblings"]
options = {
    "age": ["younger", "middle aged", "older"],
    "numparentschildren": ["0", "1", "2+"],
    "passengerclass": ["0", "1", "2"],
    "sex": ["male", "female"],
    "numsiblings": ["0", "1", "2+"],
}

st.header("Choose attribute values to see their effect:")
left_side, middle, right_side = st.columns([0.4, 0.1, 0.5])

choices = {}

for i in attributes:
    with left_side:
        choices[i] = st.selectbox(
            f"{i}",
            options[i],
            key=i
        )

with right_side:
    st.write("#### Chosen Attributes:")
    st.write(choices)

    ev = {}
    for i in choices:
        ev[i] = handle_choice(choices[i])
    
    q = infer.query(variables=['survived'], evidence=ev)
    st.write("#### Your Survivability:")

    states = q.values
    max_index = np.argmax(states)
    msg = f"{round(states[max_index] * 100, 2)}% chance that you {handle_survived(q.state_names['survived'][max_index])}"
    st.write(msg)