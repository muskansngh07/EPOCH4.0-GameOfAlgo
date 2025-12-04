import streamlit as st
import matplotlib.pyplot as plt

from utils.datasets import get_dataset
from visuals.linear_regression_visualizer import LinearRegressionVisualizer

st.title("ML Algorithm Visualizer")

algorithm = st.selectbox("Select Algorithm", ["Linear Regression"])
dataset_name = st.selectbox("Select Dataset", ["Linear", "Blobs", "Moons"])
lr = st.slider(
    "Learning Rate",
    min_value=0.0001,
    max_value=1.0,
    value=0.01,
    step=0.0001  # very fine control
)


X, y = get_dataset(dataset_name)
model = LinearRegressionVisualizer(X, y, lr=lr)

if "model_state" not in st.session_state:
    st.session_state.model = model
    st.session_state.running = False

col1, col2, col3 = st.columns(3)

if col1.button("Step"):
    st.session_state.model.step()

if col2.button("Auto Run"):
    st.session_state.running = True

if col3.button("Reset"):
    st.session_state.model = LinearRegressionVisualizer(X, y, lr=lr)
    st.session_state.running = False

if st.session_state.running:
    for _ in range(5):
        st.session_state.model.step()

fig, ax = plt.subplots()
st.session_state.model.plot(ax)
st.pyplot(fig)
