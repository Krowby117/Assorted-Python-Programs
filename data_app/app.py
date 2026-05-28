# Small mock-up / proof of concept of a project I want to work on more in the future
# System for me to drop a csv file into and automoatically run some simple analysis on it
# Uploads a csv, displays it, some general information, and a few simple visualizations

# Can be used by running "streamlit run app.py" in the console and visiting the returned link

# TO ADD:
# A data cleaning portion? Would like to write a custom class for this
# Better "general information" section, maybe custom code to get specific info
#   instead of using the build in .describe function
# More / Better visualizations, using pretty basic ones that can be harder to interpret based on the data
# ^^ Maybe an option to select certain visualizations? But would be more annoying to keep the ability to choose
#       what variables are used in each graph.

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.markdown("<h1 style='text-align: center;'>Simple CSV Visualizer</h1>", unsafe_allow_html=True)

# -- ask the user for the file name
file = st.file_uploader("Upload a csv file:", type='csv', accept_multiple_files=False,)

# -- load in the csv file
if file:
    data = pd.read_csv(file)
    data = data.dropna()

    st.markdown("<h2 style='text-align: center;'>File Viewer</h2>", unsafe_allow_html=True)
    st.write(data)
    st.markdown("<h2 style='text-align: center;'>CSV General Information</h2>", unsafe_allow_html=True)
    st.write(data.describe(include='all'))

    # -- ask if they would like to use a version of the csv where each element has been converted to a float
    # maybe add this as an option for graphs that would benefit from it, like the correlation heatmaps
    # placeholderL, placeholderR = st.columns([0.2, 0.8])
    # converToNumber = placeholderR.toggle("Convert all rows to numbers for analysis and visualizations?")
    # if converToNumber:
        # convert non numbers to numbers
        # maybe write a data cleaning class or something?

    # -- do some visualizations
    # grab all of the numeric data for visualizations
    numeric_data = data.select_dtypes(include='number').columns.tolist()
    nonNumeric_data = data.select_dtypes(include='object').columns.tolist()
    
    # histogram
    st.markdown("<h2 style='text-align: center;'>Histogram</h2>", unsafe_allow_html=True)
    histL, histR = st.columns([0.25, 0.8])
    xh = histL.radio (
        "**Choose an X-axis**",
        data.columns,
    )

    if xh:
        fig = px.histogram(data, x=xh)
        histR.plotly_chart(fig)

    # correlation heat map
    st.markdown("<h2 style='text-align: center;'>Correlation Heatmap on Numeric Values</h2>", unsafe_allow_html=True)
    corMatrix = data.select_dtypes(include='number').corr()
    fig = px.imshow(
        corMatrix,
        text_auto=True,                 # display values
        color_continuous_scale='RdBu',
        aspect="auto",
    )
    st.plotly_chart(fig)

    # bar chart
    st.markdown("<h2 style='text-align: center;'>Bar Chart</h2>", unsafe_allow_html=True)
    left_side, right_side = st.columns([0.25, 0.8])
    xb = left_side.radio (
        "**Choose an X-axis**",
        nonNumeric_data,
    )
    yb = left_side.radio (
        "**Choose a Y-axis**",
        numeric_data,
    )

    if xb and yb:
        grouped = data.groupby(xb)[yb].mean().reset_index()

        fig = go.Figure(data=[go.Bar(x=grouped[xb], y=grouped[yb])])
        fig.update_layout(xaxis_title=xb, yaxis_title=yb)

        right_side.plotly_chart(fig, use_container_width=True)

    # pie chart
    st.markdown("<h2 style='text-align: center;'>Pie Chart</h2>", unsafe_allow_html=True)
    pieL, pieR = st.columns([0.5, 0.5])
    pVal = pieR.selectbox (
        "**Choose a Value**",
        numeric_data,
    )
    pName = pieL.selectbox (
        "**Choose a Name**",
        nonNumeric_data,
    )

    if pVal and pName:
        fig = px.pie(data, values=pVal, names=pName)
        st.plotly_chart(fig)

    # scatter plot
    st.markdown("<h2 style='text-align: center;'>Scatter Plot</h2>", unsafe_allow_html=True)
    scatterL, scatterR = st.columns([0.5, 0.5])
    xs = scatterL.pills (
        "**Choose an X-axis**",
        data.columns,
        selection_mode="single",
    )
    ys = scatterR.pills (
        "**Choose a Y-axis**",
        data.columns,
        selection_mode="single",
    )

    if xs and ys:
        fig = px.scatter(x=data.get(xs), y=data.get(ys))
        fig.update_layout(xaxis_title=xs, yaxis_title=ys)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown("<h2 style='text-align: center;'>Upload a CSV to view it's contents and visualizations!</h2>", unsafe_allow_html=True)
