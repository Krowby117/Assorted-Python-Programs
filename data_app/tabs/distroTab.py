import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def show_dist(numeric_data, nonNumeric_data, data):
        # ------ histogram ------
        st.markdown("<h2 style='text-align: center;'>Histogram</h2>", unsafe_allow_html=True)
        histL, histM, histR, histRR = st.columns([0.10, 0.40, 0.40, 0.10])
        xh = histM.selectbox (
            "**Choose an X-axis**",
            data.columns,
        )

        bins = histR.slider("Number of bins", 5, 100, 20)

        if xh:
            fig = px.histogram(data, x=xh, nbins=bins)
            st.plotly_chart(fig)

        # ------ bar chart ------
        st.markdown("<h2 style='text-align: center;'>Bar Chart</h2>", unsafe_allow_html=True)
        barL, barM, barR = st.columns([0.33, 0.33, 0.33])
        xb = barL.selectbox (
            "**Bar Chart X-axis**",
            nonNumeric_data,
        )
        yb = barM.selectbox (
            "**Bar Chart Y-axis**",
            numeric_data,
        )

        agg = barR.selectbox("Aggregation", ["mean", "sum", "count"])


        if xb and yb:
            grouped = data.groupby(xb)[yb].agg(agg).reset_index()

            fig = go.Figure(data=[go.Bar(x=grouped[xb], y=grouped[yb])])
            fig.update_layout(xaxis_title=xb, yaxis_title=yb)

            st.plotly_chart(fig, width='content')

        # ------ pie chart ------
        # make this cleaner, especially for larger amounts of items
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