import streamlit as st
import plotly.express as px

def show_rela(numeric_data, nonNumeric_data, data):
        # ------ correlation heat map ------
        st.markdown("<h2 style='text-align: center;'>Correlation Heatmap on Numeric Values</h2>", unsafe_allow_html=True)
        corMatrix = data.select_dtypes(include='number').corr()
        fig = px.imshow(
            corMatrix,
            text_auto=True,                 # display values
            color_continuous_scale='RdBu',
            aspect="auto",
        )
        st.plotly_chart(fig)

        # ------ scatter plot ------
        # probably needs a re-work but this largely dependent on how the data cleaning is handled
        st.markdown("<h2 style='text-align: center;'>Scatter Plot</h2>", unsafe_allow_html=True)
        scatterL, scatterR = st.columns([0.5, 0.5])
        xs = scatterL.selectbox (
            "**Scatter Plot X-axis**",
            data.columns,
        )
        ys = scatterR.selectbox (
            "**Scatter Plot Y-axis**",
            data.columns,
        )

        if xs and ys:
            fig = px.scatter(data, x=xs, y=ys)
            fig.update_layout(xaxis_title=xs, yaxis_title=ys)
            st.plotly_chart(fig, width='stretch')