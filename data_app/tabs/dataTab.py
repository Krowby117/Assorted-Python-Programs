import streamlit as st

def show_data(data):

    # show the file
    st.markdown("<h2 style='text-align: center;'>File Viewer</h2>", unsafe_allow_html=True)
    st.dataframe(data.head(100))
    #st.write(data)

    ## display some information about the file
    st.subheader("Data Quality Overview")

    st.write("Missing values per column")
    st.dataframe(data.isna().sum())

    st.write("Column types")
    st.write(data.dtypes)

    st.markdown("<h2 style='text-align: center;'>CSV General Information</h2>", unsafe_allow_html=True)
    st.write(data.describe(include='all'))