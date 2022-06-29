from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 14.0

import plotly.graph_objects as go
import plotly.express as px


data = None

@st.cache
def filter_by_activity(dataframe, activity='repouso'):
    return dataframe[dataframe['Activity'] == activity]


@st.cache
def get_activity_entry_data(dataframe, entry=0):
    FREQ = 100
    TIME_INTERVAL = 10
    start_ind = entry*FREQ*TIME_INTERVAL
    return dataframe[start_ind:start_ind+(FREQ*TIME_INTERVAL)]

@st.cache(suppress_st_warning=True)
def read_csv_data(csv_file):
    return pd.read_csv(csv_file)




def show_unprocessed_data_view(data_dataframe):

    # Show the whole dataset
    st.header("🔢 Un-processed Dataframe")
    col1, col2 = st.columns(2)
    with col1:
        st.text(f"Total of {data_dataframe.shape[0]} entries")
        st.dataframe(data_dataframe)
    with col2:
        # Also show some info of the whole dataset
        st.subheader("Detailed stats.")
        st.dataframe(data_dataframe.drop(
            'ind', axis=1, inplace=False).describe())

    # Check how many data entries there are for each activity
    st.header("📊 Data Distribution.")

    col1, col2 = st.columns([2,1])
    with col1:
        entries_per_activity = (
            data_dataframe['Activity'].value_counts() / 1000).astype(int)
        st.bar_chart(entries_per_activity, use_container_width=True)
    with col2:
        # Pie plot
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(entries_per_activity, labels=entries_per_activity.index.tolist(
        ), autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚶‍♂️ andar", entries_per_activity['andar'])
    col2.metric("🚴 bicicleta", entries_per_activity['bicicleta'])
    col3.metric("🏃‍♀️ correr", entries_per_activity['correr'])
    col4.metric("💤 repouso", entries_per_activity['repouso'])

    samples = data_dataframe.sample(n=2000)
    fig = px.scatter_3d(samples, x='Accel. X', y='Accel. Z', z='Accel. Y', color='Activity', color_continuous_scale=px.colors.sequential.Viridis, opacity=0.7)
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=10))
    st.plotly_chart(fig, use_container_width=True)


    ################ Visualize data by activity ###################
    st.header("📈 Activity Graphs")
    st.caption("Use the sidebar options to navigate data instances/frames.")
    # Filter by 'Activity'
    st.sidebar.header("⚙️ Activity Graph Options")
    selected_activity = st.sidebar.selectbox(
        'Select an activity',
        ('repouso', 'andar', 'correr', 'bicicleta'), index=1 )
    data_filtered_by_activity = filter_by_activity(
        data_dataframe, activity=selected_activity)

    # Now that we have data relative only to a specific activity, we can show a slider that
    # alows for the selection of all the available entries for that activity
    available_entries = int(entries_per_activity[selected_activity])
    entry_index = st.sidebar.slider(
        'Select an entry', min_value=0, max_value=available_entries-1, value=5)
    entry_data = get_activity_entry_data(
        data_filtered_by_activity, entry=entry_index)

    axis_selected = st.sidebar.multiselect(
        'Sensor axis to visualize',
        ['Accel. X', 'Accel. Y', 'Accel. Z', 'Gyro. X', 'Gyro. Y', 'Gyro. Z'],
        default=['Accel. X', 'Accel. Y', 'Accel. Z']
    )

    smoothing_level = st.sidebar.slider(
        'Smoothing level', min_value=1, max_value=5, value=4)

    mpl.rcParams['font.size'] = 8.0
    for axis in axis_selected:
        st.line_chart(entry_data[axis].ewm(span=smoothing_level).mean())

    fig = px.scatter_3d(entry_data, x='Accel. X', y='Accel. Z', z='Accel. Y', color='Accel. X', color_continuous_scale=px.colors.sequential.Viridis, opacity=0.7)
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=10))
    st.plotly_chart(fig, use_container_width=True)


   


if __name__ == '__main__':

    st.set_page_config(layout='wide', page_title="Data Viz")

    st.title("Data Visualization")

    csv_file = st.file_uploader("Upload a .csv file.")
    
    
    if csv_file is None:
        st.info("No .csv data file has been uploaded / there is no valid data to visualize.")
    else:
        data = read_csv_data(csv_file)
        show_unprocessed_data_view(data)
            