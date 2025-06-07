
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker

# Correct DATA_PATH relative to current script location
DATA_PATH = Path(__file__).parent / "data" / "population.csv"
class PopulationAnalyzer:
    def __init__(self, data_path):
        # Read the CSV with updated structure
        self.df = pd.read_csv(data_path, dtype={
            'Country Name': str,
            'Country Code': str,
            'Year': np.int32,
            'Value': np.float32
        })

    def get_population_data(self, country_name, start_year, end_year):
        # Filter the dataframe for the specified country and years
        country_data = self.df[(self.df['Country Name'] == country_name) &
                               (self.df['Year'] >= start_year) &
                               (self.df['Year'] <= end_year)]
        return country_data


class PopulationDashboard:
    def __init__(self, analyzer: PopulationAnalyzer):
        self.analyzer = analyzer

    def display_dashboard(self, country_name, start_year, end_year, comparison_countries):
        tab1, tab2 = st.tabs(["Population change", "Compare"])

        with tab1:
            self._display_population_change(country_name, start_year, end_year)

        with tab2:
            self._display_comparison(country_name, start_year, end_year, comparison_countries)

    def _display_population_change(self, country_name, start_year, end_year):
        st.subheader(f"Population change for {country_name} from {start_year} to {end_year}")
        
        # Get population data for the selected range
        country_data = self.analyzer.get_population_data(country_name, start_year, end_year)
        initial = country_data.iloc[0]['Value']
        final = country_data.iloc[-1]['Value']

        percentage_diff = round((final - initial) / initial * 100, 2)
        delta = f"{percentage_diff}%"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{start_year}", value=initial)
            st.metric(f"{end_year}", value=final, delta=delta)

        with col2:
            self._plot_population_trends(country_data)

    def _display_comparison(self, country_name, start_year, end_year, comparison_countries):
        st.subheader('Compare with other countries')

        # Plot for all selected countries
        fig, ax = plt.subplots()
        for each in comparison_countries:
            data = self.analyzer.get_population_data(each, start_year, end_year)
            ax.plot(data['Year'], data['Value'], label=each)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Population')

        # Remove scientific notation on the y-axis
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

        ax.set_xticks([start_year, end_year])
        ax.legend()
        st.pyplot(fig)

    def _plot_population_trends(self, country_data):
        fig, ax = plt.subplots()
        ax.plot(country_data['Year'], country_data['Value'])
        ax.set_xlabel('Year')
        ax.set_ylabel('Population')

        # Remove scientific notation on the y-axis
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

        ax.set_xticks([country_data['Year'].iloc[0], country_data['Year'].iloc[-1]])
        fig.autofmt_xdate()
        st.pyplot(fig)



def main():
    # Initialize the analyzer and dashboard
    analyzer = PopulationAnalyzer(DATA_PATH)
    dashboard = PopulationDashboard(analyzer)

    # Streamlit form for input selection
    st.title("Population Data")
    st.markdown("Source table can be found [here](https://www.worldbank.org/en/home).")

    with st.form("population-form"):
        col1, col2, col3 = st.columns(3)

        # Set default values in session_state if not already set
        if "start_year" not in st.session_state:
            st.session_state.start_year = 1960
        if "end_year" not in st.session_state:
            st.session_state.end_year = 2023
        if "country_name" not in st.session_state:
            st.session_state.country_name = analyzer.df['Country Name'].unique()[0]

        with col1:
            st.write("Choose a starting year")
            st.session_state.start_year = st.slider("Start Year", min_value=1960, max_value=2023, value=st.session_state.start_year, step=1, key="start_y")

        with col2:
            st.write("Choose an end year")
            st.session_state.end_year = st.slider("End Year", min_value=1960, max_value=2023, value=st.session_state.end_year, step=1, key="end_y")
        
        with col3:
            st.write("Choose a country")
            st.session_state.country_name = st.selectbox("Choose a country", options=analyzer.df['Country Name'].unique(), index=analyzer.df['Country Name'].unique().tolist().index(st.session_state.country_name))

        # Adding the multiselect for country comparison in the form
        st.session_state.comparison_countries = st.multiselect("Compare with other countries", options=analyzer.df['Country Name'].unique(), default=[st.session_state.country_name])

        submit_btn = st.form_submit_button("Analyze", type="primary")

    # Display the dashboard if valid
    if submit_btn:
        dashboard.display_dashboard(st.session_state.country_name, st.session_state.start_year, st.session_state.end_year, st.session_state.comparison_countries)


if __name__ == "__main__":
    main()
