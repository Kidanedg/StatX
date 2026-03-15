import streamlit as st
import pandas as pd
import requests


# ---------------------------------------------------
# OPEN SCIENCE DATA CONNECTOR
# ---------------------------------------------------

class OpenScienceConnector:

    def search_datasets(self, keyword):

        # Example using Data.gov API style query
        url = f"https://catalog.data.gov/api/3/action/package_search?q={keyword}"

        try:
            r = requests.get(url)
            data = r.json()

            results = []

            for item in data["result"]["results"][:10]:

                results.append({
                    "title":item["title"],
                    "organization":item["organization"]["title"]
                })

            return results

        except:
            return ["Connection error"]


# ---------------------------------------------------
# GENOMIC DATABASE CONNECTOR
# ---------------------------------------------------

class GenomicConnector:

    def search_gene(self, gene):

        # simplified NCBI query simulation
        return {

            "gene":gene,
            "organism":"Homo sapiens",
            "function":"Gene associated with metabolic regulation",
            "related_pathways":["metabolism","cell signaling"]
        }


# ---------------------------------------------------
# CLIMATE DATA CONNECTOR
# ---------------------------------------------------

class ClimateConnector:

    def get_temperature_data(self):

        # simulated climate dataset

        data = {

            "Year":[2018,2019,2020,2021,2022,2023],
            "GlobalTemp":[14.8,14.9,15.1,15.0,15.2,15.3]

        }

        return pd.DataFrame(data)


# ---------------------------------------------------
# ECONOMIC DATA CONNECTOR
# ---------------------------------------------------

class EconomicConnector:

    def worldbank_gdp(self):

        # simplified economic dataset

        data = {

            "Year":[2018,2019,2020,2021,2022],
            "GDP_growth":[3.1,2.9,-3.1,5.9,3.4]

        }

        return pd.DataFrame(data)


# ---------------------------------------------------
# BIOMEDICAL DATA CONNECTOR
# ---------------------------------------------------

class BiomedicalConnector:

    def disease_dataset(self):

        data = {

            "PatientID":[1,2,3,4,5],
            "Age":[45,50,38,62,47],
            "Biomarker":[2.3,3.1,1.8,4.2,2.7],
            "DiseaseStatus":[1,0,0,1,1]

        }

        return pd.DataFrame(data)


# ---------------------------------------------------
# UNIFIED DATA NETWORK
# ---------------------------------------------------

class GlobalDataNetwork:

    def __init__(self):

        self.open_science = OpenScienceConnector()
        self.genomics = GenomicConnector()
        self.climate = ClimateConnector()
        self.economic = EconomicConnector()
        self.biomedical = BiomedicalConnector()


# ---------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------

def run():

    st.title("StatX Global Scientific Data Network")

    network = GlobalDataNetwork()

    option = st.sidebar.selectbox(

        "Select Data Source",

        [

        "Open Scientific Datasets",
        "Genomic Databases",
        "Climate Data",
        "Economic Data",
        "Biomedical Data"

        ]
    )


# ---------------------------------------------------
# OPEN SCIENCE DATASETS
# ---------------------------------------------------

    if option == "Open Scientific Datasets":

        keyword = st.text_input("Search scientific datasets")

        if st.button("Search"):

            results = network.open_science.search_datasets(keyword)

            st.write(results)


# ---------------------------------------------------
# GENOMIC DATA
# ---------------------------------------------------

    if option == "Genomic Databases":

        gene = st.text_input("Enter gene name")

        if st.button("Search Gene"):

            result = network.genomics.search_gene(gene)

            st.write(result)


# ---------------------------------------------------
# CLIMATE DATA
# ---------------------------------------------------

    if option == "Climate Data":

        df = network.climate.get_temperature_data()

        st.dataframe(df)

        st.line_chart(df.set_index("Year"))


# ---------------------------------------------------
# ECONOMIC DATA
# ---------------------------------------------------

    if option == "Economic Data":

        df = network.economic.worldbank_gdp()

        st.dataframe(df)

        st.line_chart(df.set_index("Year"))


# ---------------------------------------------------
# BIOMEDICAL DATA
# ---------------------------------------------------

    if option == "Biomedical Data":

        df = network.biomedical.disease_dataset()

        st.dataframe(df)

        st.bar_chart(df["DiseaseStatus"])


run()
