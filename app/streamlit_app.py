import json
import os
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Ames Housing Valuation",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data(ttl=30)
def get_model_schema():
    response = requests.get(
        f"{API_URL}/schema",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def check_api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("status") == "healthy"
    except requests.RequestException:
        return False


def predict_batch(dataframe):
    # Convert missing values to JSON null values
    records = json.loads(dataframe.to_json(orient="records"))

    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"records": records},
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("Ames Housing")
st.sidebar.caption("Property Valuation Dashboard")

if check_api_health():
    st.sidebar.success("Prediction API connected")
else:
    st.sidebar.error("Prediction API unavailable")
    st.sidebar.info("Start the FastAPI server at http://127.0.0.1:8000")
    st.stop()


# -------------------------
# Main header
# -------------------------

st.title("Property Valuation Dashboard")

st.markdown(
    """
    Upload a dataset containing property characteristics,
    generate estimated sale prices, and analyze the results.
    """
)

st.divider()


# -------------------------
# Upload dataset
# -------------------------

st.subheader("1. Upload Property Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="The CSV must contain all features required by the model.",
)

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()


try:
    dataframe = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read CSV file: {error}")
    st.stop()


if dataframe.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()


try:
    schema = get_model_schema()
    required_features = schema["required_features"]
except requests.RequestException as error:
    st.error(f"Could not retrieve model schema: {error}")
    st.stop()


missing_features = [
    column for column in required_features if column not in dataframe.columns
]

extra_features = [
    column for column in dataframe.columns if column not in required_features
]


if missing_features:
    st.error(
        f"Your dataset is missing {len(missing_features)} required model features."
    )

    with st.expander("View missing features"):
        st.write(missing_features)

    st.stop()


if extra_features:
    st.warning(f"{len(extra_features)} additional columns will be ignored.")


# Keep only model features, in the correct order
model_input = dataframe[required_features].copy()


st.success(
    f"Dataset validated: {len(dataframe):,} properties "
    f"and {len(required_features)} required features."
)


# -------------------------
# Dataset preview
# -------------------------

st.subheader("2. Dataset Preview")

st.caption(f"Showing the first 10 of {len(dataframe):,} uploaded records.")

st.dataframe(
    dataframe.head(10),
    use_container_width=True,
)


# -------------------------
# Generate predictions
# -------------------------

st.subheader("3. Generate Property Valuations")

if st.button(
    "Generate Predictions",
    type="primary",
    use_container_width=True,
):
    with st.spinner("Generating property valuations..."):
        try:
            result = predict_batch(model_input)

            results_df = dataframe.copy()
            results_df["Predicted Sale Price"] = result["predictions"]

            st.session_state["prediction_results"] = results_df

            st.success(
                f"Successfully generated {result['total_properties']:,} "
                "property valuations."
            )

        except requests.RequestException as error:
            st.error(f"Prediction request failed: {error}")

            if error.response is not None:
                st.error(error.response.text)

        except Exception as error:
            st.error(f"Unexpected error: {error}")


# -------------------------
# Results and analytics
# -------------------------

if "prediction_results" in st.session_state:
    results_df = st.session_state["prediction_results"]

    predictions = results_df["Predicted Sale Price"]

    st.divider()
    st.subheader("4. Valuation Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Properties Valued",
        f"{len(results_df):,}",
    )

    col2.metric(
        "Average Estimated Price",
        f"${predictions.mean():,.0f}",
    )

    col3.metric(
        "Median Estimated Price",
        f"${predictions.median():,.0f}",
    )

    col4.metric(
        "Total Estimated Value",
        f"${predictions.sum():,.0f}",
    )

    # Predicted price distribution
    st.subheader("Predicted Price Distribution")

    fig, ax = plt.subplots(figsize=(7, 3))

    sns.histplot(
        predictions,
        bins=30,
        kde=True,
        ax=ax,
    )

    ax.set_title("Distribution of Predicted Property Prices")
    ax.set_xlabel("Predicted Sale Price ($)")
    ax.set_ylabel("Number of Properties")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Price spread and outliers
    st.subheader("Price Distribution Summary")

    fig, ax = plt.subplots(figsize=(7, 2.5))

    sns.boxplot(
        x=predictions,
        ax=ax,
    )

    ax.set_xlabel("Predicted Sale Price ($)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # st.subheader("Predicted Price Distribution")

    # price_distribution = pd.cut(predictions, bins=10).value_counts().sort_index()

    # price_distribution.index = price_distribution.index.astype(str)

    # price_distribution_df = price_distribution.rename(
    #     "Number of Properties"
    # ).reset_index()

    # price_distribution_df.columns = [
    #     "Predicted Price Range",
    #     "Number of Properties",
    # ]

    # st.bar_chart(
    #     price_distribution_df,
    #     x="Predicted Price Range",
    #     y="Number of Properties",
    # )

    st.subheader("Property-Level Predictions")

    st.dataframe(
        results_df,
        use_container_width=True,
    )

    csv_data = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Valuation Results",
        data=csv_data,
        file_name="ames_housing_valuation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
