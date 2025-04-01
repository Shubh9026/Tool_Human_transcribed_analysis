# import os
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns

# st.set_page_config(page_title="Sentiment Analysis Visualization", layout="wide")

# # Define the folder where videos and data files are stored
# VIDEO_FOLDER = "/home/shubh/gsoc_assesment/videos"  # Ensure this folder contains the required files
# CSV_PATH = os.path.join(VIDEO_FOLDER, "output_sentiment.csv")
# JSON_PATH = os.path.join(VIDEO_FOLDER, "output_sentiment.json")

# # Load Data Function
# def load_data(file_format):
#     try:
#         if file_format == "CSV":
#             df = pd.read_csv(CSV_PATH)
#         elif file_format == "JSON":
#             df = pd.read_json(JSON_PATH)
#         return df
#     except Exception as e:
#         st.error(f"❌ Error loading {file_format} file: {e}")
#         return None

# # Preprocess Data
# def preprocess_data(df):
#     df["timestamp"] = df["timestamp"].astype(str).str.replace(" sec", "").astype(int)
#     df["word_count"] = df["transcription"].apply(lambda x: len(str(x).split()))
#     return df

# # Sentiment Explanation
# def sentiment_explanation():
#     st.sidebar.subheader("📌 Sentiment Meaning")
#     sentiment_map = {
#         1: "🟢 Positive - Indicates a happy or favorable tone.",
#         0: "⚪ Neutral - No strong positive or negative emotions.",
#         -1: "🔴 Negative - Indicates criticism, sadness, or dissatisfaction.",
#     }
#     for key, value in sentiment_map.items():
#         st.sidebar.write(f"**{key}:** {value}")

# # Plot Word Count
# def plot_word_count(df):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.barplot(x=df["timestamp"], y=df["word_count"], color="blue", ax=ax)
#     ax.set_xlabel("Time (Seconds)")
#     ax.set_ylabel("Word Count")
#     ax.set_title("Word Count per 5-Second Interval")
#     plt.xticks(rotation=45)
#     st.pyplot(fig)

# # Plot Sentiment Analysis Over Time
# def plot_sentiment(df, model_col, title, color):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.lineplot(x=df["timestamp"], y=df[model_col], marker="o", color=color, ax=ax)
#     ax.set_xlabel("Time (Seconds)")
#     ax.set_ylabel("Sentiment Score")
#     ax.set_title(title)
#     plt.xticks(rotation=45)
#     st.pyplot(fig)

# # Sentiment Distribution
# def plot_sentiment_distribution(df, model_col, title, color):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.histplot(df[model_col], bins=10, kde=True, color=color, ax=ax)
#     ax.set_xlabel("Sentiment Score")
#     ax.set_ylabel("Frequency")
#     ax.set_title(title)
#     st.pyplot(fig)

# # Sentiment Pie Chart
# def plot_sentiment_pie(df, model_col, title):
#     sentiment_counts = df[model_col].value_counts()
#     labels = sentiment_counts.index.map({
#         -1: "Negative", 0: "Neutral", 1: "Positive"
#     })
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.pie(sentiment_counts, labels=labels, autopct="%1.1f%%", 
#            colors=["red", "gray", "green"], startangle=90)
#     ax.set_title(title)
#     st.pyplot(fig)

# # UI: Title
# st.title("🎥 Sentiment Analysis Dashboard")

# # Show Sentiment Explanation in Sidebar
# sentiment_explanation()

# # UI: Choose file format
# file_format = st.radio("📂 Select Data Format:", ["CSV", "JSON"])

# # Load Data Automatically After Selection
# df = load_data(file_format)

# if df is not None:
#     df = preprocess_data(df)

#     # UI: Show Data
#     with st.expander("🔍 View Processed Data"):
#         st.dataframe(df)

#     # UI: Visualization Options
#     st.subheader("📊 Data Visualizations")

#     if st.checkbox("✅ Show Word Count per Interval"):
#         plot_word_count(df)

#     if st.checkbox("✅ Show Sentiment Analysis (Model 1) Over Time"):
#         plot_sentiment(df, "sentiment_model_1", "Sentiment Analysis Over Time - Model 1", "blue")

#     if st.checkbox("✅ Show Sentiment Analysis (Model 2) Over Time"):
#         plot_sentiment(df, "sentiment_model_2", "Sentiment Analysis Over Time - Model 2", "red")

#     if st.checkbox("📊 Show Sentiment Distribution - Model 1"):
#         plot_sentiment_distribution(df, "sentiment_model_1", "Sentiment Distribution - Model 1", "blue")

#     if st.checkbox("📊 Show Sentiment Distribution - Model 2"):
#         plot_sentiment_distribution(df, "sentiment_model_2", "Sentiment Distribution - Model 2", "red")

#     if st.checkbox("🥧 Show Sentiment Proportions (Pie Chart) - Model 1"):
#         plot_sentiment_pie(df, "sentiment_model_1", "Sentiment Proportion - Model 1")

#     if st.checkbox("🥧 Show Sentiment Proportions (Pie Chart) - Model 2"):
#         plot_sentiment_pie(df, "sentiment_model_2", "Sentiment Proportion - Model 2")

# else:
#     st.error("⚠ Data file not found. Please check the `video_folder/` for required files.")

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sentiment Analysis Visualization", layout="wide")

# Define the folder where videos and data files are stored
VIDEO_FOLDER = "/home/shubh/gsoc_assesment/output"  # Ensure this folder contains the required files
CSV_PATH = os.path.join(VIDEO_FOLDER, "output_sentiment.csv")
JSON_PATH = os.path.join(VIDEO_FOLDER, "output_sentiment.json")

# Load Data Function
def load_data(file_format):
    try:
        if file_format == "CSV":
            df = pd.read_csv(CSV_PATH)
        elif file_format == "JSON":
            df = pd.read_json(JSON_PATH)
        return df
    except Exception as e:
        st.error(f"❌ Error loading {file_format} file: {e}")
        return None

# Preprocess Data
def preprocess_data(df):
    df["timestamp"] = df["timestamp"].astype(str).str.replace(" sec", "").astype(int)
    df["word_count"] = df["transcription"].apply(lambda x: len(str(x).split()))
    return df

# Sentiment Explanation
def sentiment_explanation():
    st.sidebar.subheader("📌 Sentiment Meaning")
    sentiment_map = {
        1: "🟢 Positive - Indicates a happy or favorable tone.",
        0: "⚪ Neutral - No strong positive or negative emotions.",
        -1: "🔴 Negative - Indicates criticism, sadness, or dissatisfaction.",
    }
    for key, value in sentiment_map.items():
        st.sidebar.write(f"**{key}:** {value}")

# Plot Word Count
def plot_word_count(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=df["timestamp"], y=df["word_count"], color="blue", ax=ax)
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Word Count")
    ax.set_title("Word Count per 5-Second Interval")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Plot Sentiment Analysis Over Time
def plot_sentiment(df, model_col, title, color):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=df["timestamp"], y=df[model_col], marker="o", color=color, ax=ax)
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Sentiment Score")
    ax.set_title(title)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Sentiment Distribution
def plot_sentiment_distribution(df, model_col, title, color):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[model_col], bins=10, kde=True, color=color, ax=ax)
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    st.pyplot(fig)

# Sentiment Pie Chart
def plot_sentiment_pie(df, model_col, title):
    sentiment_counts = df[model_col].value_counts()
    labels = sentiment_counts.index.map({
        -1: "Negative", 0: "Neutral", 1: "Positive"
    })
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sentiment_counts, labels=labels, autopct="%1.1f%%", 
           colors=["red", "gray", "green"], startangle=90)
    ax.set_title(title)
    st.pyplot(fig)

# UI: Title
st.title("🎥 Sentiment Analysis Dashboard")

# Show Sentiment Explanation in Sidebar
sentiment_explanation()

# UI: Choose file format
file_format = st.radio("📂 Select Data Format:", ["CSV", "JSON"])

# Load Data Automatically After Selection
df = load_data(file_format)

if df is not None:
    df = preprocess_data(df)

    # 🎯 **Dropdown to Select a Video**
    available_videos = df["video_filename"].unique().tolist()
    selected_video = st.selectbox("🎬 Select a Video for Analysis:", available_videos)

    # 🎯 **Filter Data for Selected Video**
    df = df[df["video_filename"] == selected_video]

    # UI: Show Data
    with st.expander("🔍 View Processed Data"):
        st.dataframe(df)

    # UI: Visualization Options
    st.subheader("📊 Data Visualizations")

    if st.checkbox("✅ Show Word Count per Interval"):
        plot_word_count(df)

    if st.checkbox("✅ Show Sentiment Analysis (Model 1) Over Time"):
        plot_sentiment(df, "sentiment_model_1", "Sentiment Analysis Over Time - Model 1", "blue")

    if st.checkbox("✅ Show Sentiment Analysis (Model 2) Over Time"):
        plot_sentiment(df, "sentiment_model_2_top_1", "Sentiment Analysis Over Time - Model 2", "red")

    if st.checkbox("📊 Show Sentiment Distribution - Model 1"):
        plot_sentiment_distribution(df, "sentiment_model_1", "Sentiment Distribution - Model 1", "blue")

    if st.checkbox("📊 Show Sentiment Distribution - Model 2"):
        plot_sentiment_distribution(df, "sentiment_model_2_top_1", "Sentiment Distribution - Model 2", "red")

    if st.checkbox("🥧 Show Sentiment Proportions (Pie Chart) - Model 1"):
        plot_sentiment_pie(df, "sentiment_model_1", "Sentiment Proportion - Model 1")

    if st.checkbox("🥧 Show Sentiment Proportions (Pie Chart) - Model 2"):
        plot_sentiment_pie(df, "sentiment_model_2_top_1", "Sentiment Proportion - Model 2")

else:
    st.error("⚠ Data file not found. Please check the `video_folder/` for required files.")

