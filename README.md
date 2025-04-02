# Communication Analysis Tool for Human-AI Interaction Driving Simulator Experiments  

## 📌 Overview  
This project analyzes conversations from **driving simulation experiments**, extracting **speech-to-text transcriptions** and performing **sentiment analysis** to evaluate human-AI interactions. The tool processes **video recordings**, extracts audio, segments speech, and applies **NLP models** to detect sentiment and emotions.  

## 🚀 Features  
✅ Extracts **audio from driving simulation videos**  
✅ Uses **Whisper** for **speech-to-text transcription**  
✅ Performs **sentiment analysis** (Positive, Neutral, Negative)  
✅ Identifies **fine-grained emotions** (e.g., Joy, Fear, Anger)  
✅ Outputs results in **CSV** and **JSON** format  
✅ Provides **visual analysis** using Streamlit  

## 📂 Input & Output  
### 🔹 Input  
- A **folder** containing one or more **video files (.mp4)**  
- The folder path must be **specified in the script**  

### 🔹 Output  
- **CSV File (`output_sentiment.csv`)** – Processed sentiment data  
- **JSON File (`output_sentiment.json`)** – Structured sentiment data  

## 🛠 Installation & Setup  
### 1️⃣ Create & Activate a Virtual Environment  
It is **highly recommended** to run this project inside a virtual environment.  

```bash
# Create a virtual environment (Python 3.8+ recommended)
python -m venv env  

# Activate it  
# On Windows  
env\Scripts\activate  

# On macOS/Linux  
source env/bin/activate  

## Install Dependecies.
pip install -r requirements.txt

##Run the Tool
python video_processing.py

##UI Interaction in Streamlit
streamlit run app.py  


