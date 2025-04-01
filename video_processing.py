# import os
# import json
# import pandas as pd
# import whisper
# from tqdm import tqdm
# from moviepy import VideoFileClip
# from pydub import AudioSegment
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# # Extract audio from video
# def extract_audio(video_path, audio_output_path):
#     video = VideoFileClip(video_path)
#     video.audio.write_audiofile(audio_output_path, codec="mp3")

# # Segment audio into overlapping chunks
# def segment_audio(audio_path, chunk_length=5000, overlap=1000):
#     audio = AudioSegment.from_mp3(audio_path)
#     chunks = []
#     start = 0
#     while start < len(audio):
#         end = min(start + chunk_length, len(audio))
#         chunks.append(audio[start:end])
#         start += chunk_length - overlap
#     return chunks

# # Transcribe audio chunk using Whisper
# def transcribe_audio(chunk, model, temp_audio_path="temp_chunk.mp3"):
#     chunk.export(temp_audio_path, format="mp3")
#     result = model.transcribe(temp_audio_path)
#     return result['text']

# # Analyze sentiment using different models
# def analyze_sentiment(text, sentiment_pipeline):
#     result = sentiment_pipeline(text)[0]
#     label = result["label"]
#     confidence = result["score"]
    
#     sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
#     return sentiment_map.get(label, 0), confidence

# # Allow user to select videos for processing
# def select_videos(video_folder):
#     videos = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
#     if not videos:
#         print("No MP4 files found.")
#         return []

#     print("\nAvailable videos:")
#     for idx, video in enumerate(videos):
#         print(f"{idx + 1}. {video}")

#     choices = input("\nEnter the numbers of videos to process (comma-separated): ")
#     selected_videos = [videos[int(i) - 1] for i in choices.split(",") if i.strip().isdigit()]
    
#     return selected_videos

# # Process videos and save results
# def process_videos_in_folder(video_folder, output_csv):
#     video_files = select_videos(video_folder)
#     if not video_files:
#         return

#     print("\nLoading models...")
#     model = whisper.load_model("base")

#     # Load sentiment models
#     sentiment_model_1 = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
#     tokenizer_1 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
#     pipeline_1 = pipeline("sentiment-analysis", model=sentiment_model_1, tokenizer=tokenizer_1)

#     sentiment_model_2 = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
#     tokenizer_2 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
#     pipeline_2 = pipeline("sentiment-analysis", model=sentiment_model_2, tokenizer=tokenizer_2)

#     data = []

#     for video_filename in tqdm(video_files, desc="Processing videos"):
#         video_path = os.path.join(video_folder, video_filename)
#         audio_output_path = os.path.join(video_folder, f"{os.path.splitext(video_filename)[0]}_audio.mp3")
        
#         print(f"\nExtracting audio from {video_filename}...")
#         extract_audio(video_path, audio_output_path)

#         print(f"Segmenting audio from {video_filename}...")
#         audio_chunks = segment_audio(audio_output_path)

#         for idx, chunk in enumerate(audio_chunks):
#             transcription = transcribe_audio(chunk, model)
#             sentiment_1, confidence_1 = analyze_sentiment(transcription, pipeline_1)
#             sentiment_2, confidence_2 = analyze_sentiment(transcription, pipeline_2)
#             timestamp = idx * 5  # Assuming each chunk is ~5 seconds

#             data.append({
#                 "video_filename": video_filename,
#                 "timestamp": f"{timestamp} sec",
#                 "transcription": transcription,
#                 "sentiment_model_1": sentiment_1,
#                 "confidence_model_1": confidence_1,
#                 "sentiment_model_2": sentiment_2,
#                 "confidence_model_2": confidence_2
#             })

#     df = pd.DataFrame(data)
#     df.to_csv(os.path.join(video_folder, output_csv), index=False)

#     json_output_path = os.path.join(video_folder, "output_sentiment.json")
#     with open(json_output_path, "w") as f:
#         json.dump(data, f, indent=4)

#     print(f"\nProcessing complete! Data saved to {output_csv} and {json_output_path}")

# # Run processing
# if __name__ == "__main__":
#     video_folder = "/home/shubh/gsoc_assesment/videos"
#     output_csv = "output_sentiment.csv"
#     process_videos_in_folder(video_folder, output_csv)



import os
import json
import pandas as pd
import whisper
from tqdm import tqdm
from moviepy import VideoFileClip
from pydub import AudioSegment
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Extract audio from video
def extract_audio(video_path, audio_output_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path, codec="mp3", logger=None)

# Segment audio into overlapping chunks
def segment_audio(audio_path, chunk_length=5000, overlap=1000):
    audio = AudioSegment.from_mp3(audio_path)
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_length, len(audio))
        chunks.append(audio[start:end])
        start += chunk_length - overlap
    return chunks

# Transcribe audio chunk using Whisper
def transcribe_audio(chunk, model, temp_audio_path="temp_chunk.mp3"):
    chunk.export(temp_audio_path, format="mp3")
    result = model.transcribe(temp_audio_path)
    return result['text']

# Analyze sentiment using Sentiment Model 1 (Positive, Neutral, Negative)
def analyze_sentiment_model_1(text, pipeline_1):
    result = pipeline_1(text)[0]
    label = result["label"]
    confidence = result["score"]
    
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    return sentiment_map.get(label, 0), confidence

# Analyze sentiment using Sentiment Model 2 (Fine-grained Emotions)
def analyze_sentiment_model_2(text, classifier):
    emotions = classifier(text)[0]  
    top_2 = sorted(emotions, key=lambda x: x['score'], reverse=True)[:2]  
    return [(e["label"], e["score"]) for e in top_2]

# Process videos
def process_videos(video_folder, output_folder="output"):
    if not os.path.exists(video_folder):
        print(f"Error: Folder '{video_folder}' does not exist.")
        return

    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")]
    
    if not video_files:
        print("No videos found in the folder.")
        return
    
    print(f"Found {len(video_files)} videos. Loading models...")
    model = whisper.load_model("base")

    # Load Model 1 (Basic Sentiment)
    sentiment_model_1 = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    tokenizer_1 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    pipeline_1 = pipeline("sentiment-analysis", model=sentiment_model_1, tokenizer=tokenizer_1)

    # Load Model 2 (Fine-grained Emotion Classification)
    classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    data = []

    os.makedirs(output_folder, exist_ok=True)

    for video_path in tqdm(video_files, desc="Processing videos"):
        audio_output_path = os.path.join(output_folder, os.path.basename(video_path).replace(".mp4", "_audio.mp3"))

        print(f"\nExtracting audio from {os.path.basename(video_path)}...")
        extract_audio(video_path, audio_output_path)

        print(f"Segmenting audio from {os.path.basename(video_path)}...")
        audio_chunks = segment_audio(audio_output_path)

        for idx, chunk in enumerate(audio_chunks):
            transcription = transcribe_audio(chunk, model)
            sentiment_1, confidence_1 = analyze_sentiment_model_1(transcription, pipeline_1)
            top_2_emotions = analyze_sentiment_model_2(transcription, classifier)
            timestamp = idx * 5  

            data.append({
                "video_filename": os.path.basename(video_path),
                "timestamp": f"{timestamp} sec",
                "transcription": transcription,
                "sentiment_model_1": sentiment_1,
                "confidence_model_1": confidence_1,
                "sentiment_model_2_top_1": top_2_emotions[0][0] if top_2_emotions else None,
                "confidence_model_2_top_1": top_2_emotions[0][1] if top_2_emotions else None,
                "sentiment_model_2_top_2": top_2_emotions[1][0] if len(top_2_emotions) > 1 else None,
                "confidence_model_2_top_2": top_2_emotions[1][1] if len(top_2_emotions) > 1 else None,
            })

    df = pd.DataFrame(data)
    output_csv = os.path.join(output_folder, "output_sentiment.csv")
    df.to_csv(output_csv, index=False)

    json_output_path = os.path.join(output_folder, "output_sentiment.json")
    with open(json_output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nProcessing complete! CSV saved at {output_csv} and JSON saved at {json_output_path}")

if __name__ == "__main__":
    video_folder = "/home/shubh/gsoc_assesment/videos"  # Change to the folder containing your videos
    process_videos(video_folder)


