from flask import Flask, jsonify, send_from_directory, request
import os
from instagrapi import Client
from moviepy.editor import VideoFileClip
from transformers import pipeline
import whisper
import cv2
import numpy as np
from scenedetect import detect, ContentDetector
from PIL import Image

app = Flask(__name__)

# Directory to store videos, transcriptions, summaries, and visual summaries
DATA_DIR = 'data'

# Instagram API client
instagram_client = Client()

# Whisper model for speech recognition
whisper_model = whisper.load_model("base")

# Summarization model
summarizer = pipeline("summarization")

@app.route('/subscribe/<account_name>', methods=['GET', 'POST'])
def subscribe(account_name):
    username = os.environ.get('APP_USERNAME')
    password = os.environ.get('APP_PASSWORD')
    print('all,user/pass',username,password)
    if not all([account_name, username, password]):
        return jsonify({'error': 'Missing required fields','account_name':account_name, 'username':username, 'password':password}), 400

    try:
        # Login to Instagram
        instagram_client.login(username, password)

        # Get user ID for the account to subscribe
        user_id = instagram_client.user_id_from_username(account_name)

        # Create directories for the account
        account_dir = os.path.join(DATA_DIR, account_name)
        os.makedirs(os.path.join(account_dir, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(account_dir, 'transcriptions'), exist_ok=True)
        os.makedirs(os.path.join(account_dir, 'summaries'), exist_ok=True)

        # Download recent posts and reels
        medias = instagram_client.user_medias(user_id, 20)
        for media in medias:
            if media.media_type in [2, 2]:  # Photo or Video
                file_path = os.path.join(account_dir, 'videos', f"{media.pk}.mp4")
                instagram_client.video_download(media.pk, file_path)
                
                # Transcribe and summarize video
                transcribe_and_summarize(account_name, f"{media.pk}.mp4")

        return jsonify({'status': f'Subscribed to {account_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def transcribe_and_summarize(account_name, video_name):
    video_path = os.path.join(DATA_DIR, account_name, 'videos', video_name)
    audio_path = os.path.join(DATA_DIR, account_name, 'videos', f"{video_name}.wav")

    # Extract audio from video
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

    # Transcribe audio using Whisper
    result = whisper_model.transcribe(audio_path)
    transcription = result["text"]

    # Save transcription
    transcription_path = os.path.join(DATA_DIR, account_name, 'transcriptions', f"{video_name}.txt")
    with open(transcription_path, 'w') as f:
        f.write(transcription)

    # Summarize transcription
    summary = summarizer(transcription, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

    # Save summary
    summary_path = os.path.join(DATA_DIR, account_name, 'summaries', f"{video_name}.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)

    # Generate visual summary
    generate_visual_summary(account_name, video_name)

    # Clean up temporary audio file
    os.remove(audio_path)

def generate_visual_summary(account_name, video_name):
    video_path = os.path.join(DATA_DIR, account_name, 'videos', video_name)
    
    # Detect scenes
    scene_list = detect(video_path, ContentDetector())
    
    # Extract keyframes
    keyframes = []
    cap = cv2.VideoCapture(video_path)
    for scene in scene_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, scene[0].frame_num)
        ret, frame = cap.read()
        if ret:
            keyframes.append(frame)
    cap.release()
    
    # Create visual summary (storyboard)
    num_frames = min(len(keyframes), 9)  # Limit to 9 frames for a 3x3 grid
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    frame_size = (200, 200)
    grid_image = Image.new('RGB', (frame_size[0] * grid_size, frame_size[1] * grid_size))
    
    for i, frame in enumerate(keyframes[:num_frames]):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frame = Image.fromarray(frame)
        grid_image.paste(frame, ((i % grid_size) * frame_size[0], (i // grid_size) * frame_size[1]))
    
    # Save visual summary
    visual_summary_path = os.path.join(DATA_DIR, account_name, 'visual_summaries', f"{video_name}.jpg")
    grid_image.save(visual_summary_path)

# ... (keep the existing routes for listing and serving files)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)

