#!/usr/bin/env python3
"""
Audio Transcription Script
This script performs speaker diarization and transcription on audio files using
PyAnnote Audio and Whisper models.
"""

# !pip -q install torch==2.2.0
# !pip -q install typer==0.9.1

# !pip -q install pyannote.audio # diarization
# !pip -q install pydub # segmentation
# !pip -q install transformers # transcription

import torch
import torchaudio
import base64
from io import BytesIO
import shutil
import os
from typing import List, Dict
import argparse

from huggingface_hub import notebook_login, HfApi
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pydub import AudioSegment
from transformers import pipeline

def setup_environment():
    """
    Set up the environment by removing old aiohttp installations and installing required packages.
    """
    # Define paths for aiohttp files
    aiohttp_paths = [
        '/opt/conda/lib/python3.10/site-packages/aiohttp*',
        '/opt/conda/lib/python3.10/site-packages/aiohttp-3.9.1.dist-info'
    ]
    
    # Remove aiohttp package directories and files
    for path in aiohttp_paths:
        shutil.rmtree(path, ignore_errors=True)

def initialize_models(use_cuda: bool = True) -> tuple:
    """
    Initialize the diarization and transcription models.
    
    Args:
        use_cuda (bool): Whether to use CUDA for GPU acceleration
        
    Returns:
        tuple: Initialized diarization pipeline and transcriber
    """
    # Initialize diarization pipeline
    pipeline_diar = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=True
    )
    
    if use_cuda and torch.cuda.is_available():
        pipeline_diar.to(torch.device("cuda"))
    
    # Initialize transcriber
    transcriber = pipeline(
        model="openai/whisper-large-v2",
        return_timestamps=False
    )
    
    return pipeline_diar, transcriber

def read_audio_file(file_path: str) -> str:
    """
    Read an audio file and convert it to base64 encoded string.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        str: Base64 encoded audio data
    """
    with open(file_path, "rb") as file:
        mp3_data = file.read()
    return base64.b64encode(mp3_data).decode('utf-8')

def transcribe_recording(
    base64_encoded_audio: str,
    pipeline_diar: Pipeline,
    transcriber: pipeline
) -> List[Dict]:
    """
    Transcribe audio data with speaker diarization.
    
    Args:
        base64_encoded_audio (str): Base64 encoded audio data
        pipeline_diar (Pipeline): Initialized diarization pipeline
        transcriber (pipeline): Initialized transcription pipeline
        
    Returns:
        List[Dict]: List of transcription segments with speaker information
    """
    # Decode the base64 encoded string
    base64_decoded_audio = base64.b64decode(base64_encoded_audio)
    
    with BytesIO(base64_decoded_audio) as audio_buffer:
        audio_buffer.seek(0)
        
        # Load audio with pydub
        audio = AudioSegment.from_file(audio_buffer)
        temp_file = "temp_audio.mp3"
        audio.export(temp_file, format="mp3")

        try:
            # Run diarization pipeline
            dia = pipeline_diar(temp_file)
            assert isinstance(dia, Annotation)
            
            # Store transcription data
            data = []

            for i, (speech_turn, track, speaker) in enumerate(dia.itertracks(yield_label=True)):
                # Extract timestamps
                start_time, end_time = speech_turn.start, speech_turn.end

                # Process audio segment
                segment_audio = audio[int(start_time * 1000):int(end_time * 1000)]
                segment_audio.export(temp_file, format="mp3")
                
                # Transcribe segment
                text = transcriber(temp_file)['text']

                # Store segment data
                data.append({
                    "Start Time": start_time,
                    "End Time": end_time,
                    "Speaker": speaker,
                    "Transcription": text
                })

            return data
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Audio transcription with speaker diarization")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--output", help="Output file path for transcription", default="transcription.txt")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA acceleration")
    args = parser.parse_args()

    # Setup environment and login to Hugging Face
    setup_environment()
    notebook_login()

    # Initialize models
    pipeline_diar, transcriber = initialize_models(not args.no_cuda)

    # Read and process audio file
    print(f"Processing audio file: {args.audio_file}")
    encoded_audio = read_audio_file(args.audio_file)
    
    # Perform transcription
    results = transcribe_recording(encoded_audio, pipeline_diar, transcriber)

    # Write results to file
    with open(args.output, 'w', encoding='utf-8') as f:
        for segment in results:
            f.write(f"[{segment['Start Time']:.2f} - {segment['End Time']:.2f}] "
                   f"{segment['Speaker']}: {segment['Transcription']}\n")
    
    print(f"Transcription completed. Results saved to: {args.output}")

if __name__ == "__main__":
    main()