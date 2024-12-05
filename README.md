# LLM Tools and Scripts

A collection of Python scripts for working with Large Language Models (LLMs), including tools for text summarization, audio transcription with speaker diarization, and model fine-tuning.

## 🤔 What are LLMs?
Large Language Models (LLMs) are advanced AI systems trained on vast amounts of text data to understand and generate human-like text. These models can perform a wide range of tasks including:

Text generation and completion
Question answering
Summarization
Translation
Code generation
And much more

Popular examples include GPT-4, Claude, and Llama 2. These models have revolutionized natural language processing by demonstrating remarkable abilities to understand context and generate coherent, relevant responses.

## 🔄 Transformers: The Engine Behind LLMs

The Transformers architecture, introduced in the paper "Attention Is All You Need," is the fundamental architecture powering modern LLMs. Key features include:

Self-attention mechanism: Allows the model to weigh the importance of different words in context
Parallel processing: Enables efficient training on massive datasets
Bidirectional context: Considers both left and right context for better understanding
Transfer learning: Pre-trained models can be fine-tuned for specific tasks

The Hugging Face Transformers library provides easy access to state-of-the-art transformer models and tools for working with them.

## ⛓️ LangChain: Building LLM Applications

LangChain is a framework for developing applications powered by language models. It provides:

Chains: Combine LLMs with other processing steps
Agents: Create autonomous AI systems that can use tools
Memory: Manage conversation history and context
Prompt management: Template and optimize model inputs
Document processing: Handle various data formats and sources

LangChain makes it easier to build complex LLM applications by providing high-level abstractions and best practices.

## 💡 QLoRA: Efficient Fine-tuning

QLoRA (Quantized Low-Rank Adaptation) is a technique for efficiently fine-tuning large language models. Benefits include:

Reduced memory usage: Train large models on consumer GPUs
Preserved model quality: Maintain performance while reducing parameters
Fast adaptation: Quickly customize models for specific tasks
Cost-effective: Lower computational requirements for training

QLoRA makes it practical to adapt state-of-the-art models to specific use cases without extensive computational resources.

## 🚀 Features

- **Deposition Summarizer**: Automatically generates concise summaries of text using the Mixtral model, extracting key information topics discussed, and key statements.

- **Audio Transcription**: Performs speaker diarization and transcription using PyAnnote Audio and Whisper models, generating timestamped transcripts with speaker identification.

- **Model Fine-tuning**: Tools for fine-tuning language models on custom datasets.

## 📦 Requirements

- Python 3.8+
- PyTorch
- Transformers
  - Provides core model architectures and pretrained weights
  - Handles tokenization and inference
- Langchain
  - Manages prompt engineering and chain composition
  - Provides document loading and processing utilities
- PyAnnote Audio
- OpenAI Whisper
- Additional dependencies listed in `requirements.txt`

## 🎯 Usage

### Text Summarizer
Initialize the summarizer with your preferred model and generate summaries of text.

### Audio Transcription
Process audio files to generate speaker-separated transcripts with timestamps.

🙏 Acknowledgments

Hugging Face for transformer models
PyAnnote Audio for speaker diarization
OpenAI for the Whisper model
