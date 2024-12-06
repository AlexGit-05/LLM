# 🧪 **Model Testing Using Prompt Engineering (LLM)**

## 📜 **Overview**

This repository contains a Python notebook designed to evaluate the performance of various **Large Language Models (LLMs)** in **text summarization tasks**. It uses **prompt engineering** to create consistent and structured prompts for testing each model. The notebook facilitates the comparison of multiple models by analyzing their summarization capabilities, execution times, and performance across a variety of carefully crafted prompts.

---

## 🌐 **Introduction to Model Testing Using Prompt Engineering**

**Model testing** is a critical process in machine learning and natural language processing (NLP) to evaluate the effectiveness and efficiency of different models. With **prompt engineering**, we craft specific inputs to test how well a model performs on various tasks. By testing models across identical, carefully designed prompts, we can:
- Identify strengths and weaknesses of each model.
- Determine their suitability for specific applications.
- Optimize prompt structures and configurations to improve model performance.

This repository focuses on testing summarization models by using **prompt engineering** to create consistent tasks, highlighting their ability to condense text while retaining essential information. Prompt engineering allows for more controlled testing by standardizing the instructions given to each model, providing a clearer basis for comparison.

---

## 🛠 **Features**

- **Model Variety**: Tests multiple LLMs, including:
  - **Mistral Models**: Mistral-8x7B-Instruct-v0.1, Mistral-7B-Instruct-v0.2.
  - **Falcon Model**: falcon-7b-instruct.
  - **BART Model**: bart-large-cnn.
  - **Gemini Model**: From Google Generative AI.
- **Performance Metrics**:
  - Execution time for generating summaries.
  - Summarization quality comparison.
  - Response consistency across different prompt variations.
- **Customizable Input**: Easily adapt the input text and summarization prompts for testing. Users can modify prompts to test different summarization strategies or styles.
- **Prompt Engineering Framework**: Uses carefully constructed prompts to test models under consistent conditions, ensuring fair comparisons and reliable results.

---

## 💻 **How It Works**

1. **Setup**:
   - Installs necessary packages and imports required libraries.
   - Configures models using Hugging Face Hub and Google Generative AI tools.
   - Defines a **prompt engineering framework** to construct and test different summarization prompts consistently across models.

2. **Summarization Task**:
   - Sets up a **text prompt template** tailored for summarization instructions, which can be easily modified to test various styles (e.g., concise summaries, detailed summaries, etc.).
   - Passes the input text through each model using the engineered prompts to generate summaries, ensuring each model receives identical instructions.

3. **Evaluation**:
   - Records **execution time** for each model to assess efficiency.
   - Compares **output summaries** for qualitative and quantitative analysis based on criteria such as informativeness, coherence, and fidelity to the original text.
   - Tests the model’s robustness and adaptability by adjusting prompt structures (e.g., changing verbosity, altering wording) and comparing results.

---

## ⚙️ **Technologies & Libraries Used**

- **Python**:
  - `transformers`: For model loading and inference.
  - `torch`: For GPU-accelerated computations.
  - `time`: To measure execution times.
  - `pandas`: For structured result analysis.
- **Hugging Face Hub**:
  - Provides pre-trained models for Mistral, Falcon, and BART.
- **Google Generative AI**:
  - Access to the Gemini model for comparison.
- **Prompt Engineering**:
  - Custom templates and structures for consistent testing of summarization capabilities across different models, ensuring fair comparison and performance evaluation.

---

## 📊 **Key Testing and Evaluation Aspects**
1. **Prompt Variability**:
   - Test how slight changes in prompt structure impact model outputs, such as adding constraints or asking for summaries in different styles (e.g., bullet points vs. narrative summaries).
  
2. **Performance Analysis**:
   - Compare execution times for different models and prompt types.
   - Evaluate how well each model adapts to complex prompts or edge case scenarios.

3. **Output Quality**:
   - Assess the models’ abilities to condense information while retaining key facts and coherence.
   - Evaluate summary length, relevance, and whether the output captures the essence of the input text.

---

This repository is a robust framework for evaluating the performance of different summarization models and can be extended to test other NLP tasks using **prompt engineering**. It helps identify the most suitable models for specific applications and refines prompt designs for optimal model performance.
