# 🎵 Personalized Music Playlist Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![RAG](https://img.shields.io/badge/Architecture-RAG-green)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Optional EDA](#optional-eda--data-analysis)
- [Project Presentation](#project-presentation)
- [Evaluation & Reproducibility](#evaluation--reproducibility)

- [Contributors](#contributors)


---

## Overview

**Personalized Music Playlist Generation (RAG)** is a system that generates highly personalized music playlists using a hybrid of **semantic retrieval** and **Large Language Model (LLM)** generation.

Unlike traditional playlist generators that rely solely on collaborative filtering or fixed heuristics, this system interprets a user’s intent, context, and musical preferences — yielding playlists that feel tailor-made.

In short: tell it how you feel (e.g., *"Chill late-night driving vibes"*), and it will return a curated playlist — plus an explanation for each song, demonstrating **why** it matches your taste.

---

## Features

* **🧠 RAG Architecture:** Semantic vector retrieval of tracks combined with LLM‑driven playlist generation.
* **👤 Dynamic User Profiling:** The system saves user preferences (favorite songs/artists/genres) for evolving recommendations over time.
* **💡 Explainable Recommendations:** For each suggested track, you receive a short justification explaining why it fits your prompt and profile.
* **🎤 Voice Interaction:** Use the microphone to speak your request instead of typing.
* **🔄 Integrated Data Pipeline:** Choose to run the full data preprocessing (EDA) or skip directly to the chat interface.

---

## Project Structure

```text
(project root)
│── data_preprocessing.py     # Preprocess raw music dataset (cleaning, feature extraction, embedding)
│── rag.py                    # Core logic: embeddings, vector retrieval, LLM input/output handling
│── openai_client.py          # Interface with LLM / OpenAI API
│── score_by_song.py          # Script for scoring songs individually
│── states.py                 # Configuration: constants, user profile structure, caching
│── helpers.py                # Utility functions (data loading, parsing)
│── gradio_helpers.py         # Helpers for Web-App UI formatting and output display
│── record_queries.py         # Speech-to-text transcription logic
│── main.py                   # ORCHESTRATOR: Main entry point (EDA Prompt + UI Launch)
│── requirements.txt          # Python dependencies
│── .env.example              # Template for environment variables
└── README.md                 # This documentation
```
## Datasets

This project leverages the following publicly available datasets:

1. **Spotify 900K Dataset**  
   **Title:** *500K+ Spotify Songs with Lyrics, Emotions & More*  
   **Source:** [Kaggle - devdope/900k-spotify](https://www.kaggle.com/datasets/devdope/900k-spotify)  
   **Description:** A large-scale dataset of over 900,000 songs containing metadata, lyrics, mood/emotion labels, and various audio features.

2. **Billboard Hot 100 Charts Dataset**  
   **Title:** *Billboard Hot 100 & More*  
   **Source:** [Kaggle - Billboard Charts](https://www.kaggle.com/datasets/dhruvildave/billboard-the-hot-100-songs)  
   **Description:** Contains historical data on Billboard's Hot 100 charts, useful for analyzing song popularity, trends, and rankings over time.

## Getting Started

### Prerequisites
- Python 3.8+
- A valid API key for your LLM provider (e.g., OpenAI)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yuvalmar16/Personalized-Music-Playlist-Generation.git
cd Personalized-Music-Playlist-Generation
2. Create & Activate Virtual Environment
bash
Copy code
# Mac / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
Note:
On some Windows machines, antivirus software (e.g., Windows Defender) may block parts of Gradio’s installation or execution.
If you encounter issues when launching Gradio, ensure your antivirus is not preventing the installation.

4. Insert Your OpenAI API Key
Open the file:

Copy code
openai_client.py
In line 8, replace:

python
Copy code
openai_api_key = "INSERT YOUR API KEY HERE"
with your actual API key.

5. Running the Code – HuggingFace Key + Data Manipulation Options
When running the system for the first time:

The program will ask you to enter your Hugging Face API key.

After that, you will be prompted to choose whether to run the data manipulation pipeline:

Enter skip → skips all preprocessing and uses the pre-processed dataset.

Enter anything else → runs the full preprocessing & manipulation steps.

6. Optional: Enable Summaries in data_preprossesing.py
In the file:

Copy code
data_preprossesing.py
the block in lines 161–229 is currently commented out.

If you want to generate summaries as well, simply uncomment this block.

Note: Running this step requires downloading a local model to your machine.

yaml
Copy code

# Usage
Run the main application using the command below. You will be guided through the setup interactively in the terminal.

```Bash
python main.py
```
 1. API Token & Initialization
   
    When the script runs, follow the terminal prompts:
    1. Hugging Face Login: You will be asked to paste your Hugging Face Access Token directly into the terminal to authenticate.
    2. EDA Step: You will see the prompt:
    
```Plaintext
Type 'skip' to skip the EDA step, or write anything else to continue:
```
 * Type skip: Launches the Chat App immediately (once data is ready might take up to 2 minutes).
 * Type anything else: Runs the full data preprocessing, merging, and embedding pipeline before launching the app.
   
2. Web App Workflow (Gradio)
Once initialization is complete, the console will provide a local URL (e.g., Running on local URL: http://127.0.0.1:7405). Open this link in your browser.

   1. Login: Start by typing your name in the input box.

   2. Profile Setup: The system will prompt you in the chat to input your preferred genres, artists, and songs to build your user profile.

   3. Prompt Entry (Query): Once your profile is saved, the system will ask you what you would like to listen to today (e.g., "retro 80's vibes with          theme of love").
      *  Type: Write your request in the text box.

      * Speak: Click the 🎤 Record button to use your microphone.

   4.   Result: Receive a generated 5‑song playlist with AI-written explanations.

   5. Refine: Swap songs, regenerate, or edit your prompt for new results.
  
##  How It Works

1. **Input**  
   The user submits a natural‑language prompt - either by typing or using voice input (via microphone).

2. **Retrieval**  
   The system embeds the prompt and queries the vector database to retrieve semantically similar tracks using `rag.py`.

3. **Contextualization**  
   Retrieved tracks are enriched with the user's profile data (preferred genres, artists, favorite songs) and the prompt context.

4. **Generation**  
   This combined context is passed to the LLM, which generates a tailored 5‑song playlist along with explanations for why each song fits.

5. **Refinement**  
   The user can refine the playlist by:
   - Swapping individual tracks
   - Regenerating the playlist
   - Editing the original prompt

6. **Persistent Personalization**  
   Once a user logs in for the first time by entering their name and setting up preferences, their profile is saved.  
   On future visits, entering the same name automatically loads their preferences - enabling a personalized experience across sessions.


# Optional EDA / Data Analysis
The main.py script includes an integrated EDA (Exploratory Data Analysis) step. If you choose not to skip it, the system will:

   * Load the spotify_dataset.csv.

   * Convert dates and merge with Billboard popularity data.

   * Embed song lyrics for the retrieval model.

   * Generate column summaries for the LLM context.


# Project Presentation

* [▶️ Watch the Demo](https://drive.google.com/file/d/1Ez1RjBV9MMzhJ6zLJLmUPTz72QArhiMI/view?usp=sharing)


# Evaluation & Reproducibility

The system's performance was evaluated using both **quantitative metrics** and **qualitative human judgment** to assess playlist quality and personalization effectiveness.

---

### Quantitative Metrics

We employed **Normalized Discounted Cumulative Gain (NDCG)** at ranks 1 and 5:

- `NDCG@1` and `NDCG@5`

These metrics measure the relevance and ranking quality of top-recommended tracks, based on a **graded scale**:

- `0` = Not relevant  
- `1` = Relevant  
- `2` = Highly relevant  

We compared the RAG-based playlist generator against a **GPT-only baseline**, where the LLM was given the entire candidate table without structured retrieval.

The evaluation also segmented results into:

- **Known Profile** – when the user’s musical preferences were stored  
- **Unknown Profile** – when no prior profile was available  

This helped quantify the value of dynamic personalization.

---

Ensure the following Python libraries are installed:

```bash
pip install pandas openpyxl
```


# Contributors

| Name            | GitHub                       |
|-----------------|------------------------------|
| Yuval Margolin  | [yuvalmar16](https://github.com/yuvalmar16) |
| Ravid Gersh     | [RavidGersh59](https://github.com/RavidGersh59) |
| Daniel Maor     | [danielmaor0808](https://github.com/danielmaor0808) |
