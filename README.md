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


# Getting Started
Prerequisites
 * Python 3.8+
 * A valid API key for your LLM provider (e.g., OpenAI).

---

# Installation

1. Clone the repository
   
```bash
git clone https://github.com/yuvalmar16/Personalized-Music-Playlist-Generation.git
cd Personalized-Music-Playlist-Generation
```

2. Create & Activate Virtual Environment
```bash
# Mac / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:**  
On some Windows machines, antivirus software (e.g., Windows Defender) may block parts of Gradio’s installation or execution.  
If you encounter issues launching Gradio, ensure your antivirus is not blocking it.

---

4. Insert Your OpenAI API Key  
Open the file:

`openai_client.py`

In line 8 replace:

```python
openai_api_key = "INSERT YOUR API KEY HERE"
```

with your actual key.

---

# Usage

Run the main application using:

```bash
python main.py
```

When the script starts, you will be guided through setup directly in the terminal:

1. **Hugging Face Token**  
   The system will ask you to paste your **Hugging Face Access Token** for authentication.

2. **Data Manipulation Step**  
   You will see:

   ```
   Type 'skip' to skip the EDA step, or write anything else to continue:
   ```

   - Type **skip** → skip preprocessing and use the ready-made dataset  
   - Type **anything else** → run the full preprocessing pipeline (merging, cleaning, embeddings)

3. **Optional: Summaries Step**  
   In the file `data_preprossesing.py`, lines **161–229** are commented out.  
   Uncomment them if you want the system to generate text summaries as part of preprocessing.

   **Note:**  
   This requires downloading a local model to your computer.

---

## Web App Workflow (Gradio)

After initialization, the console will show a local URL:  
(e.g., `Running on local URL: http://127.0.0.1:7405`)  
Open it in your browser.

1. **Login**  
   Type your name in the input box.

2. **Profile Setup**  
   The system will ask you for your preferred genres, artists, and songs.  
   These build your personalized user profile.

3. **Prompt Entry (Music Request)**  
   Example: *retro 80’s vibes with a theme of love*

   - **Type**: Enter your request in the text box  
   - **Speak**: Click the 🎤 Record button to speak your request  
     **Note:** Voice input supports **Hebrew most reliably**.  
     English works but with lower accuracy.

4. **Result**  
   A **5-song playlist** is generated with **AI-written explanations**.

5. **Refine**  
   You can:
   - regenerate  
   - request modifications  
   - add preferences  
   - swap or remove songs  

6. **End the Session**  
   Type:
   ```
   exit
   ```
   to close the chat session.

---

# Notes & Tips

1. **Not happy with the playlist’s direction, or want to switch to a completely different theme than what you’ve requested so far?**  
   If you want a *completely new playlist*, ask the agent to **search again**, *and in the same sentence* write what you want now.  
   Example:  
   *search again, now I want summer latino vibes focused on dancing*

2. **How to replace one or several songs in the playlist:**
   Do NOT say things like 
   *"replace song number 2"*  
   Instead use:  
   *remove ‘Shape of You’*  
   or  
   *remove Ed Sheeran*

3. **The system remembers you.**  
   After your first login (and even after typing `exit`), your profile persists:
   - preferred genres  
   - favorite artists  
   - disliked songs or genres  
   - contextual preferences based on your chat  

   You can also explicitly tell the agent:  
   *remember that I love The Weeknd*  
   *remember that I don’t like rock*  
   *remember that I prefer calm playlists at night*
   
   You can also ask the agent what your current preference profile looks like after a playlist is generated,
    and request updates or changes to it at any time.
    
    Example:
    “What does my preference profile look like right now?”
    “Update my profile so I prefer more indie and less pop.”

   Your profile updates dynamically based on your interactions.

5. - **Per-User Memory**  
  The system stores a **separate preference profile for each user**, identified **only by their name**.  
  When you reconnect and type the same name, your profile is automatically loaded.
---

# Enjoy your personalized music assistant 🎵

  
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
