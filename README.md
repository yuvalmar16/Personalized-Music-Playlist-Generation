# 🎵 Personalized Music Playlist Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![RAG](https://img.shields.io/badge/Architecture-RAG-green)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Optional EDA](#optional-eda--data-analysis)
- [Demo & Screenshots](#demo--screenshots)
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
* **🔄 Interactive Editing:** After generation, you can swap songs, regenerate the playlist, or refine the prompt/profile for a new result.
* **🗣️ Custom Prompts:** Supports natural‑language prompts like *"Songs for a rainy Sunday afternoon"* or *"Energetic workout mix"*.

---

## Project Structure

```text
(project root)
│── data_preprocessing.py     # Preprocess raw music dataset (cleaning, feature extraction, embedding)
│── rag.py                    # Core logic: embeddings, vector retrieval, LLM input/output handling
│── openai_client.py          # Interface with LLM / OpenAI API
│── score_by_song.py          # (Optional) Script for scoring songs individually
│── states.py                 # Configuration: constants, user profile structure, caching
│── helpers.py                # Utility functions (data loading, parsing)
│── gradio_helpers.py         # Helpers for Web-App UI formatting and output display
│── main.py                   # ORCHESTRATOR: Main entry point (Retrieval + Generation + UI)
│── requirements.txt          # Python dependencies
│── .env.example              # Template for environment variables
└── README.md                 # This documentation
```

# Getting Started
Prerequisites
 * Python 3.8+
 * A valid API key for your LLM provider (e.g., OpenAI).

Installation
1. Clone the repository
   
```bash
git clone [https://github.com/yuvalmar16/Personalized-Music-Playlist-Generation.git](https://github.com/yuvalmar16/Personalized-Music-Playlist-Generation.git)
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

