# Personalized Music Playlist Generation (RAG)

## 1. Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to create highly personalized music playlists. Unlike traditional collaborative filtering systems, this engine leverages Semantic Search and Large Language Models (LLMs) to understand the context of a user's request and their specific musical taste.

The system retrieves track data from a curated music dataset based on vector similarity and uses an LLM to generate a curated 5-song playlist. Crucially, it provides an **explanation** for every recommendation, bridging the gap between raw data and user intent.

### Key Features
* **RAG Architecture:** Combines vector database retrieval with generative AI for context-aware recommendations.
* **Dynamic User Profiling:** Captures and remembers user preferences (favorite artists, genres, and songs) for tailored experiences.
* **Explainable AI:** The model explicitly explains *why* each song fits the specific prompt and user taste.
* **Interactive Editing:** Users can refine playlists by swapping songs or regenerating recommendations dynamically.

### System Flow
1.  **User Input:** User provides a natural language prompt (e.g., "Songs for a late-night drive").
2.  **Retrieval:** The system embeds the prompt and queries the Vector Database to find relevant tracks from the dataset.
3.  **Augmentation:** Retrieved tracks + User Profile + User Prompt are combined into a prompt for the LLM.
4.  **Generation:** The LLM selects the best 5 songs and generates justifications.

---

## 2. Project Structure

The repository is organized to separate data analysis, backend RAG logic, and the frontend application.

```text
├── data/
│   └── music_dataset.csv       # The source dataset used for retrieval
├── eda/
│   └── data_analysis.ipynb     # Exploratory Data Analysis notebook
├── src/
│   ├── rag_engine.py           # Core logic for retrieval and LLM interaction
│   ├── vector_store.py         # Vector database initialization and search functions
│   └── utils.py                # Helper functions for data processing
├── app.py                      # MAIN ENTRY POINT (Streamlit Web App)
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
└── README.md                   # Project documentation
