# evaluation_metrics.py
import pandas as pd
import numpy as np

# --- 1. NDCG Calculation ---

def compute_average_ndcg_all_segments(file_path: str) -> dict:
    """
    Computes average NDCG@1 and NDCG@5 for:
    - All prompts
    - Prompts with 'profile=Unknown'
    - Prompts with 'profile' known (not unknown)

    Assumes the NDCG scores in the Excel file represent the human/GPT relevance 
    judgment for the final recommended playlist.

    Parameters:
    - file_path (str): Path to .xlsx Excel file containing graded playlist data.

    Returns:
    - dict: Dictionary with average NDCG@1 and NDCG@5 per group (All, Unknown, Known).
    """
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return {}

    # Determine which column holds the first song score for NDCG@1 proxy.
    # NDCG@1 = (Relevance_Score @ 1) / (Ideal_Relevance @ 1)
    # Since ideal relevance (IDCG@1) is 2, and relevance score is 0, 1, or 2, 
    # NDCG@1 = Score@1 / 2.
    first_score_col = None
    for col in ["song 1", "Score 1"]:
        if col in df.columns:
            first_score_col = col
            break
    if first_score_col is None:
        raise ValueError("Could not find 'song 1' or 'Score 1' column for NDCG@1 calculation.")

    # Define masks for known/unknown profiles
    is_unknown = df["prompt"].astype(str).str.contains("profile=Unknown", na=False)
    is_known = ~is_unknown

    # Helper function to compute NDCG averages
    def compute_avg_ndcg(sub_df: pd.DataFrame):
        if sub_df.empty:
            return 0.0, 0.0
        
        # NDCG@1: The actual score is normalized by the maximum possible score (2)
        ndcg1 = (sub_df[first_score_col] / 2).mean() 
        ndcg5 = sub_df["NDCG@5"].mean()
        return round(ndcg1, 4), round(ndcg5, 4)

    # Compute for all groups
    df_all = df
    df_unknown = df[is_unknown]
    df_known = df[is_known]

    print(f"Total entries: {len(df_all)}")
    print(f"Unknown profile entries: {len(df_unknown)}")
    print(f"Known profile entries: {len(df_known)}")

    all_ndcg1, all_ndcg5 = compute_avg_ndcg(df_all)
    unknown_ndcg1, unknown_ndcg5 = compute_avg_ndcg(df_unknown)
    known_ndcg1, known_ndcg5 = compute_avg_ndcg(df_known)

    return {
        "All Users": {"NDCG@1": all_ndcg1, "NDCG@5": all_ndcg5, "N": len(df_all)},
        "Unknown Profile": {"NDCG@1": unknown_ndcg1, "NDCG@5": unknown_ndcg5, "N": len(df_unknown)},
        "Known Profile": {"NDCG@1": known_ndcg1, "NDCG@5": known_ndcg5, "N": len(df_known)}
    }


# --- 2. Inter-Annotator Agreement ---

def compare_user_scores_to_gpt(base_file: str, user_files: list) -> dict:
    """
    Compares human annotator scores (user_files) against the GPT-4o base score (base_file) 
    to calculate inter-annotator agreement (or agreement with the GPT baseline).

    Parameters:
    - base_file (str): Path to Excel file containing the GPT base scores.
    - user_files (list): List of paths to Excel files containing human annotator scores.

    Returns:
    - dict: Dictionary containing the average match percentage for each user file.
    """
    try:
        gpt_df = pd.read_excel(base_file, engine="openpyxl")
    except FileNotFoundError:
        print(f"Error: Base file not found at {base_file}")
        return {}
    
    # Identify score columns (assuming they contain the word 'score' or 'song' + digit 1-5)
    score_columns = [
        col for col in gpt_df.columns 
        if any(keyword in col.lower() for keyword in ['score', 'song']) 
        and any(str(i) in col for i in range(1, 6))
    ]
    # Ensure they are sorted numerically, e.g., 'Score 1', 'Score 2', ...
    score_columns = sorted(score_columns, key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    results = {}
    total_samples = len(gpt_df)

    for user_file in user_files:
        try:
            user_df = pd.read_excel(user_file, engine="openpyxl")
        except FileNotFoundError:
            print(f"Error: User file not found at {user_file}. Skipping.")
            continue

        # Extract score arrays
        user_scores = user_df[score_columns].values
        gpt_scores = gpt_df[score_columns].values

        # Count matches (where human score == GPT score) across all 5 songs
        match_counts_per_row = (user_scores == gpt_scores).sum(axis=1)
        total_possible_matches = total_samples * 5 
        total_actual_matches = match_counts_per_row.sum()

        average_match_percentage = round((total_actual_matches / total_possible_matches) * 100, 2)
        
        results[user_file] = {
            "Total Samples": total_samples,
            "Total Match Percentage": average_match_percentage
        }

    return results

# --- 3. Execution Block ---

if __name__ == "__main__":
    # --- Data Paths ---
    # NOTE: Replace these placeholder file names with your actual, final, and consistent file names.
    RAG_DATA_PATH = "graded_playlists_RAG.xlsx" 
    GPT_BASELINE_DATA_PATH = "Song_Scores_GPT_Baseline.xlsx"
    
    USER_FILES_FOR_AGREEMENT = [
        "Annotator_Ravid_Scores.xlsx", 
        "Annotator_Yuval_Scores.xlsx",
        "Annotator_Daniel_Scores.xlsx"
    ]
    
    # --- A. NDCG Calculation ---
    print("=" * 40)
    print("NDCG CALCULATION (RAG vs. GPT Baseline)")
    print("=" * 40)
    
    try:
        print("\n--- Scores for our RAG model ---")
        averages_rag = compute_average_ndcg_all_segments(RAG_DATA_PATH)
        for category, scores in averages_rag.items():
             # Assumes the table structure from the report (Table 1)
             print(f"{category} (N={scores['N']})")
             print(f"  NDCG@1: {scores['NDCG@1']}, NDCG@5: {scores['NDCG@5']}")
        
        print("\n--- Scores for GPT Baseline model (No RAG/Feature Scoring) ---")
        averages_gpt = compute_average_ndcg_all_segments(GPT_BASELINE_DATA_PATH)
        for category, scores in averages_gpt.items():
            print(f"{category} (N={scores['N']})")
            print(f"  NDCG@1: {scores['NDCG@1']}, NDCG@5: {scores['NDCG@5']}")
    
    except Exception as e:
        print(f"NDCG Calculation failed: {e}")


    # --- B. Agreement Calculation ---
    print("\n" + "=" * 40)
    print("AGREEMENT CALCULATION (Human vs. GPT Baseline Scores)")
    print("=" * 40)
    
    try:
        # Note: The original code compared human scores against a single GPT score file. 
        # We assume the 'RAG_DATA_PATH' contains the RAG results/scores, which GPT uses 
        # to generate the final rank. The comparison should use the scores from the RAG output.
        comparison_results = compare_user_scores_to_gpt(RAG_DATA_PATH, USER_FILES_FOR_AGREEMENT)
        
        i = 1
        for user_file, result in comparison_results.items():
            print(f"Comparison results for Annotator {i} ({user_file}):")
            print(f"  Total Song-Prompt Pairs: {result['Total Samples'] * 5}")
            print(f"  Overall Match Percentage: {result['Total Match Percentage']}%")
            print("-" * 20)
            i += 1
            
    except Exception as e:
        print(f"Agreement Calculation failed: {e}")