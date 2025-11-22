from huggingface_hub import login
import os
# from dotenv import load_dotenv - ?????????????????????????????????????????
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm
import pandas as pd
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import json 
import numpy as np
import re

SEMANTIC_MODEL = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L6-cos-v5")


HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_API")
login(HUGGING_FACE_KEY)

def load_decade_popularity_df():
    """load billboard data and convert it to be decade-based"""

    # load billboard data
    bilabord_df = pd.read_csv("charts.csv")

    # convert to decade-based data
    bilabord_df['date'] = pd.to_datetime(bilabord_df['date'])
    bilabord_df['year'] = bilabord_df['date'].dt.year
    bilabord_df['decade'] = (bilabord_df['year'] // 10) * 10


    decade_counts = (
        bilabord_df.groupby(['artist', 'song', 'decade'])
        .size()
        .reset_index(name='count')
    )


    decade_pivot = decade_counts.pivot_table(
        index=['artist', 'song'],
        columns='decade',
        values='count',
        fill_value=0
    ).reset_index()


    decade_pivot.columns = ['artist', 'song'] + [f'appear_in_weekly_bilabord_in_decade_{col}' for col in decade_pivot.columns[2:]]
    decade_pivot.rename(columns={"song": "song b"}, inplace=True)

    return decade_pivot

def aggregate_cols(df):

    # put all the artists that similar to the performer and also the performer himself in a list - for our reccomendation system
    df['Similar Artists'] = df.apply(lambda row: [
        row['Similar Artist 1'],
        row['Similar Artist 2'],
        row['Similar Artist 3'],
        row['Artist(s)']
    ], axis=1)

    #same things with songs
    df['Similar Songs'] = df.apply(lambda row: [
        row['Similar Song 1'],
        row['Similar Song 2'],
        row['Similar Song 3'],
        row['song'] 
    ], axis=1)

    df["Similar_Songs_list"] = df[["Similar Song 1", "Similar Song 2", "Similar Song 3"]].values.tolist()
    df["Similar_Songs_list"] = df["Similar_Songs_list"].apply(lambda lst: [s for s in lst if pd.notna(s)])

    return df

def sort_df_by_decade(df):
    """This function sorts the data in our df, first be decades & popularity, and lastly by popularity alone"""

    return df.sort_values(by=['appear_in_weekly_bilabord_in_decade_2020',
        'appear_in_weekly_bilabord_in_decade_2010',
        'appear_in_weekly_bilabord_in_decade_2000',
        'appear_in_weekly_bilabord_in_decade_1990',
        'appear_in_weekly_bilabord_in_decade_1980',
        'appear_in_weekly_bilabord_in_decade_1970',
        'appear_in_weekly_bilabord_in_decade_1960',
        'appear_in_weekly_bilabord_in_decade_1950', 'Popularity_Today'],
                    ascending=[False, False,False,False,False,False,False,False,False]
                                                                                )

def embed_df(df, decade_cols):

    df["embedding"] = None
    if "embedding" not in df.columns:
        df["embedding"] = None

    condition_decade = df[decade_cols].gt(0).any(axis=1)
    condition_popularity = df["Popularity_Today"] >= 70
    target_rows = df[condition_decade | condition_popularity]

    rows_to_embed = target_rows[target_rows["embedding"].apply(lambda x: not isinstance(x, list))]


  
    for idx in tqdm(rows_to_embed.index, desc="  embeddings"):
        text = df.at[idx, "text"]
        if isinstance(text, str) and text.strip():
            embedding = SEMANTIC_MODEL.encode(text)
            df.at[idx, "embedding"] = list(embedding)

            
    def clean_embedding(text):
        # case 1: if it's already array -> return as-is
        if isinstance(text, (list, np.ndarray)):
            return np.array(text, dtype=np.float32)

        # case 2: if empty or None -> return empty array
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return np.array([])

        # case 3: text is a string -> extract numbers
        nums = re.findall(r"-?\d+\.\d+(?:e-?\d+)?", str(text))
        return np.array([float(x) for x in nums], dtype=np.float32)

    df["embedding"] = df["embedding"].apply(clean_embedding)
    print(df["embedding"].head()) # TODO DELETE 
    
    # df["embedding"] = df["embedding"].apply(ast.literal_eval)
    # df['embedding'] = df['embedding'].apply(lambda x: json.dumps(x.tolist()))

    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    return df.drop_duplicates(subset=["Artist(s)", "song", "Release Date", "Genre"])



def load_df_after_eda():    
    df = pd.read_csv("texts_with_embeddings.csv",encoding="utf-8")        
    def clean_embedding(text):
        # case 1: if it's already array -> return as-is
        if isinstance(text, (list, np.ndarray)):
            return np.array(text, dtype=np.float32)

        # case 2: if empty or None -> return empty array
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return np.array([])

        # case 3: text is a string -> extract numbers
        nums = re.findall(r"-?\d+\.\d+(?:e-?\d+)?", str(text))
        return np.array([float(x) for x in nums], dtype=np.float32)

    df["embedding"] = df["embedding"].apply(clean_embedding)
    print(df["embedding"].head()) # TODO DELETE 


    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    df["Similar_Songs_list"] = df[["Similar Song 1", "Similar Song 2", "Similar Song 3"]].values.tolist()
    df["Similar_Songs_list"] = df["Similar_Songs_list"].apply(lambda lst: [s for s in lst if pd.notna(s)])
    return df.drop_duplicates(subset=["Artist(s)", "song", "Release Date", "Genre"])

# def add_song_summaries(df):
#     "adds a summary column to 7000 songs"

#     #load mistral 


#     start_index = 0	
#     end_index = 7077  

#     mone = 0
#     for row in df.itertuples(index=True):
        
        
#         i = row.Index
#         if mone < start_index or mone > end_index:
#             mone = mone + 1
#             if mone > end_index: break
#             continue
            
#         if (mone == 7077):
#             print("start")
#         mone = mone + 1
    

#         song = getattr(row, "song", "")
#         artist = getattr(row, "Artist(s)", "")
#         text = getattr(row, "text", "")



        

#         summary = summarize_song(song, artist, text)
#         df.at[i, "mistral_summary"] = summary

#     return df


# #using Mistral model to summarize top 7k songs for mood and content detections while using rag
# def summarize_song(song, artist, text, system_prompt="You are a helpful music assistant."):
#     user_prompt = (
#         f"You are given a song called '{song}'.\n"
#         f"Here are the lyrics:\n{text}\n\n"
#         "What is this song about? Briefly explain the main theme and emotional message "
#         "up to 20 words."
#     )

    
#     model_id = "mistralai/Mistral-7B-Instruct-v0.2"

#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, device_map="auto", torch_dtype=torch.float16
#     )
#     full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt} [/INST]"
  
#     inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=60,  # equall to 20-30 summary words
#             do_sample=False,
#             eos_token_id=tokenizer.eos_token_id,
#         )

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # print(response)
#     return response.split("[/INST]")[-1].strip()


def summarize_columns(df):
    summary = []

    for col in df.columns:
        if col == "text_embedding":
            continue

        col_data = df[col].dropna()
        dtype = df[col].dtype
        summary.append(f"\n🔹 {col}:")

        # datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            summary.append(f"Type: datetime")
            summary.append(f"Range: {col_data.min().date()} – {col_data.max().date()}")

        # numerical columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            summary.append(f"Type: numerical")
            summary.append(f"Range: {col_data.min()} – {col_data.max()}")
            n_unique = col_data.nunique()
            if n_unique < 10:
                summary.append(f"{n_unique} values")

        # list columns
        elif col_data.apply(lambda x: isinstance(x, list)).all():
            flattened = [item for sublist in col_data for item in sublist]
            n_unique = len(set(flattened))
            summary.append(f"Type: list of strings ({n_unique} unique elements total)")

        # string / text columns
        else:
            try:
                n_unique = col_data.nunique()
                summary.append(f"Type: string ({n_unique} unique values)")
            except:
                summary.append(f"Type: string-like, couldn't count uniques")

    return "\n".join(summary)
