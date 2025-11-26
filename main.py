import pandas as pd
import numpy as np
import torch
import os
import re
import json
import ast
import math
import inspect
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import gradio as gr
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor
from data_preprocessing import *
import gradio as gr
import inspect
from rag import *
from gradio_helpers import *
from record_queries import transcribe_speech
from states import users ,column_summary_text, df, Artists_Popularity_df
from score_by_song import score_by_artist

port_val = 7405
column_summary_text = ''

def main():
    user_input = input("Type 'skip' to skip the EDA step, or write anything else to continue: ").strip().lower()

    if user_input == 'skip':
        print(" Skipping EDA step...")
        df = load_df_after_eda()


    else:
        print(" Loading data...")
        # read data
        df = pd.read_csv("spotify_dataset.csv",encoding="utf-8")

        print(" Preprocessing data...")
        # convert release dates to datetime
        df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

        # initialize sematntic model
        semantic_model = SEMANTIC_MODEL

        # load billboard decade-based data
        decade_pivot = load_decade_popularity_df()


        print(" Merging and aggregating data...")
        #join df with decade_pivot for popularity on each relevant era
        df = df.merge(decade_pivot, how='left', left_on=["Artist(s)", "song"], right_on=["artist", "song b"])
        df.drop(columns=["artist", "song b"], inplace=True)
        df.rename(columns={"Popularity": "Popularity_Today"}, inplace=True)
        
        # concatonate similar artists and songs to one list
        df.rename(columns={"Popularity": "Popularity_Today"}, inplace=True)

        df = aggregate_cols(df)

        df = sort_df_by_decade(df)

        # due to time and recources we will aplly songs summary to songs that hitted once the bilabord (since 60s)/spotify charts (2021-2024) 
        decade_cols = [
            'appear_in_weekly_bilabord_in_decade_1950',
            'appear_in_weekly_bilabord_in_decade_1960',
            'appear_in_weekly_bilabord_in_decade_1970',
            'appear_in_weekly_bilabord_in_decade_1980',
            'appear_in_weekly_bilabord_in_decade_1990',
            'appear_in_weekly_bilabord_in_decade_2000',
            'appear_in_weekly_bilabord_in_decade_2010',
            'appear_in_weekly_bilabord_in_decade_2020'
        ]

        df['mistral_summary'] = None
        # # add a summary column to 7k songs in the dataframes (the ones that appeared in billboard) - TODO add lines : df = add_song_summaries(df)
        # 

        # embed the songs' lyrics of the ~20k most popular songs across all decades in the data 
        df = embed_df(df, decade_cols)

        # if you want to summerize the top 7000 popular songs accross all the avilabe decades you should run the following line : df = add_song_summaries(df)



    print(" Preparing column summaries...")
  
    # add explanations and stats on column names
    global column_summary_text 
    column_summary_text = summarize_columns(df)
    states.column_summary_text = column_summary_text
    states.df = df 
    states.Artists_Popularity_df = score_by_artist(df)
    print(df.head())
    #TODO DELEATE FOLLOWING AFTER
    df.to_csv("output.csv", index=False, encoding="utf-8")
    print(df.columns)
    print(" Launching Gradio app...")
    with gr.Blocks() as demo:
        gr.Markdown("## 🗣️ Music Chat\nStart by typing your **name** or click the mic.")

        conversation_box = gr.Textbox(label="Conversation", lines=20, interactive=False)
        input_box = gr.Textbox(label="Write your message", placeholder="Start with your name")
        mic_button = gr.Button("🎤 Record") # allowing option to record 

        state = gr.State(["", 0, None, None])

        input_box.submit(
            handle_message,
            inputs=[input_box, state],
            outputs=[conversation_box, state, input_box]
        )

        def mic_and_send(state):
            text = transcribe_speech()
            return next(handle_message(text, state))

        mic_button.click( # if the user want to record himself instead of sending a message he can use the mic
            fn=mic_and_send,
            inputs=[state],
            outputs=[conversation_box, state, input_box]
        )

    demo.launch(share=True, server_port=port_val)


if __name__=='__main__':

    main()

