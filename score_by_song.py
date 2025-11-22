import ast
import pandas as pd
from data_preprocessing import SEMANTIC_MODEL
from openai_client import ask_gpt
import re
import numpy as np
import pandas as pd
import torch
from sentence_transformers import util
import states









# TODO INPOTY  import utils
def parse_and_filter(gpt_response, column_summary_text):
    """
    df: pandas DataFrame
    gpt_response: string like "{'Genre': 'Pop', 'Energy': '>=70'}"
    column_summary_text: optional explanation text to re-ask GPT if needed
    ask_gpt: function to regenerate a valid GPT response
    """

    print(gpt_response, "gpt_response")
    df = states.df

    
    Artists_Popularity_df = states.Artists_Popularity_df
    # convert gpt respons for user demands to a python dict, key - the feature, val - the relevant values for the user for the feature. we will use only key the reflect the user current request
    response_dict = None
    mone = 0
    while True:
        try:
            response_dict = ast.literal_eval(gpt_response.strip('"'))
            if isinstance(response_dict, dict):
                break
        except Exception:
            pass
        if ask_gpt: # if it is not in the correct format we will ask to translate again the user prompt to frature

            if mone > 2:
                raise ValueError("GPT response could not be parsed and no ask_gpt function provided")


            gpt_response = ask_gpt(gpt_response, column_summary_text)
            # print("didnt read {gpt_response}")
            mone = mone +1
        else:
            raise ValueError("GPT response could not be parsed and no ask_gpt function provided")

    # hangle similar artists - the user asked for specific artists we will still add him to similar artists cols in order to capture similar artists to him - you like david guetta so you will like martix garrix - similar guettas artist#
    artist_keys = ["artist", "artists", "artist(s)", "Artist(s)"] # if the user asked for artist style he will use one of those keys
    artist_val = next((response_dict[k] for k in artist_keys if k in response_dict), None)

    if artist_val:
        similar_artist_key = next(
            (k for k in response_dict if k.lower().replace(" ", "_") in ["similar_artist", "similar_artists"]),
            "Similar Artists"
        )
        val = response_dict.get(similar_artist_key, [])
        if isinstance(val, str):
            val = [s.strip() for s in val.split("&")]
        elif not isinstance(val, list):
            val = [str(val)]
        if isinstance(artist_val, str):
            artist_items = [a.strip() for a in artist_val.split("&")]
        elif isinstance(artist_val, list):
            artist_items = artist_val
        else:
            artist_items = [str(artist_val)]
        response_dict[similar_artist_key] = list(set(val + artist_items))



    # same thing we have done above with artist - now for songs #
    song_keys = ["song", "songs", "Song", "Songs"]
    song_val = None
    for k in song_keys:
        if k in response_dict:
            song_val = response_dict[k]
            break
    if song_val:
        similar_song_key = next(
            (k for k in response_dict if k.lower().replace(" ", "_") in ["similar_song", "similar_songs"]),
            "Similar Song"
        )
        val = response_dict.get(similar_song_key, [])
        if isinstance(val, str):
            val = [s.strip() for s in val.split("&")]
        elif not isinstance(val, list):
            val = [str(val)]
        if isinstance(song_val, str):
            song_items = [s.strip() for s in song_val.split("&")]
        elif isinstance(song_val, list):
            song_items = song_val
        else:
            song_items = [str(song_val)]
        response_dict[similar_song_key] = list(set(val + song_items))

        
    bilabord_flag =True
    #we have dict with the user demands, we want to order the keys and iterate them using the most selective columns at first for better runtimeperformences
    preferred_order = ["Popularity_Today", "Release Date", "song"] # the main selective columns 

    # check if they are exist
    ordered_keys = [k for k in preferred_order if k in response_dict]

    remaining_keys = [k for k in response_dict if k not in ordered_keys]

    # using text column in the end if it exist - because we would like to give scores before applying semantic search - filter irrelevant songs before applying time-consuming operation - semantic search 
    if "text" in remaining_keys:
        remaining_keys.remove("text")
        remaining_keys.append("text")

  
    ordered_keys += remaining_keys


    response_dict = {k: response_dict[k] for k in ordered_keys}


    # initial each songs with the score 1 - we will use multiplications for each feature in order to detect songs the answear multiple demands better. we will give relevant weight for each feature
    filtered_df = df.copy()
    filtered_df["score"] = 1.0

    # filter by datetime - same logic we have done in the top of the cell - to check that we didnt miss this filter if it was a misscorect spelling of the feture name



    for key, value in response_dict.items():
        # print(f" key {key}, val {value}")

        if key == "text" or key == "mistral_summary":
            continue  # hanndle text for semantic search at the end, and we will use mistral summary for rag (it is better to dectect content theme by semantic search )


        # handeling grammer errors - for example if the system used artist as the name of the feature instead of Artist(s) - the actual name
        original_key = key
        candidate_keys = [
            key,
            key.lower(),
            key.capitalize(),
            key + "s",
            key.lower() + "s",
            key.capitalize() + "s"
        ]

        matched_col = next((c for c in candidate_keys if c in df.columns), None)

        if not matched_col:
          
            continue

        key = matched_col
        if isinstance(value, str):
            value = [value]
        elif isinstance(value, (int, float)):
            value = [str(value)]

        value_lower = [str(v).lower() for v in value]

        # use it in order to check for 1 song/artist in the relevant cols - beacuse we want to capture also songs the the artist performed their while he collaborate with other artsts
        pattern = '|'.join(re.escape(v) for v in value_lower)



        #filter by release date col
        if key == "Release Date" or  key == "release date" or key == "release_date" or key == "Release_Date":
            value = str(response_dict["Release Date"]).strip().lower()
            print(f" Before filtered by Release Date '{value}', remaining rows: {len(filtered_df)}")


            try:
                col = filtered_df["Release Date"]
                if not pd.api.types.is_datetime64_any_dtype(col):
                    col = pd.to_datetime(col, errors="coerce")

                #col not valid
                if col.isnull().all():
                    print("error ")
                else:
                    # filter by over/under (<2000, >=1988)
                    if any(op in value for op in ['>=', '<=', '>', '<']):
                        threshold_match = re.search(r"\d{4}", value)
                        if threshold_match:
                            year = int(threshold_match.group())

                            if '>=' in value:
                                condition = col.dt.year >= year
                            elif '<=' in value:
                                condition = col.dt.year <= year
                            elif '>' in value:
                                condition = col.dt.year > year
                            elif '<' in value:
                                condition = col.dt.year < year
                            else:
                                condition = col.dt.year == year

                            filtered_df = filtered_df[condition]

                    # filter by range (1990-1995)
                    elif re.fullmatch(r"\d{4}-\d{4}", value):
                        start, end = map(int, value.split("-"))
                        condition = col.dt.year.between(start, end)
                        filtered_df = filtered_df[condition]

                    #filter by decade (80s, 2000s)
                    elif re.fullmatch(r"\d{2}s|\d{4}s", value):
                        decade = int(value.rstrip('s'))
                        if decade < 100:
                            decade += 1900 if decade != 0 else 2000
                        start, end = decade, decade + 9
                        condition = col.dt.year.between(start, end)
                        filtered_df = filtered_df[condition]

                    # filter by one year (e.g songs from 2015)
                    elif re.fullmatch(r"\d{4}", value):
                        year = int(value)
                        condition = col.dt.year == year
                        filtered_df = filtered_df[condition]

                    else:
                        print(f"error Release Date: {value}")

            except Exception as e:
                print(f"error Release Date: {e}")

            print(f" After filtered by Release Date '{value}', remaining rows: {len(filtered_df)}")






        # ========== SONG - the user want to include specific song ==========        
        
        if original_key.lower() in ["song", "songs"] or key == "song" or key == "Song":            # all possible names for songs

            if isinstance(value, str): #if string we will convert to list for easier handelling also we will chaeck if more than 1 songs styles (if he wanted similar to 2 or more songs)
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        value = parsed
                    else:
                        value = [value.strip()]
                except:
                    value = [value.strip()]
            elif not isinstance(value, list):
                value = [str(value).strip()]

            for song_name in value:
                # print(f"song name {song_name}")
                song_name_clean = song_name.lower().strip()
                mask = (
                    filtered_df["song"]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .str.contains(song_name_clean, case=False, na=False)
                )

                matching_songs = filtered_df[mask]

                # take the top 3 songs with this names (for example there are 2 popular songs who called 'love story' we don't know now which is relevant to the query)
                top3 = matching_songs.sort_values(by="Popularity_Today", ascending=False).head(3)

                # print(f"song {top3['song']}")

                #  if the user want specific songs we will give to it great score
                filtered_df.loc[top3.index, "score"] = 99999999

        # ========== ARTIST - the user want to include specific artist ==========
        elif original_key.lower() in ["artist", "artists", "artist(s)"] or key == "Artist(s)":
            values = value if isinstance(value, list) else [value]
      

            for artist in values:  # same logic we have done with songs
                artist = str(artist).strip().lower()
                try:
                    pop_row = Artists_Popularity_df[
                        Artists_Popularity_df["Artist(s)"].astype(str).str.lower().str.strip() == artist
                    ]
                    popularity = float(pop_row["Custom_Popularity"].values[0]) if not pop_row.empty else 1
                except Exception as e:
                    print(f" Error fetching popularity for '{artist}': {e}")
                    popularity = 1

                mask = filtered_df["Artist(s)"].astype(str).str.lower().str.contains(artist, regex=False, na=False)
                matching_rows = filtered_df[mask]

                # we will take the artists top 5 popular songs and give them a great score
                top5_idx = matching_rows.sort_values(by="Popularity_Today", ascending=False).head(20).index

                filtered_df.loc[top5_idx, "score"] *= 100 / popularity * 3.5 # normalize the score by the artists popularity in order to give great score if the user want unpopular artists so to avoid favorable only to popular artists




        # ========== SIMILAR ARTIST - the user want style of artists (e.g songs *like* david guetta) ==========
        elif original_key.lower().replace(" ", "_") in ["similar_artist", "similar_artists"] or original_key == "Similar Artists" or original_key == "Similar Artist": # if the system made grammer error while writing the key name
            mask = filtered_df["Similar Artists"].astype(str).str.lower().str.contains(pattern, regex=True, na=False) # if the user love david duetta we will give better score to the songs david guetta took part at - e.g david duetta have many collaborations with other artists
            count = mask.sum()
            
            if isinstance(value, str): #should be str
                try:
                    value = ast.literal_eval(value)
                except:
                    value = []

       


            filtered_df.loc[mask, "score"] *= 5 # if the user love david duetta we will give better score to leading songs that made by similar artists to him 

            for sim_artist in value:
                sim_artist_clean = sim_artist.lower().strip()
                artist_mask = filtered_df["Artist(s)"].astype(str).str.lower().str.contains(sim_artist_clean, regex=False, na=False)
                matched = filtered_df[artist_mask]

                if not matched.empty:
                    top2_idx = matched.sort_values(by="Popularity_Today", ascending=False).head(20).index 

                    try:
                        pop_row = Artists_Popularity_df[
                            Artists_Popularity_df["Artist(s)"].astype(str).str.lower().str.contains(sim_artist_clean, regex=False, na=False)
                        ]
                        popularity_art = float(pop_row["Custom_Popularity"].values[0]) if not pop_row.empty else 1
                    except Exception as e:
                        print(f" Error fetching popularity for '{sim_artist_clean}': {e}")
                        popularity_art = 1
                    if popularity_art <10:
                        popularity_art =10
                    multiplier = 100 / popularity_art * 3.5 # popularity_art represent the popularity of the artists - we want to give better score to unpopular relevant artists in order to avoid imbalance favorable to popular artists
                    filtered_df.loc[top2_idx, "score"] *= multiplier

                    
                                

                   


        # ========== SIMILAR SONGS - (e.g the user want similar song to 'play hard') ==========
        elif original_key.lower().replace(" ", "_") in ["similar_song", "similar_songs"] \
            or original_key in ["Similar Songs", "Similar Song"] or key in ["Similar Songs", "Similar Song"]: #same as above if misspell the key

            # should be list
            values = value if isinstance(value, list) else [value]
            target_songs = set(str(v).strip().lower() for v in values if pd.notna(v))
            
            #detect if the user used not the full name - for example in python the name of the song 'play hard' is 'play hard (feat. Akon)' so if the user wrote play hard we want also to fing this songs so we used contains
            exploded = filtered_df[["Similar_Songs_list"]].explode("Similar_Songs_list")
            exploded["Similar_Songs_list"] = exploded["Similar_Songs_list"].astype(str).str.lower().str.strip()
            exploded["is_match"] = exploded["Similar_Songs_list"].isin(target_songs)
            matched_indices = exploded[exploded["is_match"]].index.unique()
            filtered_df.loc[matched_indices, "score"] *= 5
          
          

            # take the similar songs for song 'A' - and give better score the the most 2 popular of them
            count_boosted = 0
            song_col = filtered_df["song"].astype(str).str.lower()
            for sim_song in target_songs:
                match_mask = song_col.str.contains(re.escape(sim_song), case=False, na=False)
                matched = filtered_df[match_mask]
                if not matched.empty:
                    top2_idx = matched.sort_values(by="Popularity_Today", ascending=False).head(2).index
                    filtered_df.loc[top2_idx, "score"] *= 10
                    count_boosted += len(top2_idx)



        # ========== GENERAL CASE ==========
        elif key in df.columns: #if it is apply the conditions we will give better score for every feature
            col = filtered_df[key]
            if pd.api.types.is_string_dtype(col):
                mask = col.astype(str).str.lower().str.contains(pattern, regex=True, na=False)
            else:
                try:
                    value_num = [float(v) for v in value if isinstance(v, (int, float)) or str(v).replace('.', '', 1).isdigit()]
                    mask = col.isin(value_num)
                except:
                    mask = pd.Series(False, index=filtered_df.index)

            filtered_df.loc[mask, "score"] *= 1.5



            # convert type to list for uniformity
            col = df[key]
            

            
                        
            # check that the col is real in the df to avoid errors
            if key not in df.columns or key not in filtered_df.columns:
                print(f" Column '{key}' not found — skipping")
                continue


            #handle only popularity - it is important thant others so we will give it a better scores and filter irrelevant
            if key == "Popularity_Today":
                if isinstance(value, list) :
                    value = value [0]
                # print(value)
                


                # if the system translate his demands to range - i.e above 140 tempo if he wanted song for sport
                if "-" in value and not any(op in value for op in ['<', '>', '=']):
                    try:
                        lower, upper = map(float, value.split("-"))
                        condition = filtered_df[key].between(lower, upper)
                  
                        filtered_df.loc[condition, "score"] *= (1 + filtered_df.loc[condition, key].fillna(1) / 80) # the much its is higher than the require treshold the higher score it will get
                        filtered_df = filtered_df[condition]
                       

                        continue
                    except Exception as e:
                        print(f" Failed to parse range '{value}': {e}")
                        continue


                threshold_match = re.search(r"[-+]?\d*\.?\d+", value)
                threshold = float(threshold_match.group())

                # # the much its is higher than the require treshold the higher score it will get - numerical tresholds cols
                col = filtered_df[key]
                if '>=' in value:
                    condition = col >= threshold
                    filtered_df.loc[condition, "score"] *= (1 + col[condition].fillna(1) / 80)
                elif '<=' in value:
                    condition = col <= threshold
                    filtered_df.loc[condition, "score"] *= (1 + (threshold - col[condition].fillna(1)) / 80)
                elif '>' in value:
                    condition = col > threshold
                    filtered_df.loc[condition, "score"] *= (1 + col[condition].fillna(1) / 80)
                elif '<' in value:
                    condition = col < threshold
                    filtered_df.loc[condition, "score"] *= (1 + (threshold - col[condition].fillna(1)) / 80)

                # filter songs if the user want popular songs we will eliminate unpopular - it will help us for faster process and it is mandatory condition in most of the cases - (mostly he will want popular songs)
                if key == "Popularity_Today":
                
                    filtered_df = filtered_df[condition]
                    continue


            # handelling 'good for' (e.g good for running) cols that are binary col with 0 and 1 - due to lack of once we will not filter them but we will give them better score for anything that meet this condition

            if key.lower().startswith("good for") and isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):

                first_val = value[0]
                try:
                    numeric_val = float(first_val)
                    is_one = numeric_val == 1.0
                except (ValueError, TypeError):
                    is_one = False

                if key.lower().startswith("good for") and is_one:
                    condition = filtered_df[key].astype(str).str.strip() == '1'
                    num_matched = condition.sum()
                    filtered_df.loc[condition, "score"] *= 1.5
                    continue



            #now in the rest columns to check if it is list the value only should be 1 value - the first
            if isinstance(value, list) and len(value) > 0:
                value = value[0]


            # numerical conditions that still can be are tresholds and in the format of <>=
            if isinstance(value, str) and (
                any(op in value for op in ['>=', '<=', '>', '<']) or 
                ('-' in value and not any(op in value for op in ['>', '<', '=']))
            ):
                try:
                    col = filtered_df[key]

                    # range
                    if '-' in value and not any(op in value for op in ['>', '<', '=']):
                        lower, upper = map(float, value.split("-"))
                        condition = col.between(lower, upper)
                        # print(f"🔍 Key: {key}, Range: {lower}-{upper}, Matches: {condition.sum()} rows")
                        filtered_df.loc[condition, "score"] *= (1 + col[condition].fillna(1) / 80)
                        continue

                    threshold_match = re.search(r"[-+]?\d*\.?\d+", value)
                    if not threshold_match:
                        continue
                    threshold = float(threshold_match.group())

                    # check that we didnt had error with the datetime cols
                    if 'Date' in key and pd.api.types.is_datetime64_any_dtype(col):
                        year_col = col.dt.year
                        if '>=' in value:
                            condition = year_col >= threshold
                        elif '<=' in value:
                            condition = year_col <= threshold
                        elif '>' in value:
                            condition = year_col > threshold
                        elif '<' in value:
                            condition = year_col < threshold
                        filtered_df = filtered_df[condition]
                        continue

                    # handle numeric cols (except popularity) - don't filter because we dont want to lose important songs and to keep with small songs datasets, but as long as they are better in this category the higher score they will get
                    if '>=' in value:
                        condition = col >= threshold
                        filtered_df.loc[condition, "score"] *= (1 + col[condition].fillna(1) / 80)
                    elif '<=' in value:
                        condition = col <= threshold
                        filtered_df.loc[condition, "score"] *= (1 + (threshold - col[condition].fillna(1)) / 80)
                    elif '>' in value:
                        condition = col > threshold
                        filtered_df.loc[condition, "score"] *= (1 + col[condition].fillna(1) / 80)
                    elif '<' in value:
                        condition = col < threshold
                        filtered_df.loc[condition, "score"] *= (1 + (threshold - col[condition].fillna(1)) / 80)


                    if key == "Popularity_Today":
                        filtered_df = filtered_df[condition]



                except Exception as e:
                    if key != 'Release Date':
                        print(f" Failed numeric condition for '{key}': {e}")
                    continue
            


            # 2.give score for songs that are in relevant range
            elif isinstance(value, str) and '-' in value and value.replace('-', '').replace('.', '').isdigit():
                try:
                    v_min, v_max = map(float, value.split('-'))

                    if 'Date' in key and pd.api.types.is_datetime64_any_dtype(filtered_df[key]):
                        condition = filtered_df[key].dt.year.between(int(v_min), int(v_max))
                        filtered_df = filtered_df[condition]
                    else:
                        condition = filtered_df[key].between(v_min, v_max)
                        filtered_df.loc[condition, "score"] *= 2


                except Exception as e:
                    print(f" Failed to parse range '{value}' for '{key}': {e}")


            # give score to songs that are fullfil one condition (e.g if for key = emotion, val = 'love' we will give higher score for love songs)
            elif isinstance(value, str) :
                condition = filtered_df[key].astype(str).str.lower() == value.lower()
                multiplier = 2  # default

                #give higher scores for genres and emotions if relevant - ecpecially for specific and meaningful genres and emotions 

                # joy and pop are most common so they will get not higher score compare to rock and love that are more meaningful and unique.
                if key.lower() == "emotion":
                    if value.lower() == "love":
                        multiplier = 2.7
                    elif value.lower() != "joy":
                        multiplier = 3
                elif key.lower() == "genre":
                    if value.lower() not in ["pop", "dance"]:
                        multiplier = 4.2
                    elif value.lower() != "pop":
                        multiplier = 3

                filtered_df.loc[condition, "score"] *= multiplier
              


            # if the demands is for a list of genres and emotions we will give lower score for each (if he ask fo 3-4 genres it is less important to him than if he ask only for 1 genre)
            else:
                if isinstance(value, (int, float)):
                    value = [str(value)]
                elif isinstance(value, str):
                    value = [value]

                value_lower = [str(v).lower() for v in value]

                if pd.api.types.is_string_dtype(col):
                    pattern = '|'.join(re.escape(v) for v in value_lower)
                    condition = filtered_df[key].astype(str).str.lower().str.contains(pattern, na=False, regex=True)
                else:
                    condition = filtered_df[key].isin(value)

                multiplier = 2
    

                filtered_df.loc[condition, "score"] *= multiplier


        bilabord_cols = [
            'appear_in_weekly_bilabord_in_decade_1950',
            'appear_in_weekly_bilabord_in_decade_1960',
            'appear_in_weekly_bilabord_in_decade_1970',
            'appear_in_weekly_bilabord_in_decade_1980',
            'appear_in_weekly_bilabord_in_decade_1990',
            'appear_in_weekly_bilabord_in_decade_2000',
            'appear_in_weekly_bilabord_in_decade_2010',
            'appear_in_weekly_bilabord_in_decade_2020'
        ]

        # give a relevant score for popular songs today and in the past based on the relevant required era
        if (key in bilabord_cols or key == "Popularity_Today") and bilabord_flag: # bilabord_flag = true if he asked for a song that was popular in a specific era, Popularity_Today is if he asked for a song that popular today

                
            if key ==  "Popularity_Today"  : 
                max_val = filtered_df[key].max()
                if pd.isna(max_val) or max_val == 0:
                    max_val = 1  # avoid division by 0 

                bilabord_score = (
                    1 +
                    3.75 * filtered_df[bilabord_cols].fillna(0) / max_val
                ) #the formula made after a lot of attempts and fine tuning 

                # check the same indexes
                bilabord_score = bilabord_score.reindex(filtered_df.index)

            elif key in bilabord_cols:
                max_val = filtered_df[key].max()
                if max_val > 0:
                    filtered_df["score"] *= (1 + 3.75 * filtered_df[key].fillna(0) / max_val)







    if "text" in response_dict and "embedding" in filtered_df.columns:
        # for semantic search we will like also to give a little bit score for their popularity
        if (
            "Popularity_Today" not in response_dict and
            not any(col in response_dict for col in bilabord_cols)
        ):
            filtered_df["score"] *= (
                1 + filtered_df["appear_in_weekly_bilabord_in_decade_2020"].fillna(0) / 8
            )

        filtered_df = filtered_df[
            filtered_df["embedding"].apply(
                lambda x: isinstance(x, (list, np.ndarray)) and len(x) == 384
            )
        ].copy()
        # convert the theme and the required query theme to the embedding for embedding search
        query_text = (
            " ".join(str(t) for t in response_dict["text"])
            if isinstance(response_dict["text"], list)
            else str(response_dict["text"])
        )

        semantic_model = SEMANTIC_MODEL
        query_embedding = semantic_model.encode(
            query_text, convert_to_tensor=True, normalize_embeddings=True
        )

        # emmbeding search by similarity and give greater score for songs with better similarity
        try:
            song_embeddings_tensor = torch.stack([
                torch.tensor(emb, dtype=torch.float32) for emb in filtered_df["embedding"]
            ])
        except Exception as e:
            print(" error while converting embedding:", e)
            return filtered_df


        similarities = util.cos_sim(query_embedding, song_embeddings_tensor)[0].cpu().numpy()

        # create a col for the semantic score - with meaningful name that the rag will understand
        semantic_col = f"semantic_score_{query_text.strip().lower().replace(' ', '_')}"
        filtered_df[semantic_col] = similarities



        # give to the songs that have higher that 0.25 similarity better score - made after a lot of fine tuning
        mask = filtered_df[semantic_col] > 0.25
        filtered_df.loc[mask, "score"] *= (1 + 25 * filtered_df.loc[mask, semantic_col])

    return filtered_df.copy().nlargest(200, "score") #Returen the top 200 candidates- *It is not the final df before the rag* after that we will filter the returned list for keeping diversity and send to more than 45 songs




def score_by_artist(df):
    """#find the 5-th popular songs if exist by the artist in order to evaluate his popularity"""
    #aggregate evry artist's songs
    df_expanded = df.copy()
    df_expanded["Artist(s)"] = df_expanded["Artist(s)"].astype(str)
    df_expanded = df_expanded.assign(
        **{"Artist(s)": df_expanded["Artist(s)"].str.split(",")}
    ).explode("Artist(s)")
    df_expanded["Artist(s)"] = df_expanded["Artist(s)"].str.strip()

    # sort the artist's songs by decending order
    df_sorted = df_expanded.sort_values(by=["Artist(s)", "Popularity_Today"], ascending=[True, False])

    # return the fifth popular song
    def fifth_or_lowest(group):
        if len(group) > 4:
            return group.iloc[4]["Popularity_Today"]
        else:
            return group.iloc[-1]["Popularity_Today"]

    # produce
    Artists_Popularity_df = df_sorted.groupby("Artist(s)").apply(fifth_or_lowest).reset_index()
    Artists_Popularity_df.columns = ["Artist(s)", "Custom_Popularity"]

    return Artists_Popularity_df
