from concurrent.futures import ThreadPoolExecutor
import ast
from score_by_song import *
from openai_client import *
from states import  users, column_summary_text

from speech_recognition import *
import numpy as np
#columns we will allways send to the rag (we will add other if relevant to the query but we cant send them all the to its financial cost
columns_to_keep = [
    "Artist(s)", "song", "Genre", "Release Date", "emotion", "Tempo", "Danceability",  "Energy", "Positiveness", 'appear_in_weekly_bilabord_in_decade_1950',
       'appear_in_weekly_bilabord_in_decade_1960',
       'appear_in_weekly_bilabord_in_decade_1970',
       'appear_in_weekly_bilabord_in_decade_1980',
       'appear_in_weekly_bilabord_in_decade_1990',
       'appear_in_weekly_bilabord_in_decade_2000',
       'appear_in_weekly_bilabord_in_decade_2010',

       'appear_in_weekly_bilabord_in_decade_2020',
    "Good for Exercise", "Good for Driving","Good for Party", "Good for Work/Study","Good for Relaxation/Meditation" ,"Good for Social Gatherings" ,"Popularity_Today", "score" ,  'Similar Artists', 'mistral_summary'
]

#used in order to clean the returned dictonaries by the gpt in oderder to send it to ours's reccomendations system
def safe_eval_dynamic(text):
    text = text.strip('"') 
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        fixed = re.sub(r'(?<!\\)"(.*?)"(?![:,}\]])', lambda m: f"'{m.group(1)}'" if '"' in m.group(1) else f'"{m.group(1)}"', text)
        try:
            return ast.literal_eval(fixed)
        except Exception as e:
            print(f"error {e}")
            return {}
        

def ask_first (user_query, users_name):
        #first message by the user for playlist

    df = states.df

    #send his current request and history perferences in order to explain in the best way what is he want
    user_query_based_history = (
        f"user history preference {users[users_name]['preference']} take it into account only if it is relevant to the current query. if not relevant at all ignore that"
        f"based on {users[users_name]['sent messages']}. "
        f"user current request is: {user_query}"
    )
    yield " Build a special playlist for you"
    try:
            # we will get 2 dict each represent one element of his profile -good at most for mixing styles (want 80s and today's)

        gpt_response_filtered = ask_gpt_twice(user_query_based_history, column_summary_text)
        gpt_response_filtered_1, gpt_response_filtered_2 = gpt_response_filtered.split("&&&")
        print(gpt_response_filtered_1, gpt_response_filtered_2)
        
        # handle each dict - convert to python from string

        dict1 = safe_eval_dynamic(gpt_response_filtered_1)
        dict2 = safe_eval_dynamic(gpt_response_filtered_2)

    except Exception as e:

        # if gpt returned the dictonaries not in the right format we will call her again
        gpt_response_filtered = ask_gpt_twice(user_query_based_history, column_summary_text)
        gpt_response_filtered_1, gpt_response_filtered_2 = gpt_response_filtered.split("&&&")
        print(gpt_response_filtered_1, gpt_response_filtered_2)

        dict1 = safe_eval_dynamic(gpt_response_filtered_1)
        dict2 = safe_eval_dynamic(gpt_response_filtered_2)
   
   
    # build a dictionaris with all the keys in both dictionaries - we will use all the keys that appeared at atleast one or more dictionaries

    response_dict = {}

    for key in set(dict1) | set(dict2):
        val1 = dict1.get(key)
        val2 = dict2.get(key)

        if isinstance(val1, list) or isinstance(val2, list):
            list1 = val1 if isinstance(val1, list) else [val1] if val1 is not None else []
            list2 = val2 if isinstance(val2, list) else [val2] if val2 is not None else []
            response_dict[key] = list(set(list1 + list2))
        else:
            response_dict[key] = val2 if val2 is not None else val1

# ["Similar Artists", "Similar Songs", "text"] will be used onlt in the search but not while sending to the rag due the the cost of irrelevant words, text is the song summary by mistral
    #valid_gpt_keys will include all the relevant keys in both dict except what we mentioned in the sentence above
    valid_gpt_keys = [
        col for col in response_dict.keys()
        if col in df.columns and col not in ["Similar Artists", "Similar Songs", "text"]
    ]

    # if text his relevant then a semantic similarity score with the value in text is relevant for the promt (e.g want love song  - the val of 'text' will be 'love' and we would like to know songs with higher score with it)
    for single_dict in [dict1, dict2]:
        if "text" in single_dict and single_dict["text"]:
            semantic_col = f"semantic_score_{single_dict['text']}"
            if semantic_col in df.columns:
                valid_gpt_keys.append(semantic_col)

    # all the relevany keys with semantic score if exist (if it was important in the query)
    merged_columns = list(dict.fromkeys(columns_to_keep + valid_gpt_keys))

    with ThreadPoolExecutor() as executor:
        #for faster search we will use threads and run parallarly our reccomenation functions with the 2 dictionaris (each thread with one dict)
        playlist_description = executor.submit(ask_gpt_playlist_preview, gpt_response_filtered)
        future1 = executor.submit(parse_and_filter, gpt_response_filtered_1, column_summary_text)
        future2 = executor.submit(parse_and_filter, gpt_response_filtered_2, column_summary_text)
        response = playlist_description.result()  #wait untill the gpt will publish to the user short description about to next playlist
        yield f" {response}"

    df_filtered_1 = future1.result()
    df_filtered_2 = future2.result()


    # from each returned df we will take the leading songs but no more than 5 songs by each artist for diversity

    df_filtered_1 = df_filtered_1.sort_values(by="score", ascending=False)
    df_filtered_1 = df_filtered_1.groupby("Artist(s)", group_keys=False).head(6)
    df_filtered_1 = df_filtered_1.sort_values(by="score", ascending=False).head(17)

    df_filtered_2_sorted = df_filtered_2.sort_values(by="score", ascending=False)

    df_filtered_2_top5_per_artist = df_filtered_2_sorted.groupby("Artist(s)", group_keys=False).head(6)

    df_filtered_2 = df_filtered_2_top5_per_artist.sort_values(by="score", ascending=False).head(17)

    # join the 2 finsl df 2 one df, overall upto 34 songs will be send to the rag

    df_filtered_1, df_filtered_2 = df_filtered_1.align(df_filtered_2, join='outer', axis=1, fill_value=pd.NA)

    df_filtered = pd.concat([df_filtered_1, df_filtered_2], ignore_index=True)

    # we will not include score col because we dont want to make bias beacuse each dict had diffrent score patterns due to its features *but we will send the list ordered according to the scores*
    safe_group_cols = [
        col for col in df_filtered.columns
        if col != "score"
        and not (
            pd.api.types.is_object_dtype(df_filtered[col]) or
            df_filtered[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any()
        )
    ]

    df_filtered = df_filtered.drop_duplicates(subset=safe_group_cols)


    # convert to csv for rag

    csv_text = df_filtered[merged_columns] \
        .rename(columns={
            'appear_in_weekly_bilabord_in_decade_1960': 'billboard_1960s_weeks',
            'appear_in_weekly_bilabord_in_decade_1970': 'billboard_1970s_weeks',
            'appear_in_weekly_bilabord_in_decade_1980': 'billboard_1980s_weeks',
            'appear_in_weekly_bilabord_in_decade_1990': 'billboard_1990s_weeks',
            'appear_in_weekly_bilabord_in_decade_2000': 'billboard_2000s_weeks',
            'appear_in_weekly_bilabord_in_decade_2010': 'billboard_2010s_weeks',
            'appear_in_weekly_bilabord_in_decade_2020': 'billboard_2020s_weeks',
        }) \
        .to_csv(index=False)



    # system message for the rag to return the relevant songs with explanatin in each and why is it relevant to the query 
    system_messages = [
        {
            "role": "system",
            "content": (
                "You are a music recommendation assistant. "
                "Your job is to create a playlist of 5 songs based ONLY on the list provided below. "
                        "If the user's request doesn't include a specific constraint we can't meet, return a playlist with songs from a variety of artists that alighn the user current query\n."
                        "Balance fit to the user's request with the song’s relative popularity in its time (Popularity_Today or Billboard_Weeks), but never prioritize popularity over relevance.\n "
                        "For each selected song, you must explain clearly and briefly (1 sentence) **why** it was chosen, based on its features in the data (such as lyrics summary if exist, mood, genre, danceability, tempo, today popularity , bilabord weeks in each decade, release date, Instrumentalness, Positiveness Acousticness Speechiness and etc.), based on user's current request. approach their historical preferences only if it is relevant to their current query. \n"
                                                        "you can use popularity today and bilabords week in the relevant decade to explain and give songs with relevant popularity. you can use lyrics summary to explain and give songs with specific meanings. \n"

                "you can understand similarities and relations between songs by the  features in the data \n"
                "Focus your reasoning on the song’s relevance to the user's request"
                "Do NOT invent songs or artists that not it the list. Recommend strictly from the list."
                "give the most optimal list to the user prompt even if none match"

            )
        },
        {
            "role": "system",
            "content": f"The available songs are:\n{csv_text}"
        },
            {
            "role": "system",
"content": f"This is the user's preference history ({users[users_name]['preference']}), based on {users[users_name]['sent messages']} messages and the user's self-description. Some preferences may still be relevant, even if not explicitly mentioned. You may consider any part of this history if it seems to fit the current query — otherwise, disregard it."

        }
    ]





    # user query
    user_initial_query_message = {"role": "user", "content": user_query}
    history = []  # conversation history - in order to make conversation with the user and to keep updated memory about it

    # call gpt for rag
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=system_messages + [user_initial_query_message],
        temperature=0.2
    )

    reply = completion.choices[0].message.content.strip()




    #add gpt response to the history - session memory
    history.append({"role": "assistant", "content": reply})

    return reply , gpt_response_filtered, history, csv_text


def ask_loop(user_input, history , user_initial_query, users_name, user_query , gpt_response_filtered , csv_text):
# handelling future conversation with the user

    df = states.df

    yield " Think...."

    history.append({"role": "user", "content": user_input}) #add current user message to the session history
    
    # the user send another message. we will suggest him other songs based on what he asked or we will search again for relevant songs if we cant create relevant playlist out of the 34 he have in the csv

    system_messages = [
        {
            "role": "system",
            "content": (
                "You are a music recommendation assistant. Your task is to take a previously provided list of songs and **fix or improve it based on the user's most recent comment**. \n "
                "Always base your response primarily on the user's most recent message. You may also refer to previous conversation turns (user and assistant messages) if they help clarify intent or musical context — but only use them if relevant. \n"


                "If the user's request doesn't include a specific constraint we can't meet, return a playlist with songs from a variety of artists that align with the user’s current query.\n"
       
"Balance fit to the user's request and most recent comment with the song’s relative popularity in its time (Popularity_Today or Billboard_Weeks), but never prioritize popularity over relevance.\n "  
                "Do NOT invent songs or artists that are not in the list. Recommend strictly from the list. "
                "For each selected song, you must explain clearly and briefly (1 sentence) **why** it was chosen, based on its features in the data "
                "(such as lyrics summary if exists, mood, genre, danceability, tempo, today popularity, billboard weeks in each decade, release date, Instrumentalness, Positiveness, Acousticness, Speechiness, etc.), "
                "based on user's current request. approach their historical preferences only if it is relevant to their current query.\n"

                "You can use 'popularity today' and 'billboard weeks in the relevant decade' to justify popularity. "
                "You can use lyrics summary to explain songs with specific meanings. "
                "You can understand similarities and relations between songs by the features in the data.\n"

                "Focus your reasoning on the song’s relevance to the user's request or similar. "
                "If the user specifies a year, give them songs from that year.\n"
                "If the songs in the list are not a good match, or if the user changed his query theme such as the list is not fit and not reflect his demand "
                "then output a line that starts with '&&& ', which is a rephrased version of the user's request — include what the user asked for, what is available in the data, and what is missing. "
                "Don't explain or talk to the user — just rewrite the request clearly and optimally for internal search use. Do not ask the user to rephrase — do it yourself."
            )
        },
        {
            "role": "system",
            "content": f"The available songs and features are:\n{csv_text}"
        },
        {
            "role": "system",
"content": f"This is the user's preference history ({users[users_name]['preference']}), based on {users[users_name]['sent messages']} messages and the user's self-description. Some preferences may still be relevant, even if not explicitly mentioned. You may consider any part of this history if it seems to fit the current query — otherwise, disregard it."
        }
    ]




    # send  current and previous message by the user from the current session 
    user_initial_query_message = {"role": "user", "content": user_query}
    current_messages = system_messages + [user_initial_query_message] + history




    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=current_messages,
        temperature=0.2
    )

    reply = completion.choices[0].message.content.strip()

    


    history.append({"role": "assistant", "content": reply}) #add assistant message to the history memory session

    if "&&&" in reply:
        #this is happens when the user ask for songs / themes that are not in the list (e.g now he wants a playlist of a diffrent theme or with specific song that is not in the returened 34)

        yield "Ok I will find the best playlist for that" #telling the user that we will make broader search in the whole catalog

        index = reply.find("&&&")
        user_query_after = reply[index + 3:]  #in this case we asked gpt to write '&&&' and then to reformulate his query and we will start all the previous seach for songs again 

        user_pre_query = user_query

         #update the new query we will send to gpt what he wanted before and what the changes he asked in his last message
        user_query = user_query_after
        users[users_name]["preference"] = ask_gpt_history_update (users[users_name], users[users_name]["sent messages"] , history)
        users[users_name]["sent messages"] += 1

        user_query_based_history = (
            f"""
            User history and preferences: {users[users_name]['preference']}
            This is based on {users[users_name]['sent messages']} past messages. take it into account only if it is relevant to the current query, if not relevant at all ignore that.

            In the previous interaction, the user requested:
            "{user_pre_query}"

            The system responded with:
            {gpt_response_filtered}

            Now, the user's current request is:
            "{user_query}"

            Please take into account the user's evolving taste and try to improve the relevance of the response accordingly.
            """ 

            )

        #same process in ask_first_2 func

        try:
            gpt_response_filtered = ask_gpt_twice(user_query_based_history, column_summary_text)
            gpt_response_filtered_1, gpt_response_filtered_2 = gpt_response_filtered.split("&&&")
            print(gpt_response_filtered_1, gpt_response_filtered_2)

            dict1 = safe_eval_dynamic(gpt_response_filtered_1)
            dict2 = safe_eval_dynamic(gpt_response_filtered_2)

        except Exception as e:


            gpt_response_filtered = ask_gpt_twice(user_query_based_history, column_summary_text)
            gpt_response_filtered_1, gpt_response_filtered_2 = gpt_response_filtered.split("&&&")
            print(gpt_response_filtered_1, gpt_response_filtered_2)

            dict1 = safe_eval_dynamic(gpt_response_filtered_1)
            dict2 = safe_eval_dynamic(gpt_response_filtered_2)


        response_dict = {}

        for key in dict1.keys() | dict2.keys():
            val1 = dict1.get(key)
            val2 = dict2.get(key)

            if isinstance(val1, list) or isinstance(val2, list):
                list1 = val1 if isinstance(val1, list) else [val1] if val1 is not None else []
                list2 = val2 if isinstance(val2, list) else [val2] if val2 is not None else []
                response_dict[key] = list(set(list1 + list2))
            else:
                response_dict[key] = val2 if val2 is not None else val1

        excluded_keys = {"Similar Artists", "Similar Songs", "text"}
        valid_gpt_keys = [k for k in response_dict if k in df.columns and k not in excluded_keys]

        for d in (dict1, dict2):
            txt = d.get("text")
            if txt:
                semantic_col = f"semantic_score_{txt}"
                if semantic_col in df.columns:
                    valid_gpt_keys.append(semantic_col)

        merged_columns = list(dict.fromkeys(columns_to_keep + valid_gpt_keys))

        with ThreadPoolExecutor() as executor:
            playlist_description = executor.submit(ask_gpt_playlist_preview, gpt_response_filtered)
            future1 = executor.submit(parse_and_filter,  gpt_response_filtered_1, column_summary_text)
            future2 = executor.submit(parse_and_filter,  gpt_response_filtered_2, column_summary_text)
            response = playlist_description.result()  #wait untill the gpt will publish to the user short description about to next playlist
            yield f"🎧 {response}"

        df_filtered_1 = future1.result()
        df_filtered_2 = future2.result()
        df_filtered_1 = df_filtered_1.sort_values(by="score", ascending=False)
        df_filtered_1 = df_filtered_1.groupby("Artist(s)", group_keys=False).head(6)
        df_filtered_1 = df_filtered_1.sort_values(by="score", ascending=False).head(17)

        df_filtered_2_sorted = df_filtered_2.sort_values(by="score", ascending=False)

        df_filtered_2_top5_per_artist = df_filtered_2_sorted.groupby("Artist(s)", group_keys=False).head(6)

        df_filtered_2 = df_filtered_2_top5_per_artist.sort_values(by="score", ascending=False).head(17)


        cols1 = set(df_filtered_1.columns)
        cols2 = set(df_filtered_2.columns)
        all_columns = list(cols1 | cols2)

        df_filtered_1 = df_filtered_1.reindex(columns=all_columns)
        df_filtered_2 = df_filtered_2.reindex(columns=all_columns)

        df_filtered = pd.concat([df_filtered_1, df_filtered_2], ignore_index=True)

        #score and embedding cols are irrelevant for rag judgment (score beacuse we calculate 2 dictionaries in a diffrent way and features)
        safe_group_cols = [
        col for col in df_filtered.columns
        if col != "score"
        and not (
            pd.api.types.is_object_dtype(df_filtered[col]) or
            df_filtered[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any()
        )
    ]

        df_filtered = df_filtered.drop_duplicates(subset=safe_group_cols)


        csv_text = df_filtered[merged_columns] \
            .rename(columns={
                'appear_in_weekly_bilabord_in_decade_1960': 'billboard_1960s_weeks',
                'appear_in_weekly_bilabord_in_decade_1970': 'billboard_1970s_weeks',
                'appear_in_weekly_bilabord_in_decade_1980': 'billboard_1980s_weeks',
                'appear_in_weekly_bilabord_in_decade_1990': 'billboard_1990s_weeks',
                'appear_in_weekly_bilabord_in_decade_2000': 'billboard_2000s_weeks',
                'appear_in_weekly_bilabord_in_decade_2010': 'billboard_2010s_weeks',
                'appear_in_weekly_bilabord_in_decade_2020': 'billboard_2020s_weeks',
                'misral_summary': 'lyrics_summary',
            }).to_csv(index=False)
    

        # ask gpt to build a playlist and explain why each ong is now relevant
        system_messages = [
            {
                "role": "system",
                "content": (
                    "You are a music recommendation assistant. "
                    "Your job is to create a playlist of 5 songs based ONLY on the list provided below.\n "
                    "If the user's request doesn't include a specific constraint we can't meet, return a playlist with songs from a variety of artists that alighn the user current query\n."
                    "Read the last request the user made according to the list you gave him in the previous turn and explain why the current list fit his needs \n"

"Balance fit to the user's request with the song’s relative popularity in its time (Popularity_Today or Billboard_Weeks), but never prioritize popularity over relevance."
                    "For each selected song, you must explain clearly and briefly (1 sentence) **why** it was chosen, based on its features in the data (such as lyrics summary if exist, mood, genre, danceability, tempo, popularity , release date, bilabord weeks in each decade, etc.), based on user's current request. approach their historical preferences only if it is relevant to their current query. \n"
                    "you can use popularity today and bilabords week in the relevant decade to explain and give songs with relevant popularity. you can use lyrics summary to explain and give songs with specific meanings. \n"
                                        "you can use popularity today and bilabords week in the relevant decade to explain and give songs with relevant popularity. you can use lyrics summary to explain and give songs with specific meanings. \n"
                                        "you can understand similarities and relations between songs by the  features in the data \n"
            

                    "Focus your reasoning on the song’s relevance to the user's request"
                    "Do NOT invent songs or artists that not it the list. Recommend strictly from the list."
                    "give the most optimal list to the user prompt even if none match"
                )
            } ,
            {
                "role": "system",
                "content": f"The available songs are:\n{csv_text}"
            },
                {
                "role": "system",
"content": f"This is the user's preference history ({users[users_name]['preference']}), based on {users[users_name]['sent messages']} messages and the user's self-description. Some preferences may still be relevant, even if not explicitly mentioned. You may consider any part of this history if it seems to fit the current query — otherwise, disregard it."

            }
            ]
        
        user_initial_query_message = {"role": "user", "content": user_query}

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages= system_messages + history + [user_initial_query_message],
            temperature=0.2
        )

        reply = completion.choices[0].message.content.strip()
        history = []  #initialize the session history due to massive changes by the user. we updated his whole profile based on the previous messages exchange



        print(f"\n Music Bot:🎵 Your playlist\n{reply}\n")

        
        history.append({"role": "assistant", "content": reply})

        return reply , user_input, history , user_initial_query,  user_query, gpt_response_filtered , csv_text

        
    else: #if we got to here the system can fix the playlist/give an explanation such as it is meet the user damands in the last messafe


        return reply , user_input, history , user_initial_query,  user_query , gpt_response_filtered, csv_text

