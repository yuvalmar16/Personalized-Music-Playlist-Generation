import os
# from dotenv import load_dotenv
from openai import OpenAI

# load_dotenv()

# openai_api_key = os.getenv("OPENAI_KEY")
openai_api_key = "INSERT YOUR API KEY HERE"

def ask_gpt(user_query, column_summary_text): #if we had error in one of the 2 dictionaries we will ask gpt to fix him
    system_prompt = (
        "You are a music recommendation assistant.\n"
        "Given a user request, return only a valid Python dictionary.\n"
        "Use the provided column descriptions below to infer appropriate keys and values.\n"
        "Instructions:\n"
        "• READ ALL THE COLUMNS IN THE PROMT\n"
        "• Try to use multiple features\n"
        "• you may use numerical features for situations/feelings (e.g., 'Energy', 'Danceability', 'Positiveness','Speechiness', 'Liveness', 'Acousticness', 'Tempo; , 'Instrumentalness') due to request and common sense , return a number or a range (e.g., 50 or '>=70').\n"
        "• For binary features (e.g., Good for Running, Good for Party,Good for Work/Study, Good for Relaxation/Meditation', 'Good for Social Gatherings' and for Morning Routine, return 1 if relevant, otherwise omit.\n"
        "• For situations/feelings you may use string/categorical features (e.g., Genre, and emotion), return a string value if appropriate, if more than 1 values in each column put them in list.\n"
        "• emotion values are 'sadness', 'joy', 'love', 'surprise', 'anger', 'fear', 'angry','True', 'thirst', 'confusion', 'pink', 'interest', 'Love'\n"
        "• For datetime features (e.g., Release Date), use only if relevant, return a string or date range.\n"
                "• You may use the `text` column to understand the semantic meaning of each song such as its theme, "
    "emotional tone, or lyrical content — in order to better match user queries like 'songs about heartbreak' "
    "or 'uplifting songs'.\n"
        "• Do not include columns that are irrelevant to the user request.\n"
        "• you can use appear_in_weekly_bilabord_in_decade_(relevant decade such as 1990 or 2010) column for poplarity in each deacade if it was relevant or for music style relevancy (e.g '<=2', 1). for today's relevancy you may use Popularity_Today (e.g '>=70').\n"
        "• if someone want only spesific artist you may use Artist(s) column.\n"
        # "• if someone want similar_artist you may use Similar Artists column. if more than 1 similars put them in list.\n"
        # "• if someone want similar_songs you may use Similar Songs column if more than 1 similars put them in list\n"
        "• you can add the artist full name and the song to artist(s) and song columns\n"
        "• allways give similar well-knowns artists that fit for the user demand" #and songs
        "• you may use also Geners according to the similar artsits " #or songs
        " Return the output as a raw Python dictionary only — **do not wrap it in a code block**, do not add any text, explanation, or formatting around it."
    ) 

    user_prompt = (
        f"User request: {user_query}\n\n"
        f"Those are the only Columns - descriptions:\n{column_summary_text}\n\n"
        f"Return only a Python dictionary with inferred column values."
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )

    reply = completion.choices[0].message.content.strip()
    return reply


#Ask gpt to return 2 dictionaries with the relevant feature each of them reflect one of the 2 main possible relevant features set the will be relevant for him
def ask_gpt_twice(user_query, column_summary_text):
    system_prompt = (
        
        "You are a music recommendation assistant.\n"
        "Given a user current request and their historical preferences (approach it only if it is relevant to the current request. *if not relevant ignore* his history and take into account only the current query), return only 2 diffrent valid Python dictionary separated by &&&\n"
        "Use the provided column descriptions below to infer appropriate keys and values.\n"
        "Instructions: for each dictionary\n"
        "• READ ALL THE COLUMNS IN THE PROMT\n"
        "• Try to use multiple features\n"
        "• you may use numerical features for situations/feelings (e.g., 'Energy', 'Danceability', 'Positiveness','Speechiness', 'Liveness', 'Acousticness', 'Tempo; , 'Instrumentalness') due to request and common sense , return a number (e.g., 50 or '>=70') which is the preferred optin. tou can also use range if relevant (e.g 100-120) with no less than 20 units.\n"
        "• For binary features (e.g., Good for Running, Good for Party,Good for Work/Study, Good for Relaxation/Meditation', 'Good for Social Gatherings' and for Morning Routine, return 1 if relevant, otherwise omit. \n"
        "• For situations/feelings you may use string/categorical features (e.g., Genre, and emotion), return a string value if appropriate, if more than 1 values in each column put them in list and also return 1 or more relevant artists or songs for the emotion/genre.\n"
        "• emotion values are 'sadness', 'joy', 'love', 'surprise', 'anger', 'fear', 'angry','True', 'thirst', 'confusion', 'pink', 'interest', 'Love'\n"
        "• For datetime features (e.g., Release Date), use only if relevant, return a string or date range.\n"
        "• Use the `text` column to understand the semantic meaning of each song such as its theme, emotional tone, or lyrical content — in order to better match user queries like 'songs about heartbreak'. • When analyzing the text column, extract and rely on only the 2–3 most semantically central words or phrases (e.g., 'heartbreak', 'self-love', 'new beginning') that represent the main emotional or thematic meaning of the song."
    "or 'uplifting songs'.\n"
        "• Do not include columns that are irrelevant to the user request.\n"
        "• you can use appear_in_weekly_bilabord_in_decade_(relevant decade such as 1990 or 2010) column for poplarity in each deacade if it was relevant or for music style relevancy (e.g '<=2', 1). for today's relevancy you may use Popularity_Today (e.g '>=70').\n"
        "• if someone want only spesific artist you may use Artist(s) column.\n"
        "• if someone want similar_artist you may use Similar Artists column. if more than 1 similars put them in list.\n"
        "• if someone want similar_songs you may use Similar Songs column if more than 1 similars put them in list\n"
        "• you can add the artist full name and the song to artist(s) and song columns\n"
        "• allways give similar well-knowns artist and similar songs that fit for the user demands" 
        "• When providing similar songs (titles only), write **only the song name** (without the performer/artist name), e.g., ['Titanium'] \n"
        "• you may use also Geners according to the similar artsits or songs" 
        "• if the user want a specific song write it in the value of the key 'song' in one of the returned dictionaries.\n " 
        "• if the user is looking for variations such as 'Remaster', 'Remix', 'Cover' or similar, check the 'Similar Songs' column without hyphen, e.g., 'Remaster' and also the song itself the song column only by title e.g ['shape of you'] \n."
        "  If the user requests multiple distinct musical styles (e.g., upbeat and calm), return a different style in each dictionary for each style, with its own relevant features, artists. "


        " Return the output as a raw Python dictionary only — **do not wrap it in a code block**, do not add any text, explanation, or formatting around it."
    )
    user_prompt = (
        f"User request: {user_query}\n\n"
        f"Those are the only Columns - descriptions:\n{column_summary_text}\n\n"
        f"Return only a Python dictionary with inferred column values."
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )

    reply = completion.choices[0].message.content.strip()
    return reply


#update the user perferences after every dialouge in order to build his music-profile for future conversations
def ask_gpt_history_update(history_preferences, num_messages_in_history, last_conversation):
    system_messages = [
        {
            "role": "system",
            "content": (
                f"The following is a summary of the user's musical preferences, based on ~{num_messages_in_history} past messages:\n"
                f"{history_preferences}\n\n"
                f"Now, update the profile based only on the recent conversation below. Add relevant preferences if clearly supported, but do not remove or narrow down existing preferences unless the user explicitly rejected them. Always keep in mind that the user may have a wide and diverse taste in music, and past preferences remain valid unless contradicted."

                f"Do not change or remove any preference unless it was clearly addressed in this conversation. "
                f"If the user clearly says or strongly implies that they do not like or never want a particular genre, artist, or style and etc — treat that as a permanent dislike and avoid recommending it in the future. "
                f"If they say they are not interested in something 'right now', or express temporary disinterest, treat that as a contextual (not permanent) preference. "
                f"If the user expresses or clearly implies preferences for song characteristics (e.g., energy, tempo, mood), you may include them, but consider that these may be context-specific (e.g., calm for mornings, energetic for workouts)."
                f"The user may have a wide and diverse taste in music, which can vary depending on context, mood, or situation."



                f"Use judgment to distinguish between permanent and temporary preferences. "
                f"For example, a user might enjoy a song for workouts but not mornings — that doesn’t mean they dislike it. "
f"Present the updated preferences as a clear bullet-point list, grouped by category (e.g., likes, dislikes, and context-specific preferences). Reflect the user's broader musical taste — include genres or features they consistently enjoy, even if not mentioned in this specific conversation, unless contradicted. Add new preferences from this conversation only if clearly supported. Limit to 4 concise sentences total.\n"

                f"Update only what’s directly supported by this conversation. Respond in up to 5-6 concise sentences.\n\n"
                f"Recent conversation:\n{last_conversation}"
            )
        }

    ]

    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=system_messages ,
        temperature=0.2
    )
        
    
    reply = completion.choices[0].message.content.strip()

    return reply


#expin to the user wht wi be in the next playlist before publish it (not make him wait a long without interactions from the app)
def ask_gpt_playlist_preview(playlist_description):
    system_messages = [
        {
            "role": "system",
            "content": (
                f"You are a music recommendation system. "
                f"Describe to the user in one sentence the playlist you are going to build for them, "
                f"based on the following description: {playlist_description}"
            )
        }
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=system_messages,
        temperature=0.2
    )

    reply = completion.choices[0].message.content.strip()
    return reply