import gradio as gr
import inspect
# from main import users
from rag import ask_first, ask_loop
import states
from openai_client import *
from states import users, df, column_summary_text, Artists_Popularity_df

def handle_message(message, state):
    df = states.df
    
    users_name, stage,  user_query, user_initial_query = state


    # dict_indexes help us to include global variables about user desires in the session - it his better use this way because return a lot of words in gardio take much more time due to thie servers
    dict_indexes = {
    "users_name": 0,
    "stage": 1,
    "user_query": 2,
    "user_initial_query": 3,

}
    conversation = ""

    # user's identication
    if stage == 0:
        users_name = message
        if users_name in users:
            bot_msg = f"Welcome back {users_name}! what would you like to listen to today?"
            conversation += f"You: {message}\nBot: {bot_msg}\n"
            state[dict_indexes["users_name"]] = users_name
            state[dict_indexes["stage"]] = 2
            yield conversation, state, ""
            return
        
        # if this is the first time the user is in the app we will ask him to describe his music perferences
        users[users_name] = {"sent messages": 0, "preference": "Unknown"}
        bot_msg = "Before we start, please write your own music profile (geners/artists/songs/eras for each mood) that you like to listen to"
        conversation += f"You: {message}\nBot: {bot_msg}\n"
        state[dict_indexes["users_name"]] = users_name
        state[dict_indexes["stage"]] = 1
        yield conversation, state, ""
        return

    # describe your music profile if you are new to the app - we got the user description here
    if stage == 1:
        users[users_name]["preference"] = message
        bot_msg = "what would you like to listen to today?"
        conversation += f"You: {message}\nBot: {bot_msg}\n"
        state[dict_indexes["stage"]] = 2
        yield conversation, state, ""
        return







    # send the first message
    if stage == 2:
        conversation += f"You: {message}\n"

        # update state and initial state variables
        state[dict_indexes["user_query"]] = str(message)
        state[dict_indexes["user_initial_query"]] = str(message)

        users[users_name]["gpt_response_filtered"] = ""
        users[users_name]["history"] = {}
        users[users_name]["csv_text"] = ""


        gpt_response_filtered = users[users_name]["gpt_response_filtered"]
        history = users[users_name]["history"]
        csv_text = users[users_name]["csv_text"]

        # send to the user that we are working on that
        yield conversation + "Bot: 🤖 Looking for the best playlist...\n", state, ""

         # call ask first to return playlist

        result = ask_first(message, users_name)

        reply = gpt_response_filtered = history = csv_text = None

        if inspect.isgenerator(result):
            try:
                while True: #we will got a message with short description about the playlist and after that the playlist with explaination on every returned song
                    partial = next(result)
                    if isinstance(partial, str):
                        yield conversation + f"Bot: {partial}\n", state, ""
            except StopIteration as e:
                if e.value is not None:
                    reply, gpt_response_filtered, history, csv_text = e.value
                else:
                    raise RuntimeError("ask_first did not return final values properly")
        else:
            reply, gpt_response_filtered, history, csv_text = result

        # update state after we have a playlist
        users[users_name]["gpt_response_filtered"] = gpt_response_filtered 
        users[users_name]["history"] = history 
        users[users_name]["csv_text"] = csv_text 
        state[dict_indexes["stage"]] = 3

        bot_msg = f" Music Bot: Your playlist is ready!\n{reply}"
        conversation += f"Bot: {bot_msg}\n"
        yield conversation, state, ""
        return


    # conversation with the user - every message after the first message in the session

    if stage == 3:
        users[users_name]["sent messages"] += 1
        gpt_response_filtered = users[users_name]["gpt_response_filtered"]
        history = users[users_name]["history"]
        csv_text = users[users_name]["csv_text"]

        if message.strip().lower() in ["exit", "quit", "סיים"]: #want to get out of the app 
            bot_msg = " Music Bot: Enjoy your playlist! Session ended."
            conversation += f"You: {message}\nBot: {bot_msg}\n"
            state[dict_indexes["stage"]] = 2

            # clear the states variables
            state[dict_indexes["user_query"]] = ""  # last query  (the last time he wanted in the conversation to completely change the playlist)
            state[dict_indexes["user_initial_query"]] = ""  # user first message (what the conversation start about)

            users[users_name]["gpt_response_filtered"] = " "  # feature and variables we infer from his prompt
            users[users_name]["history"] = {}  # session history
            users[users_name]["csv_text"] = None  # csv with candidate 34 songs



            users[users_name]["preference"] = ask_gpt_history_update (users[users_name], users[users_name]["sent messages"] , history) #we will update his music profile before he leaving


            yield conversation, state, ""
            return
        

        # if we got to here the user still want to keep to change the playlist and we will use ask_loop func to update the playlist

        conversation += f"You: {message}\n"
        user_input = message

        result = ask_loop(user_input, history, user_initial_query, users_name, user_query, gpt_response_filtered, csv_text)

        reply = user_query = user_input = user_initial_query = gpt_response_filtered = history = csv_text = None

        if inspect.isgenerator(result):
            try:
                while True: #similar to stage 2, we will tell the user that we are working about it. if he want another playlist we will tell him that we will create another playlist and its description. at the end we will return a playlist
                    partial = next(result)
                    if isinstance(partial, str):
                        yield conversation + f"Bot: {partial}\n", state, ""
            except StopIteration as e:
                if e.value is not None:
                    reply , user_input, history , user_initial_query, user_query,gpt_response_filtered, csv_text = e.value
                else:
                    raise RuntimeError("ask_loop did not return final values properly")
        else:
             reply , user_input, history , user_initial_query, user_query,gpt_response_filtered, csv_text = result

        if reply is None:
            raise RuntimeError("ask_loop did not return final values properly")

        # state update
        state[dict_indexes["user_query"]] = user_query
        state[dict_indexes["user_initial_query"]] = user_initial_query

        users[users_name]["gpt_response_filtered"] = gpt_response_filtered 
        users[users_name]["history"] = history 
        users[users_name]["csv_text"] = csv_text 

        bot_msg = f"🎵 Music Bot: {reply}"
        conversation += f"Bot: {bot_msg}\n"
        yield conversation, state, ""
        return