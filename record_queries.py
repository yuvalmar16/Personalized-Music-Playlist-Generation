import speech_recognition as sr  

# Allow the user to record instead of typing
def transcribe_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    recognizer.pause_threshold = 2  # automatically pauses after 2 seconds of silence

    with mic as source:
        print(" Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    print(" Transcribing...")
    try:
        return recognizer.recognize_google(audio, language="he-IL")  # supports both Hebrew and English
    except Exception as e:
        return f" I couldn’t understand you: {e}"  # user didn’t pronounce clearly or recognition failed
