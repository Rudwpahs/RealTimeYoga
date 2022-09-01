from gtts import gTTS
import os
import time
import playsound
#greetings
def start():
   tts = gTTS(text="잘하셨어요!10초동안 자세를 유지하셨습니다.", lang='ko' ,slow=0)
   tts.save('audio.mp3')

   from playsound import playsound
   playsound('audio.mp3')
   os.remove('audio.mp3')

start()
