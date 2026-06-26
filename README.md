# RealTimeYoga

This program uses mediapipe to inform you of the accuracy of yoga postures.
I hope that people who lack exercise in the COVID era can use it.

![Image 20220906 101019](https://user-images.githubusercontent.com/296403/188530081-c2b957d2-734c-46d9-b615-eca190976835.png)

# Requirements

- python 3.9+
- webcam

# Download

- Download packaged builds from the [Releases](https://github.com/Rudwpahs/RealTimeYoga/releases) page.
- Install dependencies with `pip install -r requirements.txt`.

# How to use

- Run the program.
- Watch the display in front of the webcam.
- Follow the posture that shows at the display.
- If the posture of each part is incorrect, a red circle will appear. Adjust your posture so that it becomes blue.
- If you maintain the posture for 10 seconds, you are successful.
- Without a webcam, run `python main.py --image easy.png` to test with the demo image.
