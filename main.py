import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image 
import streamlit as st

st.set_page_config(layout="wide")



col1,col2 = st.columns([2,1])

with col1:
    run = st.checkbox('Run',value= True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.subheader("")








genai.configure(api_key="AIzaSyB3lADfmOjBRIl7TWzSDqxg4AaWQpCNu8s")
model = genai.GenerativeModel("gemini-2.0-flash")





cap = cv2.VideoCapture(0)
# cls

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList = hand1["lmList"]  # List of 21 landmarks for the first hand
        
        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand1)
        print(fingers  )
        return fingers,lmList
    else:
        return None,None




def draw(info,prev_pos,canvas):
    fingers,lmlist = info
    curr_pos = None

    if fingers == [0,1,0,0,0]:
        curr_pos = lmlist[8][0:2]
        if prev_pos is None: prev_pos = curr_pos
        cv2.line(canvas,curr_pos,prev_pos,(0, 0, 255),4)

    elif fingers == [0,0,0,0,0]:
        canvas = np.zeros_like(img)


    return curr_pos,canvas



def sendToAI(model,canvas,fingers):
    if fingers == [1,1,1,1,0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve the problem and give only final answer ",pil_image])
        return response.text




    

prev_pos = None
canvas = None
image_combine = None

output_text = ""

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.flip(img,1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img=img)

    if info:
        fingers,lmlist = info
        
        prev_pos,canvas = draw(info=info,prev_pos=prev_pos,canvas=canvas)

        output_text = sendToAI(model,canvas,fingers)
    image_combine = cv2.addWeighted(img,0.7,canvas,0.15,0)
    FRAME_WINDOW.image(image_combine,channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    # Display the image in a window
    # cv2.imshow("Image", img)
    # cv2.imshow("canvas", canvas)
    cv2.imshow("combine", image_combine)
    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)


