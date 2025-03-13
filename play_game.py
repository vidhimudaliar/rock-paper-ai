# import cv2
# import numpy as np
# import tensorflow as tf
# import random

# # Load trained model
# model = tf.keras.models.load_model("rps_model.h5")
# labels = ["rock", "paper", "scissors"]

# # Game rules
# def determine_winner(player, ai):
#     if player == ai:
#         return "It's a Tie!"
#     elif (player == "rock" and ai == "scissors") or \
#          (player == "paper" and ai == "rock") or \
#          (player == "scissors" and ai == "paper"):
#         return "You Win!"
#     else:
#         return "AI Wins!"

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Preprocess frame for model
#     resized_frame = cv2.resize(frame, (100, 100)) / 255.0
#     input_frame = np.expand_dims(resized_frame, axis=0)

#     # Predict gesture
#     prediction = model.predict(input_frame)
#     player_choice = labels[np.argmax(prediction)]

#     # AI makes a random move
#     ai_choice = random.choice(labels)

#     # Determine game result
#     result = determine_winner(player_choice, ai_choice)

#     # Display game result
#     cv2.putText(frame, f"You: {player_choice}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(frame, f"AI: {ai_choice}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     cv2.putText(frame, result, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("Rock, Paper, Scissors - AI", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

# from keras.models import load_model
# import cv2
# import numpy as np
# from random import choice

# REV_CLASS_MAP = {
#     0: "rock",
#     1: "paper",
#     2: "scissors",
#     3: "none"
# }


# def mapper(val):
#     return REV_CLASS_MAP[val]


# def calculate_winner(move1, move2):
#     if move1 == move2:
#         return "Tie"

#     if move1 == "rock":
#         if move2 == "scissors":
#             return "User"
#         if move2 == "paper":
#             return "Computer"

#     if move1 == "paper":
#         if move2 == "rock":
#             return "User"
#         if move2 == "scissors":
#             return "Computer"

#     if move1 == "scissors":
#         if move2 == "paper":
#             return "User"
#         if move2 == "rock":
#             return "Computer"


# model = load_model("rock-paper-scissors-model.h5")

# cap = cv2.VideoCapture(0)

# prev_move = None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     # rectangle for user to play
#     cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
#     # rectangle for computer to play
#     cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

#     # extract the region of image within the user rectangle
#     roi = frame[100:500, 100:500]
#     img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (227, 227))

#     # predict the move made
#     pred = model.predict(np.array([img]))
#     move_code = np.argmax(pred[0])
#     user_move_name = mapper(move_code)

#     # predict the winner (human vs computer)
#     if prev_move != user_move_name:
#         if user_move_name != "none":
#             computer_move_name = choice(['rock', 'paper', 'scissors'])
#             winner = calculate_winner(user_move_name, computer_move_name)
#         else:
#             computer_move_name = "none"
#             winner = "Waiting..."
#     prev_move = user_move_name

#     # display the information
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(frame, "Your Move: " + user_move_name,
#                 (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(frame, "Computer's Move: " + computer_move_name,
#                 (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(frame, "Winner: " + winner,
#                 (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

#     if computer_move_name != "none":
#         icon = cv2.imread(
#             "images/{}.png".format(computer_move_name))
#         icon = cv2.resize(icon, (400, 400))
#         frame[100:500, 800:1200] = icon

#     cv2.imshow("Rock Paper Scissors", frame)

#     k = cv2.waitKey(10)
#     if k == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

from keras.models import load_model
import cv2
import numpy as np

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}

def mapper(val):
    return REV_CLASS_MAP[val]

def get_winning_move(user_move):
    # Return the move that beats the user's move
    if user_move == "rock":
        return "paper"
    elif user_move == "paper":
        return "scissors"
    elif user_move == "scissors":
        return "rock"
    else:
        return "none"  # If user shows none, computer shows none

def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"
    
    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"
    
    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"
    
    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"

# Load the pre-trained model
model = load_model("rock-paper-scissors-model.h5")

# Initialize webcam
cap = cv2.VideoCapture(0)
prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    
    # Rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)
    
    # Extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))
    
    # Predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)
    
    # Debug confidence scores
    confidence_rock = pred[0][0]
    confidence_paper = pred[0][1]
    confidence_scissors = pred[0][2]
    confidence_none = pred[0][3]
    
    # Print confidence scores to console for debugging
    print(f"Confidence scores: Rock: {confidence_rock:.4f}, Paper: {confidence_paper:.4f}, " 
          f"Scissors: {confidence_scissors:.4f}, None: {confidence_none:.4f}")
    
    # Choose the winning move for computer
    if prev_move != user_move_name:
        if user_move_name != "none":
            # Computer cheats by looking at user's move and choosing a winning move
            computer_move_name = get_winning_move(user_move_name)
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    
    prev_move = user_move_name
    
    # Display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display winner with blue for Computer wins
    if winner == "Computer":
        winner_color = (255, 0, 0)  # Blue for computer wins
    elif winner == "User":
        winner_color = (0, 255, 0)  # Green (should never happen with this code)
    else:
        winner_color = (0, 0, 255)  # Red for ties or waiting
        
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, winner_color, 4, cv2.LINE_AA)
    
    # Display computer's move image
    if computer_move_name != "none":
        try:
            icon = cv2.imread("images/{}.png".format(computer_move_name))
            icon = cv2.resize(icon, (400, 400))
            frame[100:500, 800:1200] = icon
        except Exception as e:
            print(f"Error loading image: {e}")
    
    # Show the frame
    cv2.imshow("Rock Paper Scissors", frame)
    
    # Check for quit key
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()