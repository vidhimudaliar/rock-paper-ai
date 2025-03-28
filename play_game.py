from keras.models import load_model
import cv2
import numpy as np

# Reverse class mapping
REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}

def mapper(val):
    return REV_CLASS_MAP[val]

def get_winning_move(user_move):
    if user_move == "rock":
        return "paper"
    elif user_move == "paper":
        return "scissors"
    elif user_move == "scissors":
        return "rock"
    return "none"

def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"
    if move1 == "rock" and move2 == "scissors":
        return "User"
    if move1 == "paper" and move2 == "rock":
        return "User"
    if move1 == "scissors" and move2 == "paper":
        return "User"
    return "Computer"

# Load model
model = load_model("rock-paper-scissors-model.h5")

# Webcam initialization
cap = cv2.VideoCapture(0)
prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Draw rectangles
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    # Extract ROI
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))
    img = np.array(img, dtype="float32") / 255.0  # Normalize

    # Predict move
    pred = model.predict(np.expand_dims(img, axis=0))[0]
    move_code = np.argmax(pred)
    user_move_name = mapper(move_code)

    # Confidence thresholding
    if max(pred) < 0.6:  # If confidence is too low, assume "none"
        user_move_name = "none"

    print(f"Confidence scores: {pred}")

    # Computer's move
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = get_winning_move(user_move_name)
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."

    prev_move = user_move_name

    # Display text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Your Move: {user_move_name}", (50, 50), font, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"Computer: {computer_move_name}", (750, 50), font, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"Winner: {winner}", (400, 600), font, 2, (0, 255, 255), 4)

    # Display computer move image
    if computer_move_name != "none":
        try:
            icon = cv2.imread(f"images/{computer_move_name}.png")
            icon = cv2.resize(icon, (400, 400))
            frame[100:500, 800:1200] = icon
        except:
            print(f"Error loading image for {computer_move_name}")

    # Show frame
    cv2.imshow("Rock Paper Scissors", frame)

    # Exit
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
