import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime
import yagmail
from twilio.rest import Client

# Email & SMS config
EMAIL_USER = "gayathrivalluri07@gmail.com"
EMAIL_PASS = "your_app_password_here"  # Replace with your real app password
TO_EMAIL = "gayathrivalluri07@gmail.com"

TWILIO_SID = 'your_twilio_sid_here'
TWILIO_AUTH_TOKEN = 'your_twilio_token_here'
FROM_NUMBER = '+1234567890'   # Twilio verified number
TO_NUMBER = '+919876543210'   # Your personal phone number

# Load known faces
path = 'known_faces'
images = []
names = []
for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)
    names.append(os.path.splitext(file)[0])

# Encode faces
def encode_faces(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings.append(face_recognition.face_encodings(img)[0])
    return encodings

known_encodings = encode_faces(images)

# Initialize attendance log
def mark_attendance(name):
    df = pd.read_csv('attendance.csv')
    now = datetime.now()
    time_string = now.strftime('%H:%M:%S')
    date_string = now.strftime('%Y-%m-%d')

    # Prevent proxy by limiting 30-min re-entry
    recent_entries = df[(df['Name'] == name) & (df['Date'] == date_string)]
    if not recent_entries.empty:
        last_time = datetime.strptime(recent_entries.iloc[-1]['Time'], '%H:%M:%S')
        if (now - datetime.combine(now.date(), last_time.time())).seconds < 1800:
            print(f"⛔ Proxy attempt detected for {name}")
            return

    df.loc[len(df)] = [name, time_string, date_string]
    df.to_csv('attendance.csv', index=False)
    print(f"✅ {name} marked present at {time_string}")

# Send alerts
def send_alert(image):
    cv2.imwrite('intruder.jpg', image)

    # Email Alert
    yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
    yag.send(
        to=TO_EMAIL,
        subject="Unknown Face Detected!",
        contents="An unrecognized person was detected at the attendance system.",
        attachments='intruder.jpg'
    )

    # SMS Alert
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body="🚨 Unknown person detected in the Smart Attendance System!",
        from_=FROM_NUMBER,
        to=TO_NUMBER
    )

# Start camera
cap = cv2.VideoCapture(0)

if not os.path.exists('attendance.csv'):
    pd.DataFrame(columns=['Name', 'Time', 'Date']).to_csv('attendance.csv', index=False)

print("📷 Starting face recognition system...")
while True:
    success, frame = cap.read()
    if not success:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for encode_face, loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encode_face)
        face_distances = face_recognition.face_distance(known_encodings, encode_face)
        match_index = np.argmin(face_distances)

        if matches[match_index]:
            name = names[match_index].capitalize()
            mark_attendance(name)
            y1, x2, y2, x1 = [v*4 for v in loc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            y1, x2, y2, x1 = [v*4 for v in loc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            send_alert(frame)

    cv2.imshow('Smart Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
