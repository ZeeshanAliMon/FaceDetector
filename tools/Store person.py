import cv2
import pickle
from deepface import DeepFace
import time

name = "Name"   # change for each friend
SAVE_COUNT = 10

embeddings = []

cap = cv2.VideoCapture(0)
print(f"Capturing face for {name}")
print("Look straight, slight left/right, press Q to stop early")

while len(embeddings) < SAVE_COUNT:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True,
            align=True
        )

        emb = results[0]["embedding"]
        embeddings.append(emb)
        print(f"Captured {len(embeddings)}/{SAVE_COUNT}")
        time.sleep(0.5)  # small delay

    except:
        pass

    cv2.imshow("Capture Face", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save to DB
with open("deep.pkl", "rb") as f:
    db = pickle.load(f)

db[name] = embeddings

with open("deep.pkl", "wb") as f:
    pickle.dump(db, f)

print(f"{name} added successfully ✔")
