import cv2
import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

THRESHOLD = 0.35

# Load FAST DB
with open("deep_fast.pkl", "rb") as f:
    db = pickle.load(f)

cap = cv2.VideoCapture(0)
print("Fast Face Recognition ON (Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 Resize for speed
    small = cv2.resize(frame, (640, 480))

    try:
        results = DeepFace.represent(
            img_path=small,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=False,
            align=True
        )

        for res in results:
            if "embedding" not in res:
                continue

            current_emb = np.array(res["embedding"])

            best_score = 1.0
            best_person = "Unknown"

            # 🔥 ONE comparison per person
            for person, avg_emb in db.items():
                dist = cosine(current_emb, avg_emb)
                if dist < best_score:
                    best_score = dist
                    best_person = person

            if best_score < THRESHOLD:
                label = f"{best_person} ({(1-best_score)*100:.1f}%)"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            area = res["facial_area"]
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]

            cv2.rectangle(small, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                small, label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Fast Face ID", small)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
