from deepface import DeepFace
import pickle
name = "Name"
images = []
with open("deep.pkl", "rb") as f:
    db = pickle.load(f)
# db = {}
# 2. Add new person
db[name] = []
print(f"Processing {name}...")

for img_path in images:
    try:
        results = DeepFace.represent(
    img_path=img_path,
    model_name="Facenet512",
    detector_backend="opencv",
    enforce_detection=True,
    align=True
)


        embedding = results[0]["embedding"]
        db[name].append(embedding)
        print(f"  Success: {img_path}")
    except Exception as e:
        print(f"  Failed {img_path}: {e}")

# 3. Save updated database
with open("deep.pkl", "wb") as f:
    pickle.dump(db, f)

print("\nNew person added successfully!")