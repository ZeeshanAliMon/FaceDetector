import pickle
import numpy as np

with open("./deep.pkl", "rb") as f:
    db = pickle.load(f)

fast_db = {}
for person, embs in db.items():
    fast_db[person] = np.mean(np.array(embs), axis=0)

with open("deep_fast.pkl", "wb") as f:
    pickle.dump(fast_db, f)

print("Fast DB updated ✔")
