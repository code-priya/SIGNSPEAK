import asyncio, json, cv2, numpy as np, websockets, tensorflow as tf
from collections import deque

model = tf.keras.models.load_model('models/model_weights.h5', compile=False)
actions = np.array(['hello','thanks','iloveyou'])

sequence = deque(maxlen=30)
sentence = []

async def handle(ws):
    async for message in ws:
        nparr = np.frombuffer(message, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: continue

        # Dummy prediction (replace with real logic)
        res = np.random.rand(3)
        idx = np.argmax(res)

        if res[idx] > 0.7:
            gesture = actions[idx]
            if not sentence or sentence[-1] != gesture:
                sentence.append(gesture)

        await ws.send(json.dumps({
            "prediction": actions[idx],
            "confidence": float(res[idx]),
            "sentence": sentence[-5:],
            "confidence_scores": res.tolist()
        }))

async def main():
    async with websockets.serve(handle, "0.0.0.0", 8765):
        await asyncio.Future()

asyncio.run(main())
