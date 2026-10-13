"""Indian Sign Language input.

landmarks.py  : MediaPipe Holistic -> per-frame hand/pose keypoints (ready-made library)
classifier.py : keypoint sequence -> sign gloss (we train it, see training/sign/)
The pipeline then turns glosses into a sentence with the SLM (Pipeline.ask_signs).
"""
