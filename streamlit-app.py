import streamlit as st
import cv2
import numpy as np
import time
from tensorflow.keras.models import model_from_yaml, load_model
from tensorflow.keras.preprocessing.image import img_to_array
import yaml
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ------------------- Patch PyYAML for compatibility -------------------
_yaml_load_orig = yaml.load
def _yaml_load_patch(*args, **kwargs):
    if 'Loader' not in kwargs:
        kwargs['Loader'] = yaml.FullLoader
    return _yaml_load_orig(*args, **kwargs)
yaml.load = _yaml_load_patch

# ------------------- Load Anti-Spoofing Model -------------------
with open("trained_model/RGB_rPPG_merge_softmax_.yaml", 'r') as yaml_file:
    loaded_model_yaml = yaml_file.read()
anti_spoof_model = model_from_yaml(loaded_model_yaml)
anti_spoof_model.load_weights("trained_model/RGB_rPPG_merge_softmax_.h5")

# ------------------- Load Emotion Model -------------------
emotion_model = load_model("emotion_model/emotion_model.hdf5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ------------------- Load Haar Cascade -------------------
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ------------------- rPPG Setup -------------------
from rPPG.rPPG_Extracter import rPPG_Extracter

def get_rppg_pred(frame):
    use_classifier = True
    sub_roi = []
    fs = 20
    rPPG_extracter = rPPG_Extracter()
    rPPG_extracter.measure_rPPG(frame, use_classifier, sub_roi)
    rPPG = np.transpose(rPPG_extracter.rPPG)
    return rPPG

def make_pred_anti_spoof(frame, rppg_data):
    resized_img = cv2.resize(frame, (128, 128))
    single_x = img_to_array(resized_img)
    single_x = np.expand_dims(single_x, axis=0)
    pred = anti_spoof_model.predict([single_x, rppg_data])
    return pred

def preprocess_face_emotion(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    return face_img

def predict_emotion(face_img):
    processed_face = preprocess_face_emotion(face_img)
    predictions = emotion_model.predict(processed_face)
    max_index = np.argmax(predictions)
    return emotion_labels[max_index]

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Emotion & Anti-Spoof Detection", layout="wide")
st.title("🎭 Real-Time Emotion & Face Anti-Spoofing Detection")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.collected_results = []
        self.frames_buffer = 5
        self.accepted_falses = 1

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_crop = img[y:y+h, x:x+w]

            try:
                rppg_s = get_rppg_pred(face_crop)
                rppg_s = rppg_s.T
                spoof_pred = make_pred_anti_spoof(face_crop, rppg_s)
                spoof_result = np.argmax(spoof_pred)
                label_spoof = "Real" if spoof_result == 0 else "Fake"
                prob_real = round(float(spoof_pred[0][0]), 2)
                prob_fake = round(float(spoof_pred[0][1]), 2)

                self.collected_results.append(spoof_result)
                if len(self.collected_results) == self.frames_buffer:
                    if sum(self.collected_results) <= self.accepted_falses:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    self.collected_results.pop(0)
                else:
                    color = (255, 255, 0)

            except:
                label_spoof = "N/A"
                color = (128, 128, 128)
                prob_real, prob_fake = 0, 0

            try:
                emotion = predict_emotion(face_crop)
            except:
                emotion = "Unknown"

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f'{label_spoof}', (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img, f'Emotion: {emotion}', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(img, f'Real: {prob_real}  Fake: {prob_fake}',
                        (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img

# Run the streamlit-webrtc streamer
webrtc_streamer(key="emotion-spoof", video_processor_factory=VideoTransformer)

