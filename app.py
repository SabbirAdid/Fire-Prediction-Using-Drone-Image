import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# মডেল লোড (Ensure the model file is present)
model = load_model("fire_detection_resnet50.h5")

# প্রেডিকশন ফাংশন
def predict_fire_gradio(image):
    img = cv2.resize(image, (250, 250))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    label = "🔥 Fire" if prediction < 0.5 else "❄️ Non-Fire"
    confidence = 1 - prediction if prediction < 0.5 else prediction
    return f"{label} (Confidence: {confidence:.2f})"

# Gradio ইন্টারফেস
interface = gr.Interface(
    fn=predict_fire_gradio,
    inputs=gr.Image(type="numpy", label="আপনার ছবি আপলোড করুন"),
    outputs=gr.Textbox(label="প্রেডিকশন ফলাফল"),
    title="🔥 Fire Detection System",
    description="একটি স্যাটেলাইট বা যেকোনো ছবি আপলোড করুন, মডেল বলবে এটি Fire না Non-Fire।"
)

interface.launch()