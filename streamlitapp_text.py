import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
# from transformers import pipeline
import datetime
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./models/keras_model.h5", compile=False)
# generator = pipeline('text-generation', model='gpt2')
# Load the labels
class_names = open("./models/labels.txt", "r").readlines()
class_name = ''

# Streamlit interface
st.title("Image Classification with Keras")
st.header("Upload an Image to Classify")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]


    # Display results
    st.write("Class:", class_name[2:].strip())
    st.write("Confidence Score:", confidence_score)

    # Display Report.
    if class_name[2:].strip() != "":
      if st.button("Generate Additional Recommendations:->"):
        report = f"""
        Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
        Skin Infection Detection:
        Skin Infection Present: Yes
        Severity: Moderate
    
    
        Recommendations:
        {'Further examination required'}
        """

        st.write(report)
        from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

        # Load the pre-trained GPT-2 model and tokenizer
        model_name = 'gpt2'
        model = TFGPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)  
        input_text = "Provide additional clinical recommendations based on the: " + class_name[2:].strip()
        input_ids = tokenizer.encode(input_text, return_tensors='tf')

    # Generate text
        output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Optionally, generate additional details using text generation
    # additional_info = generator("" , max_length=200)[0]['generated_text']
    # report += f"\nAdditional Recommendations:\n{additional_info}"
        st.write(generated_text)
