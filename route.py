import os
import pandas as pd
import numpy as np
import librosa
from PIL import Image
from torchvision.transforms import transforms
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import asyncio

# Define the MultiTaskAudioProcessing class
class MultiTaskAudioProcessing(nn.Module):
    def __init__(self, num_gender_classes, num_age_classes, num_accent_classes):
        super(MultiTaskAudioProcessing, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Increased size of fully connected layer
        self.fc2 = nn.Linear(1024, 512)  # Reduced size of fully connected layer
        self.fc_gender = nn.Linear(512, num_gender_classes)
        self.fc_age = nn.Linear(512, num_age_classes)
        self.fc_accent = nn.Linear(512, num_accent_classes)
        self.dropout = nn.Dropout(p=0.5)  # Increased dropout rate

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        gender_output = self.fc_gender(x)
        age_output = self.fc_age(x)
        accent_output = self.fc_accent(x)
        return gender_output, age_output, accent_output
    
save_data = pd.read_csv('output.csv')
le = LabelEncoder()

# Encoding age, accent, and gender data
label_encoders = {}
for column in ['age', 'gender', 'accents']:
    le = LabelEncoder()
    save_data[column] = le.fit_transform(save_data[column])
    label_encoders[column] = le

train_data, test_and_eval_data = train_test_split(save_data, test_size=0.4, random_state=42)
    
num_gender_classes = len(train_data['gender'].unique())
num_age_classes = len(train_data['age'].unique())
num_accent_classes = len(train_data['accents'].unique())

# Initialize a new instance of the MultiTaskAudioProcessing model
loaded_model_pt = MultiTaskAudioProcessing(num_gender_classes, num_age_classes, num_accent_classes)

# Load the model state dictionary
loaded_model_pt.load_state_dict(torch.load('multitask_model.pth'))

# Set the model to evaluation mode
loaded_model_pt.eval()


# Initialize Flask app
app = Flask(__name__)

# Define preprocessing functions
async def test_preprocess_audio(audio_file_path):
    # Load the audio file
    y, sr = librosa.load(audio_file_path, sr=16000)  # Adjust sr as per your model requirements

    # Extract audio features (e.g., Mel spectrograms)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max(mel_spec))

    # Normalize the spectrogram
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    # Convert to PIL Image
    mel_spec_image = Image.fromarray((mel_spec_normalized * 255).astype(np.uint8))

    # Convert the image to grayscale
    mel_spec_image = mel_spec_image.convert('RGB')


    # Apply transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomResizedCrop(size=64, scale=(1.0, 1.1), ratio=(1.0, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mel_spec_tensor = transform(mel_spec_image)

    # Add batch dimension
    mel_spec_tensor = mel_spec_tensor.unsqueeze(0)

    return mel_spec_tensor

# Define prediction function
async def make_predictions(audio_file_path):
    # Preprocess the audio file
    mel_spec_tensor = await test_preprocess_audio(audio_file_path)

    # Make predictions
    with torch.no_grad():
        gender_output, age_output, accent_output = loaded_model_pt(mel_spec_tensor)

    # Decode predictions
    gender_prediction = label_encoders['gender'].inverse_transform([torch.argmax(gender_output).item()])[0]
    age_prediction = label_encoders['age'].inverse_transform([torch.argmax(age_output).item()])[0]
    accent_prediction = label_encoders['accents'].inverse_transform([torch.argmax(accent_output).item()])[0]
    
    # Return predictions
    return gender_prediction, age_prediction, accent_prediction

# Define Flask routes
@app.route('/', methods=['GET', 'POST'])
async def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            # Create the 'uploads' directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            # Save the audio file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Make predictions
            gender_prediction, age_prediction, accent_prediction = await make_predictions(file_path)

            # Return predictions to the user
            return render_template('index.html', gender=gender_prediction, age=age_prediction, accent=accent_prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
