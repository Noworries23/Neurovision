import io
import json
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
import base64
import io
from io import BytesIO
from sklearn.metrics import confusion_matrix


app = Flask(__name__)

# Load the TorchScript model
model = torch.jit.load(r"C:\Users\Dsubs\Desktop\Neurovision_App\Med_Models\my_classification_model_epoch_10.pt", map_location=torch.device('cpu'))
model.eval()  # Evaluation mode, IMPORTANT
# Load the model
model = torch.jit.load(r"C:\Users\Dsubs\Desktop\Neurovision_App\Med_Models\my_classification_model_epoch_10.pt", map_location=torch.device('cpu'))

# Print the first few weights
for name, param in model.named_parameters():
    print(f"{name}: {param.data[:5]}")  # Print the first 5 values of each parameter
    break  # To avoid printing too much data

def transform_image(image_bytes):
    transform_image = transforms.Compose([
        transforms.Resize((496, 248)),  # Resize to model's input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform_image(image).unsqueeze(0)  # Add batch dimension

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    print(f"Input tensor: {tensor}")

    outputs = model.forward(tensor)
    print(f"Model outputs: {outputs}")

    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    
    # Define your class labels here
    class_labels = [
        "Healthy",
        "Mild Dementia",
        "Moderate Detected",
        "Very mild Dementia"
    ]
    
    # Get the predicted class label
    predicted_class = class_labels[predicted_idx]
    
    # Calculate confidence
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_idx].item()
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")
    
    return predicted_class, confidence * 100  # Return class name and confidence as a percentage

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('file')  # Get all uploaded files
    
    predictions = []
    for file in files:
        if file.filename == '':
            continue
        
        try:
            # Get the prediction for each image
            prediction, confidence = get_prediction(file.read())
            
            # Store the prediction and confidence
            predictions.append({
                'filename': file.filename,
                'prediction': prediction,
                'confidence': confidence
            })
        except Exception as e:
            predictions.append({
                'filename': file.filename,
                'error': f"Error processing image: {e}"
            })
    
    return render_template('results.html', predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)
