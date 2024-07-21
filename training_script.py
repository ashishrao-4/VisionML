import io
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import firebase_admin
from firebase_admin import credentials, storage
from PIL import Image
import requests
import os

# Initialize Firebase Admin SDK
cred = credentials.Certificate("visionml-flask-firebase-adminsdk-njze6-bd9b7dd69d.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'visionml-flask.appspot.com'
})

# Define your model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 10)  # Example for MNIST

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x

def download_images_from_firebase():
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="train_images/")
    
    class_images = {}
    for blob in blobs:
        if not blob.name.endswith('/'):
            class_name = blob.name.split('/')[1]
            if class_name not in class_images:
                class_images[class_name] = []
            buffer = io.BytesIO()
            blob.download_to_file(buffer)
            buffer.seek(0)
            image = Image.open(buffer).convert('L')  # Convert to grayscale for MNIST
            class_images[class_name].append(image)
    
    return class_images

def train():
    # Initialize model, loss function, and optimizer
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Define data transformations
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    class_images = download_images_from_firebase()

    data = []
    class_name_to_label = {class_name: idx for idx, class_name in enumerate(class_images.keys())}
    
    for class_name, images in class_images.items():
        label = class_name_to_label[class_name]
        for image in images:
            data.append((transform(image), label))

    if not data:
        raise ValueError("No training data found in Firebase storage.")

    trainloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)

    num_epochs = 2  # Define the number of epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # Print loss after each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}] completed with loss: {running_loss / len(trainloader):.4f}')

    # Save the model to an H5 file
    model_save_path = 'trained_model.h5'
    torch.save(model.state_dict(), model_save_path)

    # Upload the model to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob('models/trained_model.h5')
    blob.upload_from_filename(model_save_path, content_type='application/octet-stream')
    print('Training completed and model uploaded to Firebase.')

    # Remove local model file
    os.remove(model_save_path)

if __name__ == "__main__":
    train()
