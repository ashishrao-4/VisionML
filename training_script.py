import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import io
import firebase_admin
from firebase_admin import credentials, storage
import os

# Initialize Firebase Admin SDK
cred = credentials.Certificate("visionml-flask-firebase-adminsdk-njze6-b90ca009af.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'visionml-flask.appspot.com'
})

# Custom dataset to load images from Firebase
class FirebaseImageDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = self._get_class_names_from_firebase()
        self._load_images_from_firebase()

    def _get_class_names_from_firebase(self):
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix='train_images/')
        class_names = set()
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) > 1 and parts[1]:
                class_names.add(parts[1])
        return sorted(class_names)

    def _load_images_from_firebase(self):
        bucket = storage.bucket()
        for label, class_name in enumerate(self.class_names):
            blobs = bucket.list_blobs(prefix=f'train_images/{class_name}/')
            for blob in blobs:
                image_data = blob.download_as_bytes()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                self.images.append(image)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a fixed size
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = FirebaseImageDataset(transform=train_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the CNN model
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = CatDogCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Save the trained model
os.makedirs('models', exist_ok=True)
model_path = 'models/trained_model.h5'
torch.save(model.state_dict(), model_path)
