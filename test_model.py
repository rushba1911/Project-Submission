import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score, precision_score, recall_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from tqdm import tqdm

# Set device and random seeds for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print(f"Using device: {device}")

# Data Augmentation and Normalization
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Data
train_dir = 'C:/Users/HP/Documents/Smart-Sort/DATASET/TRAIN'
val_dir = 'C:/Users/HP/Documents/Smart-Sort/DATASET/TEST'

print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Compute and print class distribution
train_targets = np.array(train_dataset.targets)
class_distribution = np.bincount(train_targets)
print("\nClass distribution in training set:")
for i, count in enumerate(class_distribution):
    print(f"Class {train_dataset.classes[i]}: {count} samples ({count/len(train_dataset)*100:.2f}%)")

# Compute Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_targets), y=train_targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("\nComputed class weights:", class_weights.cpu().numpy())

# Model Definition
print("\nInitializing model...")
# model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model = models.mobilenet_v2()
model.load_state_dict(torch.load('C:/Users/HP/models/mobilenet_v2-b0353104.pth'))


print("Freezing pretrained layers...")
for param in model.parameters():
    param.requires_grad = False

# Modify final layer
model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
model = model.to(device)
print("Model architecture modified for binary classification")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_times = []
    
    print(f"\nEpoch {epoch+1} Training:")
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        batch_start = time()
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        batch_end = time()
        batch_times.append(batch_end - batch_start)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    avg_batch_time = np.mean(batch_times)
    
    print(f"\nEpoch {epoch+1} Statistics:")
    print(f"Average Loss: {epoch_loss:.4f}")
    print(f"Training Accuracy: {epoch_acc:.2f}%")
    print(f"Average batch processing time: {avg_batch_time:.3f} seconds")
    
    return epoch_loss, epoch_acc

def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    val_loss = running_loss / len(val_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return val_loss, all_preds, all_labels, all_probs

def plot_metrics(all_labels, all_preds, all_probs, classes):
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(all_labels, all_preds)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    
    # ROC Curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(all_labels, all_probs):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    # Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.plot(recall, precision, 
            label=f'PR curve (AP = {average_precision_score(all_labels, all_probs):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

# Training Loop
num_epochs = 20
best_val_loss = float('inf')
training_history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_metrics': []
}

print("\nStarting training...")
training_start = time()

for epoch in range(num_epochs):
    epoch_start = time()
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
    training_history['train_loss'].append(train_loss)
    training_history['train_acc'].append(train_acc)
    
    # Evaluate
    val_loss, val_preds, val_labels, val_probs = evaluate_model(model, val_loader, criterion)
    training_history['val_loss'].append(val_loss)
    
    epoch_end = time()
    epoch_duration = epoch_end - epoch_start
    
    # Calculate metrics
    accuracy = (val_preds == val_labels).mean() * 100
    precision = precision_score(val_labels, val_preds, average='weighted')
    recall = recall_score(val_labels, val_preds, average='weighted')
    f1 = f1_score(val_labels, val_preds, average='weighted')
    auc_roc = roc_auc_score(val_labels, val_probs)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }
    training_history['val_metrics'].append(metrics)
    
    print(f"\nEpoch {epoch+1} Complete:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation AUC-ROC: {auc_roc:.4f}")
    print(f"Epoch duration: {epoch_duration:.2f} seconds")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("New best model saved!")

training_duration = time() - training_start
print(f"\nTraining completed in {training_duration:.2f} seconds")

# Final Evaluation
print("\nFinal Model Evaluation:")
val_loss, final_preds, final_labels, final_probs = evaluate_model(model, val_loader, criterion)

# Print detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(final_labels, final_preds, 
                          target_names=train_dataset.classes,
                          digits=4))

# Plot all metrics
plot_metrics(final_labels, final_preds, final_probs, train_dataset.classes)

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(training_history['train_loss'], label='Training Loss')
plt.plot(training_history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_history['train_acc'], label='Training Accuracy')
plt.plot([m['accuracy'] for m in training_history['val_metrics']], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Print final summary
print("\nFinal Model Performance Summary:")
print("-" * 50)
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Final Metrics:")
for metric, value in training_history['val_metrics'][-1].items():
    print(f"{metric}: {value:.4f}")

# Calculate per-class metrics
print("\nPer-class Performance Metrics:")
print("-" * 50)
for i, class_name in enumerate(train_dataset.classes):
    class_precision = precision_score(final_labels, final_preds, labels=[i], average=None)[0]
    class_recall = recall_score(final_labels, final_preds, labels=[i], average=None)[0]
    class_f1 = f1_score(final_labels, final_preds, labels=[i], average=None)[0]
    
    print(f"\nClass: {class_name}")
    print(f"Precision: {class_precision:.4f}")
    print(f"Recall: {class_recall:.4f}")
    print(f"F1-Score: {class_f1:.4f}")

# Save final results to file
with open('training_results.txt', 'w') as f:
    f.write("Waste Classification Model Training Results\n")
    f.write("-" * 50 + "\n")
    f.write(f"Training Duration: {training_duration:.2f} seconds\n")
    f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
    f.write("\nFinal Classification Report:\n")
    f.write(classification_report(final_labels, final_preds, 
                                target_names=train_dataset.classes,
                                digits=4))