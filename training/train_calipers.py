import pandas as pd
from PIL import Image
import torch, os, cv2
from torch.utils.data import Dataset, Sampler
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchinfo import summary
from tqdm import tqdm
from training_util import *
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")


def load_image(image_path, image_size, crop_x=None, crop_y=None, crop_w=None, crop_h=None, train=True):
    image = Image.open(image_path)
    
    # Apply crop if crop parameters are provided
    if crop_x is not None and crop_y is not None and crop_w is not None and crop_h is not None:
        # Convert to integers in case they're floats
        crop_x, crop_y, crop_w, crop_h = int(crop_x), int(crop_y), int(crop_w), int(crop_h)
        # PIL crop uses (left, upper, right, lower)
        image = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    
    if train:
        # Apply augmentations
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=(0.7, 1.1), contrast=(0.35, 1.15), saturation=(0, 1.5), hue=(-0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            StaticNoise(intensity_min=0.0, intensity_max=0.025),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        # Only preprocessing, no augmentations
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    return preprocess(image)


class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, image_size, train=True, save_debug=False, debug_dir=None):
        self.train = train
        self.df = dataframe
        self.img_dir = img_dir
        self.image_size = image_size
        self.save_debug = save_debug
        self.debug_dir = debug_dir
        
        # Create debug directory if it doesn't exist and we're saving debug images
        if self.save_debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id = row['ImageName']
        
        # Get crop parameters from CSV
        crop_x = row.get('crop_x', None)
        crop_y = row.get('crop_y', None)
        crop_w = row.get('crop_w', None)
        crop_h = row.get('crop_h', None)
        
        # Load and crop the image
        img = load_image(f"{self.img_dir}/{img_id}", self.image_size, 
                        crop_x=crop_x, crop_y=crop_y, crop_w=crop_w, crop_h=crop_h,
                        train=self.train)
        
        # Save debug image if enabled
        if self.save_debug and self.debug_dir:
            img_disp = img.clone().detach()  
            img_disp = img_disp * 0.5 + 0.5
            if img_disp.shape[0] == 1:
                img_disp = img_disp.squeeze(0)
            
            # Save the image
            plt.figure(figsize=(6, 6))
            plt.imshow(img_disp.numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f"Label: {int(row['has_calipers'])}")
            
            # Create filename with label info
            base_name = os.path.splitext(img_id)[0]
            save_path = os.path.join(self.debug_dir, f"{base_name}_label{int(row['has_calipers'])}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        
        has_calipers = torch.tensor(row['has_calipers'], dtype=torch.float32)
        return img, has_calipers


class BalancedBatchSampler(Sampler):
    """
    Sampler that creates batches with equal numbers of positive and negative samples.
    Each batch will have batch_size/2 positives and batch_size/2 negatives.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Get indices for positive and negative samples
        self.positive_indices = []
        self.negative_indices = []
        
        for idx in range(len(dataset)):
            label = dataset.df.loc[idx, 'has_calipers']
            if label == 1:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)
        
        self.positive_indices = np.array(self.positive_indices)
        self.negative_indices = np.array(self.negative_indices)
        
        print(f"BalancedBatchSampler initialized:")
        print(f"  Positive samples: {len(self.positive_indices)}")
        print(f"  Negative samples: {len(self.negative_indices)}")
        
        # Calculate number of batches based on the smaller class
        samples_per_class = self.batch_size // 2
        self.num_batches = min(len(self.positive_indices), len(self.negative_indices)) // samples_per_class
        
    def __iter__(self):
        # Shuffle indices at the start of each epoch
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)
        
        samples_per_class = self.batch_size // 2
        
        for i in range(self.num_batches):
            # Sample positives and negatives for this batch
            batch_positive = self.positive_indices[i * samples_per_class:(i + 1) * samples_per_class]
            batch_negative = self.negative_indices[i * samples_per_class:(i + 1) * samples_per_class]
            
            # Combine and shuffle within batch
            batch = np.concatenate([batch_positive, batch_negative])
            np.random.shuffle(batch)
            
            yield batch.tolist()
    
    def __len__(self):
        return self.num_batches

    
class Net(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        
        # Load the pretrained ResNet18 model
        self.model = models.resnet18(pretrained=pretrained)
        
        # Change the first layer because the images are black and white.
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Change the final layer because this is a binary classification problem.
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 1)
        
        # Apply sigmoid activation for binary classification
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x)
        return x
    
    
def test_images(csv_path, img_dir, model_name, image_size, batch_size=16, threshold=0.37):
    # Load the model
    model = Net()
    model.load_state_dict(torch.load(f"{env}/model/{model_name}.pt"))
    model = model.to(device)
    model.eval()
    
    # Load validation data
    df = pd.read_csv(csv_path)
    val_df = df[df['val'] == 1].reset_index(drop=True)
    
    val_dataset = CustomDataset(val_df, img_dir, image_size, train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    failed_images = []
    
    current_idx = 0
    
    with torch.no_grad():
        for img, has_calipers in val_loader:
            img = img.to(device)
            has_calipers = has_calipers.to(device)
            
            has_calipers_pred = model(img)
            predicted = (has_calipers_pred.data > threshold).float()
            
            # Check for failed predictions
            for i in range(len(has_calipers)):
                if current_idx >= len(val_df):
                    break
                    
                image_name = val_df.loc[current_idx, 'ImageName']
                true_label = has_calipers[i].item()
                pred_label = predicted[i].item()
                pred_prob = has_calipers_pred[i].item()
                
                total += 1
                if pred_label == true_label:
                    correct += 1
                else:
                    failed_images.append({
                        'image': image_name,
                        'true_label': int(true_label),
                        'pred_label': int(pred_label),
                        'probability': pred_prob
                    })
                
                current_idx += 1
    
    accuracy = 100 * correct / total
    print(f"\nValidation Set Results (threshold={threshold}):")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"\nFailed Images ({len(failed_images)}):")
    
    for fail in failed_images:
        print(f"{fail['image']}: True={fail['true_label']}, Pred={fail['pred_label']}, Prob={fail['probability']:.4f}")
    
    return accuracy, failed_images



def train_model(model, train_loader, val_loader, num_epochs, model_name):
    print("Training..")
    criterion_class = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_auc = 0.0  # Track the best validation AUC

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        total_train = 0
        correct_train = 0
        train_losses = []
        
        # Lists to store predictions and labels for AUC calculation
        train_predictions = []
        train_labels = []

        for I, (img, has_calipers) in enumerate(train_loader):
            img = img.to(device)
            has_calipers = has_calipers.to(device)

            optimizer.zero_grad()
            has_calipers_pred = model(img)

            predicted_train = (has_calipers_pred.data > 0.5).float() # calculate predicted labels
            total_train += has_calipers.size(0)
            correct_train += (predicted_train == has_calipers).sum().item()

            caliper_loss = criterion_class(has_calipers_pred, has_calipers)
            train_losses.append(caliper_loss.item())
            caliper_loss.backward()
            optimizer.step()
            
            # Store predictions and labels for AUC
            train_predictions.extend(has_calipers_pred.detach().cpu().numpy().tolist())
            train_labels.extend(has_calipers.cpu().numpy().tolist())

        train_loss = sum(train_losses) / len(train_losses) # Average loss
        train_acc = 100 * correct_train / total_train  # Training accuracy
        train_auc = roc_auc_score(train_labels, train_predictions)  # Training AUC

        # Start of validation
        model.eval()  # Set the model to evaluation mode
        val_losses_class = []
        val_total = 0
        val_correct = 0
        
        # Lists to store predictions and labels for AUC calculation
        val_predictions = []
        val_labels = []

        with torch.no_grad():  # No need to track the gradients
            for img, has_calipers in val_loader:
                img = img.to(device)
                has_calipers = has_calipers.to(device)

                has_calipers_pred = model(img)
                predicted_val = (has_calipers_pred.data > 0.5).float() # calculate predicted labels
                val_total += has_calipers.size(0)
                val_correct += (predicted_val == has_calipers).sum().item()

                loss_class2 = criterion_class(has_calipers_pred, has_calipers)
                val_losses_class.append(loss_class2.item())
                
                # Store predictions and labels for AUC
                val_predictions.extend(has_calipers_pred.cpu().numpy().tolist())
                val_labels.extend(has_calipers.cpu().numpy().tolist())

        val_loss_class = sum(val_losses_class) / len(val_losses_class)
        val_acc = 100 * val_correct / val_total  # Validation accuracy
        val_auc = roc_auc_score(val_labels, val_predictions)  # Validation AUC

        print(f"\nEpoch {epoch+1} Train | Validation")
        print(f"Loss:     {train_loss:.4f} / {val_loss_class:.4f}")
        print(f"Accuracy: {train_acc:.1f}% / {val_acc:.1f}%")
        print(f"AUC:      {train_auc:.4f} / {val_auc:.4f}")
        
        # Save model if validation AUC improved
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            
            # Calculate best threshold
            val_predictions_array = np.array(val_predictions)
            val_labels_array = np.array(val_labels)
            
            # Test different thresholds to find the one with best accuracy
            thresholds = np.arange(0.0, 1.0, 0.01)
            best_threshold = 0.5
            best_threshold_acc = 0.0
            
            for threshold in thresholds:
                predicted_with_threshold = (val_predictions_array > threshold).astype(float)
                threshold_acc = 100 * np.sum(predicted_with_threshold == val_labels_array) / len(val_labels_array)
                
                if threshold_acc > best_threshold_acc:
                    best_threshold_acc = threshold_acc
                    best_threshold = threshold
            
            torch.save(model.state_dict(), f"{env}/model/{model_name}.pt")
            print(f"*** New best validation AUC: {best_val_auc:.4f} - Model saved! ***")
            print(f"*** Best threshold: {best_threshold:.3f} - Validation accuracy: {best_threshold_acc:.2f}% ***")


def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # No need to track the gradients
        for img, has_calipers in test_loader:
            img = img.to(device)
            has_calipers = has_calipers.to(device)

            outputs = model(img)
            predicted = (outputs.data > 0.5).float() # calculate predicted labels
            total += has_calipers.size(0)
            correct += (predicted == has_calipers).sum().item()

    return 100 * correct / total  # Test accuracy

def compare_models(model1, model2, csv_path, img_dir, image_size, batch_size):
    df = pd.read_csv(csv_path)
    # Use val column to get validation data (where val == 1)
    test_df = df[df['val'] == 1]
    test_dataset = CustomDataset(test_df.reset_index(drop=True), img_dir, image_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model1 = model1.to(device)
    model2 = model2.to(device)

    model1_acc = evaluate_model(model1, test_loader)
    model2_acc = evaluate_model(model2, test_loader)

    print(f"Model 1 accuracy: {model1_acc:.2f}%")
    print(f"Model 2 accuracy: {model2_acc:.2f}%")
    
    
    
    

if __name__ == "__main__":
    
    dataset_dir = "D:/DATA/CADBUSI/training_sets/Caliper Set/"
    csv_path = f"{dataset_dir}/train_caliper.csv"
    img_dir = f"{dataset_dir}/images"
    debug_dir = f"{dataset_dir}/debug"

    # load data
    df = pd.read_csv(csv_path)
    
    image_size = 256
    batch_size = 16
    model_name = "caliper_detect_10_7_25"
    num_epochs = 25

    # Split based on val column: 0 for training, 1 for validation
    train_df = df[df['val'] == 0]
    val_df = df[df['val'] == 1]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Count positives and negatives in training set
    train_positive = train_df[train_df['has_calipers'] == 1]
    train_negative = train_df[train_df['has_calipers'] == 0]
    
    print(f"\nTraining set composition:")
    print(f"  Positive samples: {len(train_positive)}")
    print(f"  Negative samples: {len(train_negative)}")
    
    # create dataset
    train_dataset = CustomDataset(train_df.reset_index(drop=True), img_dir, image_size, 
                                   save_debug=False, debug_dir=debug_dir, train=True)
    val_dataset = CustomDataset(val_df.reset_index(drop=True), img_dir, image_size,
                                 save_debug=False, debug_dir=None, train=False)

    # Create balanced batch sampler for training
    train_sampler = BalancedBatchSampler(train_dataset, batch_size)
    
    # Create dataloaders - use batch_sampler for training, regular for validation
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nBatches per epoch: {len(train_loader)}")

    # Initialize the model and move it to the GPU
    model = Net().to(device)
    summary(model, input_size=(batch_size, 1, image_size, image_size))



    if True:
        train_model(model, train_loader, val_loader, num_epochs, model_name)

    if False:
        test_images(csv_path, img_dir, model_name, image_size, batch_size=batch_size, threshold=0.37)
        
    if False:
        # Initialize the models and move them to the GPU
        model1 = Net().to(device)
        model2 = Net().to(device)

        # Load the trained weights
        model1.load_state_dict(torch.load(f'F:\CODE\CASBUSI\CASBUSI-Database\ML_processing/models/caliper_model.pt'))
        model2.load_state_dict(torch.load(f"{env}/model/{model_name}.pt"))

        compare_models(model1, model2, csv_path, img_dir, image_size, batch_size)