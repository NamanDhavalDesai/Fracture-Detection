import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # allow loading of truncated images
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve
from collections import Counter
import random
import shutil
import pickle

print(f"cuda available: {torch.cuda.is_available()}")

# directories for the dataset
data_dir = 'FracAtlas/images' # root folder for images
classes = ['Fractured', 'Non_fractured'] # classes in the dataset
splits = ['train', 'val', 'test'] # dataset splits
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15} # ratio for each split

def is_split_ready(data_dir, splits, classes):
    # check if every split folder for each class exists and has images
    for split in splits:
        for cls in classes:
            split_class_dir = os.path.join(data_dir, split, cls)
            if not os.path.isdir(split_class_dir) or not os.listdir(split_class_dir):
                return False
    return True

# if splits are not prepared then split the data
if not is_split_ready(data_dir, splits, classes):
    print("splitting data into train, val, and test sets...")
    # create directories for each split and class
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(data_dir, split, cls), exist_ok=True)

    # process each class folder in the original data
    for cls in classes:
        orig_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(orig_dir):
            print(f"original folder '{orig_dir}' not found. skipping...")
            continue
        # list image files with supported extensions
        images = [f for f in os.listdir(orig_dir) if f.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'))]
        random.shuffle(images) # shuffle images randomly
        total = len(images)
        train_end = int(total * split_ratios['train'])
        val_end = train_end + int(total * split_ratios['val'])

        # create dictionary to hold split images
        splits_images = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        # move images into their respective split folders
        for split, files in splits_images.items():
            for file in files:
                src = os.path.join(orig_dir, file)
                dst = os.path.join(data_dir, split, cls, file)
                shutil.move(src, dst)
    print("data split completed.")
else:
    print("train, val, and test folders already exist and contain images")

# directories for each split
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# check that necessary directories exist
for dir_path in [train_dir, val_dir, test_dir]:
    if not os.path.isdir(dir_path):
        print(f'could not find {dir_path}. double-check that the data is in {data_dir}')
        exit()
    else:
        print(f"found directory {dir_path}")

# select device for training based on availability
if torch.cuda.is_available():
    device = torch.device("cuda") # use cuda if available
elif torch.backends.mps.is_available():
    device = torch.device("mps") # use mps if available on mac
else:
    device = torch.device("cpu") # fall back to cpu

print("using device:", device)

# normalization ImageNet parameters for transforms
mean = [0.485, 0.456, 0.406] # mean for normalization
std = [0.229, 0.224, 0.225] # std for normalization

# transform for training with augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224), # randomly crop and resize to 224x224
    transforms.RandomHorizontalFlip(), # randomly flip horizontally
    transforms.RandomRotation(10), # randomly rotate by up to 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # jitter colors
    transforms.ToTensor(), # convert image to tensor
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)), # randomly erase parts of image
    transforms.Normalize(mean, std) # normalize using set mean and std
])

# transform for validation
val_transform = transforms.Compose([
    transforms.Resize(256), # resize smallest side to 256
    transforms.CenterCrop(224), # center crop to 224x224
    transforms.ToTensor(), # convert image to tensor
    transforms.Normalize(mean, std) # perform normalization
])
test_transform = val_transform # use the same transform as validation for test

# create datasets using image folder structure
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

# remap targets so that label 1 is fractured and 0 is non-fractured
train_dataset.samples = [(path, 1 - label) for (path, label) in train_dataset.samples]
train_dataset.targets = [1 - label for label in train_dataset.targets]
val_dataset.samples = [(path, 1 - label) for (path, label) in val_dataset.samples]
val_dataset.targets = [1 - label for label in val_dataset.targets]

# custom dataset for test images without labels
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_dir, transform=None):
        # initialize with test directory and optional transform
        self.test_dir = test_dir
        self.transform = transform
        # sort image files with supported extensions
        self.image_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(
            ('.jpg', '.png'))])
    def __len__(self):
        # return the number of test images
        return len(self.image_files)
    def __getitem__(self, idx):
        # load and transform the image at a given index
        img_path = os.path.join(self.test_dir, self.image_files[idx])
        image = Image.open(img_path).convert('rgb')
        if self.transform:
            image = self.transform(image)
        return image

test_dataset = TestDataset(test_dir, transform=test_transform)

# compute class weights for imbalanced classes
num_classes = 2 # total classes
counter = Counter(train_dataset.targets) # count occurrences per class
total = sum(counter.values()) # get total images
weights = [total / counter[i] for i in range(num_classes)] # compute weights inversely proportional to frequency
weights_tensor = torch.tensor(weights, dtype=torch.float).to(device) # create tensor for loss weighting
print("class weights:", weights)

# create dataloaders for train, val and test datasets
batch_size = 32 # batch size for each iteration
num_workers = 2 # workers for data loading
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                 shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
               shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
               shuffle=False, num_workers=num_workers)

# loss function with class weights
criterion = nn.CrossEntropyLoss(weight=weights_tensor) # cross entropy loss weighted for imbalance

# training function
def train_model(model, train_loader, val_loader, criterion, optimizer,
                device, num_epochs, patience=10):
    model.to(device) # move model to selected device
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None # use mixed precision if available
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # decay lr every 5 epochs
    best_val_auc = 0.0 # best validation auc so far
    patience_counter = 0 # counter for early stopping

    # history dictionary for tracking metrics
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model.train() # model to training mode
        running_loss = 0.0 # accumulate training loss
        running_corrects = 0 # accumulate correct predictions
        for inputs, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # reset gradients
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs) # forward pass
                    loss = criterion(outputs, labels) # compute loss
                scaler.scale(loss).backward() # backward pass with scaling
                scaler.step(optimizer) # update parameters
                scaler.update() # update scaler
            else:
                outputs = model(inputs) # forward pass
                loss = criterion(outputs, labels) # compute loss
                loss.backward() # backward pass
                optimizer.step() # update parameters
            running_loss += loss.item() * inputs.size(0) # update loss
            _, preds = torch.max(outputs, 1) # get predictions from output
            running_corrects += torch.sum(preds == labels.data) # update correct counts
        epoch_loss = running_loss / len(train_loader.dataset) # calculate average loss
        epoch_acc = running_corrects.float() / len(train_loader.dataset) # calculate accuracy

        history['loss'].append(epoch_loss) # store training loss
        history['accuracy'].append(epoch_acc.item()) # store training accuracy

        print(f"epoch {epoch+1}/{num_epochs} - train loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

        model.eval() # model to evaluation mode
        val_loss = 0.0 # initialize validation loss
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs) # forward pass
                loss = criterion(outputs, labels) # compute loss
                val_loss += loss.item() * inputs.size(0) # accumulate loss
            val_loss /= len(val_loader.dataset) # compute average validation loss
        history['val_loss'].append(val_loss) # store validation loss

        # validation metrics
        metrics, _, _, _ = evaluate_model_metrics(model, val_loader, device)
        val_auc = metrics.get("roc_auc", 0.0) # get roc auc
        val_acc = metrics.get("accuracy", 0.0) # get accuracy
        val_f1 = metrics.get("f1", 0.0) # get f1 score
        val_recall = metrics.get("recall", 0.0) # get recall

        history['val_accuracy'].append(val_acc) # store validation accuracy

        print(f"validation loss: {val_loss:.4f} | val auc: {val_auc:.4f} | val acc: {val_acc:.4f} | val f1: {val_f1:.4f}")
        
        # check for improvement to decide about early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc # update best score
            patience_counter = 0 # reset counter
            os.makedirs("models", exist_ok=True) # create models folder if needed
            torch.save(model.state_dict(),
                       f"models/best_{model.__class__.__name__}.pth") # save best model
        else:
            patience_counter += 1 # increment patience counter
            if patience_counter >= patience:
                print(f"early stopping triggered at epoch {epoch+1}")
                break

        scheduler.step() # update learning rate scheduler

    return model, history

# evaluation function to compute metrics on a given dataloader
def evaluate_model_metrics(model, dataloader, device):
    model.eval() # set model to evaluation mode
    all_labels = [] # accumulate true labels
    all_preds = [] # accumulate predicted labels
    all_probs = [] # accumulate predicted probabilities for class 1
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="evaluating metrics"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs) # forward pass
            probs = torch.softmax(outputs, dim=1) # get softmax probabilities
            prob_positive = probs[:, 1] # probability for class 1 (fractured)
            _, preds = torch.max(outputs, 1) # get predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(prob_positive.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds) # compute accuracy
    f1 = f1_score(all_labels, all_preds, pos_label=1) # compute f1 score for fractured
    recall = recall_score(all_labels, all_preds, pos_label=1) # compute recall for fractured
    try:
        roc_auc = roc_auc_score(all_labels, all_probs) # compute roc auc
    except ValueError:
        roc_auc = None # set to none if error
    fpr, tpr, _ = roc_curve(all_labels, all_probs) # compute roc curve values
    return {"accuracy": acc, "f1": f1, "recall": recall, "roc_auc": roc_auc,
            "fpr": fpr, "tpr": tpr}, all_labels, all_preds, all_probs

# plot roc curve and save as pdf
def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    plt.figure()
    plt.plot(fpr, tpr, label=f"roc curve (auc = {roc_auc:.3f})") # plot roc curve
    plt.plot([0, 1], [0, 1], "k--") # plot chance line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title(f"roc curve - {model_name}") # set title
    plt.legend(loc="lower right")
    os.makedirs("figs", exist_ok=True) # create figs folder if needed
    plt.savefig(f"figs/roc_{model_name}.pdf")
    plt.close()

# plot summary metrics across models
def plot_summary_metrics(results_dict):
    metrics = ["accuracy", "f1", "recall", "roc_auc"] # list of metrics to plot
    model_names = list(results_dict.keys()) # get model names
    summary = {metric: [results_dict[m][metric] for m in model_names]
               for metric in metrics} # organize metrics per model
    x = np.arange(len(model_names))
    width = 0.2 # bar width
    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, summary[metric], width, label=metric)
    plt.xticks(x + width * (len(metrics) - 1) / 2, model_names)
    plt.ylabel("score")
    plt.title("validation metrics comparison")
    plt.legend()
    os.makedirs("figs", exist_ok=True) # ensure figs folder exists
    plt.savefig("figs/metrics_summary.pdf")
    plt.close()

# evaluate ensemble by averaging predictions
def evaluate_ensemble(models, dataloader, device):
    for model in models:
        model.eval() # set each model to evaluation mode
    all_labels = [] # store true labels
    all_preds = [] # store predicted labels from ensemble
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="evaluating ensemble"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            probs = [] # list to collect probabilities from each model
            for model in models:
                outputs = model(inputs)
                probs.append(torch.softmax(outputs, dim=1))
            avg_probs = torch.mean(torch.stack(probs), dim=0) # average the probabilities
            _, preds = torch.max(avg_probs, 1) # get predictions from average
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds) # compute ensemble accuracy
    return acc

# evaluate ensemble metrics
def evaluate_ensemble_metrics(models, dataloader, device):
    for model in models:
        model.eval() # set each model to evaluation mode
    all_labels, all_preds, all_probs = [], [], [] # lists for metrics
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="evaluating ensemble metrics"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            probs = [] # list for each model's probabilities
            for model in models:
                outputs = model(inputs)
                probs.append(torch.softmax(outputs, dim=1))
            avg_probs = torch.mean(torch.stack(probs), dim=0) # average probabilities
            _, preds = torch.max(avg_probs, 1) # get predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(avg_probs[:, 1].cpu().numpy()) # probability for class 1
    acc = accuracy_score(all_labels, all_preds) # compute accuracy
    f1 = f1_score(all_labels, all_preds, pos_label=1) # compute f1 score
    recall = recall_score(all_labels, all_preds, pos_label=1) # compute recall
    try:
        roc_auc = roc_auc_score(all_labels, all_probs) # compute roc auc
    except ValueError:
        roc_auc = 0.0
    fpr, tpr, _ = roc_curve(all_labels, all_probs) # compute roc curve values
    return {"accuracy": acc, "f1": f1, "recall": recall, "roc_auc": roc_auc,
            "fpr": fpr, "tpr": tpr}

# plot all roc curves together
def plot_all_roc_curves(results_dict):
    plt.figure()
    for name, info in results_dict.items():
        metrics = info["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"],
                 label=f'{name} (auc = {metrics["roc_auc"]:.3f})') # plot each model's roc curve
    plt.plot([0, 1], [0, 1], "k--") # plot the diagonal chance line
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curves comparison")
    plt.legend()
    os.makedirs("figs", exist_ok=True) # create figs directory if needed
    plt.savefig("figs/roc_all_models.pdf")
    plt.close()

# plot training and validation loss over epochs
def plot_training_validation_loss(model_histories):
    model_names = list(model_histories.keys()) # get names of models
    histories = list(model_histories.values()) # get histories from each model

    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5),
                 sharey=True) # create subplots

    for i, (name, history) in enumerate(zip(model_names, histories)):
        ax = axes[i]
        ax.plot(history['loss'], label='training loss', color='blue')
        ax.plot(history['val_loss'], label='validation loss', color='red')
        ax.set_title(f'{name} loss')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.legend()
        ax.set_ylim(0, 10) # set y-axis limit

    plt.tight_layout()
    plt.savefig("figs/training_validation_loss.pdf")
    plt.close()

# plot training and validation accuracy over epochs
def plot_training_validation_accuracy(model_histories):
    model_names = list(model_histories.keys()) # get model names
    histories = list(model_histories.values()) # get history for each model

    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5),
                 sharey=True) # create subplots

    for i, (name, history) in enumerate(zip(model_names, histories)):
        ax = axes[i]
        ax.plot(history['accuracy'], label='training accuracy', color='blue')
        ax.plot(history['val_accuracy'], label='validation accuracy', color='red')
        ax.set_title(f'{name} accuracy')
        ax.set_xlabel('epochs')
        ax.set_ylabel('accuracy')
        ax.legend()
        ax.set_ylim(0, 1) # set y-axis to [0,1]

    plt.tight_layout()
    plt.savefig("figs/training_validation_accuracy.pdf")
    plt.close()

# main execution
if __name__ == '__main__':
    num_epochs = 100
    learning_rate = 1e-4 # for optimizer

    # flag to load pretrained models or train from scratch
    load_pretrained = True
    models_dict = {} # dictionary to hold models and their metrics

    if load_pretrained:
        print("loading pre-trained models...")

        # load resnet50 model
        resnet50 = models.resnet50(pretrained=True)
        in_features = resnet50.fc.in_features # get input dimension for final layer
        resnet50.fc = nn.Linear(in_features, num_classes) # replace final layer for our task
        resnet50.load_state_dict(torch.load(os.path.join("models", "best_ResNet.pth")))
        resnet50.to(device)
        models_dict["ResNet50"] = {"model": resnet50,
                             "metrics": evaluate_model_metrics(resnet50, val_loader, device)[0]}

        # load vgg16 model
        vgg16 = models.vgg16(pretrained=True)
        in_features = vgg16.classifier[6].in_features # get input dimension for final classifier layer
        vgg16.classifier[6] = nn.Linear(in_features, num_classes) # replace classifier layer
        vgg16.load_state_dict(torch.load(os.path.join("models", "best_VGG.pth")))
        vgg16.to(device)
        models_dict["VGG16"] = {"model": vgg16,
                          "metrics": evaluate_model_metrics(vgg16, val_loader, device)[0]}

        # load vision transformer model
        vit = models.vit_b_16(pretrained=True)
        in_features = vit.heads.head.in_features # get input dimension for transformer head
        vit.heads.head = nn.Linear(in_features, num_classes) # replace head
        vit.load_state_dict(torch.load(os.path.join("models", "best_VisionTransformer.pth")))
        vit.to(device)
        models_dict["ViT"] = {"model": vit,
                        "metrics": evaluate_model_metrics(vit, val_loader, device)[0]}

        # plot roc curves for each pretrained model
        for name, info in models_dict.items():
            metrics = info["metrics"]
            plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"], name)

        # plot summary metrics across models
        results_summary = {name: info["metrics"] for name, info in models_dict.items()}
        plot_summary_metrics(results_summary)

        print("\nmetrics for individual models:")
        header = "{:<20} {:<10} {:<10} {:<10} {:<10}".format("model", "accuracy",
                                                         "f1", "recall", "roc_auc")
        print(header)
        print("-" * len(header))
        for name, info in models_dict.items():
            m = info["metrics"]
            print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(name, m["accuracy"],
                                                                      m["f1"], m["recall"],
                                                                      m["roc_auc"]))

        # evaluate ensemble predictions for best ensemble combination
        best_ensemble = ['ResNet50', 'ViT']
        best_ensemble_models = [models_dict[name]["model"] for name in best_ensemble]
        ensemble_metrics = evaluate_ensemble_metrics(best_ensemble_models, val_loader, device)
        print("\nmetrics for best ensemble (resnet50, vit):")
        print("{:<20} {:<10} {:<10} {:<10} {:<10}".format("model", "accuracy",
                                                      "f1", "recall", "roc_auc"))
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format("best ensemble",
                                                                  ensemble_metrics["accuracy"],
                                                                  ensemble_metrics["f1"],
                                                                  ensemble_metrics["recall"],
                                                                  ensemble_metrics["roc_auc"]))
    else:
        model_histories = {} # dictionary to store training histories

        # train resnet50 model
        print("training resnet50...")
        resnet50 = models.resnet50(pretrained=True)
        in_features = resnet50.fc.in_features # get dimension for final layer
        resnet50.fc = nn.Linear(in_features, num_classes) # replace final layer
        optimizer = optim.Adam(resnet50.parameters(), lr=learning_rate)
        resnet50, history_resnet50 = train_model(resnet50, train_loader, val_loader,
                                                 criterion, optimizer, device, num_epochs)
        metrics_resnet50, _, _, _ = evaluate_model_metrics(resnet50, val_loader, device)
        models_dict["ResNet50"] = {"model": resnet50, "metrics": metrics_resnet50}
        model_histories["ResNet50"] = history_resnet50

        # train vgg16 model
        print("training vgg16...")
        vgg16 = models.vgg16(pretrained=True)
        in_features = vgg16.classifier[6].in_features # get dimension for classifier
        vgg16.classifier[6] = nn.Linear(in_features, num_classes) # replace last layer
        optimizer = optim.Adam(vgg16.parameters(), lr=learning_rate)
        vgg16, history_vgg16 = train_model(vgg16, train_loader, val_loader,
                                           criterion, optimizer, device, num_epochs)
        metrics_vgg16, _, _, _ = evaluate_model_metrics(vgg16, val_loader, device)
        models_dict["VGG16"] = {"model": vgg16, "metrics": metrics_vgg16}
        model_histories["VGG16"] = history_vgg16

        # train vision transformer model
        print("training vit...")
        vit = models.vit_b_16(pretrained=True)
        in_features = vit.heads.head.in_features # get dimension for transformer head
        vit.heads.head = nn.Linear(in_features, num_classes) # replace head
        optimizer = optim.Adam(vit.parameters(), lr=learning_rate)
        vit, history_vit = train_model(vit, train_loader, val_loader,
                                       criterion, optimizer, device, num_epochs)
        metrics_vit, _, _, _ = evaluate_model_metrics(vit, val_loader, device)
        models_dict["ViT"] = {"model": vit, "metrics": metrics_vit}
        model_histories["ViT"] = history_vit

        # save model histories to file
        with open("model_histories.pkl", "wb") as f:
            pickle.dump(model_histories, f)

        plot_training_validation_loss(model_histories) # plot loss curves
        plot_training_validation_accuracy(model_histories) # plot accuracy curves

        # plot roc curves for each model
        for name, info in models_dict.items():
            metrics = info["metrics"]
            plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"], name)

        # plot summary metrics
        results_summary = {name: info["metrics"] for name, info in models_dict.items()}
        plot_summary_metrics(results_summary)

        # define ensemble combinations to evaluate
        ensemble_combinations = [
            ["ResNet50", "VGG16"],
            ["ResNet50", "ViT"],
            ["VGG16", "ViT"],
            ["ResNet50", "VGG16", "ViT"]
        ]
        best_ensemble_acc = 0 # best ensemble accuracy so far
        best_ensemble = None # store best ensemble combination
        for combo in ensemble_combinations:
            ensemble_models = [models_dict[name]["model"] for name in combo]
            acc = evaluate_ensemble(ensemble_models, val_loader, device)
            print(f"ensemble {combo} validation accuracy: {acc:.4f}")
            if acc > best_ensemble_acc:
                best_ensemble_acc = acc
                best_ensemble = combo
        print(f"best ensemble: {best_ensemble} with validation accuracy: {best_ensemble_acc:.4f}")

        # plot all roc curves together, including ensemble if available
        all_metrics = {name: info for name, info in models_dict.items()}
        if best_ensemble is not None:
            ensemble_name = "ensemble_" + "_".join(best_ensemble)
            ensemble_metrics = evaluate_ensemble_metrics([models_dict[name]["model"]
                                                       for name in best_ensemble],
                                                       val_loader, device)
            all_metrics[ensemble_name] = {"model": None, "metrics": ensemble_metrics}
        plot_all_roc_curves(all_metrics)

        # save example training images in a grid and save as pdf
        folder = datasets.ImageFolder(train_dir, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]))
        loader = DataLoader(folder, batch_size=8, shuffle=True)
        Xexamples, Yexamples = next(iter(loader))
        for i in range(8):
            plt.subplot(2, 4, i+1)
            img = Xexamples[i].numpy().transpose(1, 2, 0)
            plt.imshow(img, interpolation="none")
            plt.title("fractured" if Yexamples[i] == 1 else "non-fractured")
            plt.xticks([])
            plt.yticks([])
        os.makedirs("figs", exist_ok=True)
        plt.savefig("figs/examples.pdf")