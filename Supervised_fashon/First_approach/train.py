import torch
import numpy as np

from helpful import print_trainable_parameters, setTrainable, FreezeFirstN
from base_model import device

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score

def train(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs):
    # Lists to store metrics
    train_accuracy = []
    train_precision = []
    train_recall = []
    train_loss = []

    test_accuracy = []
    test_precision = []
    test_recall = []
    test_loss = []

    FreezeFirstN(model, 10000)
    print_trainable_parameters(model)

    print('Training started.')

    for epoch in range(num_epochs):
        model.train()
        all_labels = []
        all_preds = []
        cum_loss = 0

        # set more parameters
        if (epoch == 0):
            setTrainable(model, 692)
            print_trainable_parameters(model)
        elif (epoch == 9):
            setTrainable(model, 687)
            print_trainable_parameters(model)
        elif (epoch == 14):
            setTrainable(model, 675)
            print_trainable_parameters(model)
        elif (epoch == 19):
            setTrainable(model, 549)
            print_trainable_parameters(model)
        elif (epoch == 24):
            setTrainable(model, 248)
            print_trainable_parameters(model)
        elif (epoch == 27):
            setTrainable(model, 0)
            print_trainable_parameters(model)

        for images, labels, sites in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels, sites = images.to(device), labels.to(device), sites.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, sites)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Get predictions and true labels
            _, preds = torch.max(outputs, 1)
            print(preds.cpu().numpy(), labels.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            cum_loss += loss.item()
        scheduler.step()

        # Calculate metrics
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average=None)
        epoch_recall = recall_score(all_labels, all_preds, average=None)
        cum_loss /= len(train_loader)

        # Append metrics to lists
        train_accuracy.append(epoch_accuracy)
        train_precision.append(epoch_precision.tolist())
        train_recall.append(epoch_recall.tolist())
        train_loss.append(cum_loss)

        model.eval()
        test_labels = []
        test_preds = []
        t_loss = 0

        with torch.no_grad():
            for images, labels, sites in tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                images, labels, sites = images.to(device), labels.to(device), sites.to(device)

                # Forward pass
                outputs = model(images, sites)
                t_loss += criterion(outputs, labels).item()

                # Get predictions and true labels
                _, preds = torch.max(outputs, 1)

                test_labels.extend(labels.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        Tepoch_accuracy = accuracy_score(test_labels, test_preds)
        Tepoch_precision = precision_score(test_labels, test_preds, average=None)
        Tepoch_recall = recall_score(test_labels, test_preds, average=None)
        t_loss /= len(test_loader)

        # Append metrics to lists
        test_accuracy.append(Tepoch_accuracy)
        test_precision.append(Tepoch_precision.tolist())
        test_recall.append(Tepoch_recall.tolist())
        test_loss.append(t_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {cum_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Precision: {np.mean(epoch_precision):.4f}, Recall: {np.mean(epoch_recall):.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {t_loss:.4f}, Accuracy: {Tepoch_accuracy:.4f}, Precision: {np.mean(Tepoch_precision):.4f}, Recall: {np.mean(Tepoch_recall):.4f}')
        print("----------------------------------------------------------------------------------------------")
    print('Training finished.')
    return train_accuracy, train_precision, train_recall, train_loss, test_accuracy, test_precision, test_recall, test_loss