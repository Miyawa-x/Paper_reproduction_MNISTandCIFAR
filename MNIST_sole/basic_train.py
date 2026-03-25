import torch
import torch.nn as nn
import torch.optim as optim

def train_and_evaluate(model, train_loader, test_loader, device, epochs=3, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'test_acc': [], 'test_loss':[]} 

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_loss+=loss.item()

            if batch_idx % 100 == 99:  
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        avg_loss = total_loss/len(train_loader)
        history['train_loss'].append(avg_loss)


        model.eval()
        correct = 0
        total = 0
        
        test_loss = 0.0

        with torch.no_grad():
        
            for batch_idx, (images, labels) in enumerate(test_loader):
                
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs,labels)
                test_loss += loss.item()

                _, predicted =  torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        history['test_loss'].append(avg_test_loss)

        history['test_acc'].append(100*correct/total)

    return history