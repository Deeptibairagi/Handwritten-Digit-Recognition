from losses_28x28 import *

print(".........................Training.............................")
epochs=6
train_loss_list=[]
train_acc_list=[]
test_loss_list=[]
test_acc_list=[]

for epoch in range(epochs):
    model_28x28.train()
    train_correct=0
    train_total=0
    train_loss=0

    for train_images,train_labels in train_loader:
        train_images,train_labels=train_images.to(device),train_labels.to(device)

        optimizer.zero_grad()
        train_output = model_28x28(train_images)
        loss1 = criterion(train_output, train_labels)
        loss1.backward()
        optimizer.step()

        _,train_prediction=torch.max(train_output,1)
        train_total += train_labels.size(0)
        train_correct += (train_prediction == train_labels).sum().item()
        train_loss+= loss1.item()

    train_loss_list.append(train_loss/len(train_loader))
    
    train_accuracy = 100 * (train_correct / train_total)
    train_acc_list.append(train_accuracy)
    print(f"{epoch+1}/{epochs}..........train loss : {train_loss/len(train_loader)} | train accuarcy : {train_accuracy}")   
    
    print("..................evaluation.....................")
    model_28x28.eval()
    test_correct = 0
    test_total = 0
    test_loss=0

    with torch.no_grad():  
        for test_images, test_labels in test_loader:
            test_images,test_labels=test_images.to(device),test_labels.to(device)

            test_output = model_28x28(test_images)
            loss2=criterion(test_output,test_labels)
            _, test_prediction = torch.max(test_output, 1)
            test_total += test_labels.size(0)
            test_correct += (test_prediction == test_labels).sum().item()
            test_loss+= loss2.item()

        test_loss_list.append(test_loss/len(test_loader))
    
        test_accuracy = 100 * (test_correct / test_total)
        test_acc_list.append(test_accuracy)
        print(f"{epoch+1}/{epochs}..........test_loss : {test_loss/len(test_loader)} | Test_accuracy : {test_accuracy}")   
    
print(f"Train Accuracy is: {train_accuracy}")    
print(f"Test Accuracy : {test_accuracy}")   
    
        

        



