import torch
import pandas as pd
import tools.utils as gal_tools
from tools import read
from scripts.data_pipeline import process_image
from tools.network import CNN
from torchsummary import summary
import numpy as np
import os
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn import metrics


def train(learning_rate, epochs):
    contents = gal_tools.open_config()
    train_dir, test_dir, valid_dir = [contents["directories"][x] for x in list(contents["directories"].keys())[-3:]]
    train_list, test_list, valid_list = [
        os.listdir(contents["directories"][x]) for x in list(contents["directories"].keys())[-3:]
    ]
    errors_dir = contents["directories"]["errors"]
    model_path = os.path.join(contents["directories"]["model"], f"model_{np.random.randint(100000, 999999)}")
    os.mkdir(model_path)

    with open(errors_dir, "r") as file:
        errors_list = [ele[:-1] for ele in file.readlines()]

    device = ["cuda:0" if torch.cuda.is_available() else "cpu"][0]
    model = CNN().to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    summary(model, input_size=(3, 424, 424))
    labels_table = read.labels()

    training_loss = []
    validation_loss = []
    plt.ion()
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
    for epoch in trange(epochs, desc="Model: "):
        print("\n")
        np.random.shuffle(train_list)
        np.random.shuffle(valid_list)
        print(f"Epoch {epoch + 1}:")
        epoch_loss = 0
        model.train()
        for img in tqdm(train_list[0:500], desc="Training: ", colour="GREEN"):

            if img[:-4] in errors_list:
                continue

            optimizer.zero_grad()

            obj = labels_table.loc[labels_table["IMG_ID"] == int(img[:-4])]
            label = torch.tensor([
                [float(obj["SPIRAL"]), float(obj["ELLIPTICAL"]), float(obj["UNCERTAIN"])].index(1.)
            ]).to(device)
            model_input = torch.tensor(
                gal_tools.load_image(os.path.join(train_dir, img)).T
            ).unsqueeze(0).float().to(device)

            output = model(model_input)
            loss = loss_func(output, label)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        training_loss.append(epoch_loss)
        epoch_loss = 0
        model.eval()

        predicted = []
        actual = []
        for img in tqdm(valid_list[0:167], desc="Validation: ", colour="CYAN"):
            if img[:-4] in errors_list:
                continue
            obj = labels_table.loc[labels_table["IMG_ID"] == int(img[:-4])]
            label = [[float(obj["SPIRAL"]), float(obj["ELLIPTICAL"]), float(obj["UNCERTAIN"])].index(1.)]
            model_input = torch.tensor(
                gal_tools.load_image(os.path.join(valid_dir, img)).T
            ).unsqueeze(0).float().to(device)

            with torch.no_grad():
                output = model(model_input)
            loss = loss_func(output, torch.tensor(label).to(device))
            epoch_loss += loss.cpu().item()
            predicted.append(output.detach().cpu().numpy().argmax())
            actual.append(label[0])

        if not ax2.lines:
            gal_tools.plot(populated=False,
                           actual=actual,
                           predicted=predicted,
                           ax1=ax1,
                           ax2=ax2,
                           training_loss=training_loss,
                           validation_loss=validation_loss
                           )
        else:
            gal_tools.plot(populated=True,
                           actual=actual,
                           predicted=predicted,
                           ax1=ax1,
                           ax2=ax2,
                           training_loss=training_loss,
                           validation_loss=validation_loss
                           )
        validation_loss.append(epoch_loss)

    predicted = []
    actual = []
    for img in tqdm(test_list[0:167], desc="Testing: ", colour="RED"):
        if img[:-4] in errors_list:
            continue
        obj = labels_table.loc[labels_table["IMG_ID"] == int(img[:-4])]
        label = [[float(obj["SPIRAL"]), float(obj["ELLIPTICAL"]), float(obj["UNCERTAIN"])].index(1.)]
        model_input = torch.tensor(gal_tools.load_image(os.path.join(test_dir, img)).T).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(model_input)
        test_loss = loss_func(output, torch.tensor(label).to(device))
        predicted.append(output.detach().cpu().numpy().argmax())
        actual.append(label[0])

    plt.savefig(os.path.join(model_path, "learning_metrics.png"))
    print(f"Model ID: {model_path[13:]}")
