import torch
import pandas as pd
from tools import read, plot, utils
from scripts.data_pipeline import process_image
from tools.network import CNN
from torchsummary import summary
import numpy as np
import os
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


def train(learning_rate, epochs):
    contents = read.config()
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
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # bare in mind here you will need to edit line 60 of torchsummary.torchsummary.py to show:
    # x = [torch.rand(batch_size, *in_size).type(dtype) for in_size in input_size]
    # https://github.com/sksq96/pytorch-summary/issues/168
    # and line 100 to show:
    # total_input_size = abs(np.prod(sum(input_size, ())) * batch_size * 4. / (1024 ** 2.))
    # https: // github.com / sksq96 / pytorch - summary / issues / 90

    # summary(model, input_size=[(3, 52, 52), tuple([1])], batch_size=1)

    labels_table = read.labels()
    label_headers = labels_table.columns[3:-1]

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
        for img in tqdm(train_list[0:200], desc="Training: ", colour="GREEN"):

            if img[:-4] in errors_list:
                continue

            optimizer.zero_grad()

            obj = labels_table.loc[labels_table["IMG_ID"] == int(img[:-4])]
            label = torch.tensor([
                np.float32(round(float(obj[heading]))) for heading in label_headers
            ]).unsqueeze(0).to(device)
            extinction = torch.tensor(np.float32(obj["EXTINCTION"].to_numpy())).unsqueeze(0).to(device)
            compressed_image = process_image(utils.load_image(os.path.join(train_dir, img)))
            model_input = torch.tensor(compressed_image).unsqueeze(0).float().to(device)
            output = model(model_input, extinction)
            loss = loss_func(output, label)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        training_loss.append(epoch_loss)
        epoch_loss = 0
        model.eval()

        predicted = []
        actual = []
        for img in tqdm(valid_list[0:66], desc="Validation: ", colour="CYAN"):
            if img[:-4] in errors_list:
                continue
            obj = labels_table.loc[labels_table["IMG_ID"] == int(img[:-4])]
            label = torch.tensor([np.float32(round(float(obj[heading]))) for heading in label_headers]).unsqueeze(0).to(device)
            extinction = torch.tensor(np.float32(obj["EXTINCTION"].to_numpy())).unsqueeze(0).to(device)
            compressed_image = process_image(utils.load_image(os.path.join(valid_dir, img)))
            model_input = torch.tensor(compressed_image).unsqueeze(0).float().to(device)

            with torch.no_grad():
                output = model(model_input, extinction)
            loss = loss_func(output, label)
            epoch_loss += loss.cpu().item()
            predicted.append(np.rint(output.detach().cpu().numpy()))
            actual.append(label[0].detach().cpu().numpy())

        actual = np.array(actual)
        predicted = np.array(predicted)[:, 0, :]

        if not ax2.lines:
            plot.learning_metrics(populated=False,
                                  actual=actual,
                                  predicted=predicted,
                                  ax1=ax1,
                                  ax2=ax2,
                                  training_loss=training_loss,
                                  validation_loss=validation_loss,
                                  display_labels=label_headers
                                  )
        else:
            plot.learning_metrics(populated=True,
                                  actual=actual,
                                  predicted=predicted,
                                  ax1=ax1,
                                  ax2=ax2,
                                  training_loss=training_loss,
                                  validation_loss=validation_loss,
                                  display_labels=label_headers
                                  )
        validation_loss.append(epoch_loss)

    predicted = []
    actual = []
    test_loss = 0
    for img in tqdm(test_list[0:66], desc="Testing: ", colour="RED"):
        if img[:-4] in errors_list:
            continue
        obj = labels_table.loc[labels_table["IMG_ID"] == int(img[:-4])]
        label = torch.tensor([np.float32(round(float(obj[heading]))) for heading in label_headers]).unsqueeze(0).to(device)
        compressed_image = process_image(utils.load_image(os.path.join(test_dir, img)))
        extinction = torch.tensor(np.float32(obj["EXTINCTION"].to_numpy())).unsqueeze(0).to(device)
        model_input = torch.tensor(compressed_image).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(model_input, extinction)
        test_loss = loss_func(output, torch.tensor(label).to(device))
        predicted.append(np.rint(output.detach().cpu().numpy()))
        actual.append(label[0].detach().cpu().numpy())
    actual = np.array(actual)
    predicted = np.array(predicted)[:, 0, :]

    plot.learning_metrics(populated=True,
                          actual=actual,
                          predicted=predicted,
                          ax1=ax1,
                          ax2=ax2,
                          training_loss=training_loss,
                          validation_loss=validation_loss,
                          display_labels=label_headers
                          )
    ax2.get_legend().remove()
    ax2.scatter(epochs-1, test_loss.item(), color="black", label="Testing")
    ax2.legend()
    plt.savefig(os.path.join(model_path, "learning_metrics.png"))
    print(f"Model ID: {model_path[13:]}")
