import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_s

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size1=20, hidden_size2=40):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x, y=None):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        output = self.output(x)
        if y is not None:
            loss = nn.functional.cross_entropy(output, y)
            return loss
        else:
            return torch.softmax(output, dim=1)

# def built_simple():
#     x = np.random.random(5)
#     tensor_x = torch.Tensor(x) + np.random.normal(0, 0.5)
#     y = torch.argmax(tensor_x).item()
#     return x, y
    
def built_database(total_sample_num, num_classes):
    X = torch.randn(total_sample_num, num_classes)
    Y = torch.argmax(X, dim=1)
    return X, Y

def evaluate(model):
    model.eval()
    test_simple = 100
    x, y_test = built_database(test_simple, 5)
    count = torch.bincount(y_test, minlength=5)
    print(f"本次分类排布为：{count.tolist()}")

    correct, total = 0, 0
    with torch.no_grad():
        y_pred_prob = model(x)
        y_pred = torch.argmax(y_pred_prob, dim=1)
        correct = (y_pred == y_test).sum().item()
        total = y_test.size(0)
    acc = correct / total
    print(f"预测正确数：{correct},正确率：{acc:.2f}")
    return acc

def main():
    epoch_num = 40
    batch_size = 20
    total_simple = 6000
    input_size = 5
    num_classes = 5
    learn_rate = 0.001

    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = lr_s.CosineAnnealingLR(optim, T_max=epoch_num, eta_min=1e-8)
    train_x, train_y = built_database(total_simple, num_classes)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(total_simple // batch_size):
            start = batch_index * batch_size
            end = start + batch_size
            x = train_x[start : end]
            y = train_y[start : end]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        scheduler.step()
        mean_loss = np.mean(watch_loss)
        print(f"第{epoch + 1}轮训练，loss：{float(mean_loss)}")
        acc = evaluate(model)
        log.append([mean_loss, acc])
    
    torch.save(model.state_dict(), "multi_classes_model.bin")
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="loss")
    plt.plot(range(len(log)), [l[1] for l in log], label="acc")
    plt.legend()
    plt.show()

def predict(model_path, input_vec, num_classes):
    input_vec = np.array(input_vec)
    model = TorchModel(input_vec.shape[1], num_classes)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        y_prob = model(torch.FloatTensor(input_vec))
        y_pred = torch.argmax(y_prob, dim=1)
    for vec, prob, pred in zip(input_vec, y_prob, y_pred):
        print(f"输入：{vec}")
        print(f"预测类别：{pred.item()}, 各类概率：{prob.numpy()}")
if __name__ == '__main__':
    main()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("multi_classes_model.bin", test_vec, 5)
