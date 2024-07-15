import matplotlib.pyplot as plt
import torch
from 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

in_len = 1
out_len = 100
input = torch.rand(0, size=(in_len,))

results = []
for i in range(3):
    model = ESNReservoir(2, 2048, 2, pred_len=out_len).to(device)
    model.eval()
    result = model(input)
    results.append(result)

# plot the result in 2 separate plots one for x and y
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.ylim(-0.1,0.1)
plt.title("x")
for result in results:
    plt.plot(result[0, :, 0].cpu().detach().numpy())
plt.grid()

plt.subplot(2, 1, 2)
plt.ylim(-0.1,0.1)
plt.title("y")
for result in results:
    plt.plot(result[0, :, 1].cpu().detach().numpy())
plt.grid()

plt.show()
