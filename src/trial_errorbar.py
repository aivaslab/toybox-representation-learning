import matplotlib.pyplot as plt
ff = 'x-large'
x = [0.0, 0.25, 0.5, 0.75, 1.0]
y = [31.14, 33.46, 29.19, 27.00, 29.56]
y2 = [63.58, 65.50, 59.8, 58.82, 60.86]
errs = [1.20, 0.27, 0.44, 0.82, 0.54]
errs2 = [0.71, 0.07, 0.82, 0.76, 0.44]
plt.errorbar(x, y, yerr = errs, label = "Cifar100")
plt.errorbar(x, y2, yerr = errs2, label = "Cifar10")
plt.xticks(x)
plt.ylim(0, 80)
plt.ylabel("Test Accuracy", fontsize = ff)
plt.xlabel("Weight of SimCLR-like loss in combined loss", fontsize = ff)
plt.legend(loc = "upper right")
plt.show()
