import comb_network


counter = 0
network = comb_network.SimClRNet(numClasses = 12).cuda()
for child, mods in network.named_parameters():
	print("child", counter, "is")
	print(child)
	counter += 1
