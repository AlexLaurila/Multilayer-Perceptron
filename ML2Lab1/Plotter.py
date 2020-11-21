from matplotlib import pyplot as plt
from itertools import chain

class Plotter:
	def Plot(InfoList):
		rows = columns = 1 if len(InfoList) == 1 else 3
		fig=plt.figure(1)

		for i in range(len(InfoList)):
			# Get min/max values for plot axis
			max_all = max(chain(InfoList[i].predictedRenomalized, InfoList[i].target))
			min_all = min(chain(InfoList[i].predictedRenomalized, InfoList[i].target))
		
			# Create figure and subplots
			ax = fig.add_subplot(rows, columns, i+1)
			plt.title(f"{InfoList[i].Dataset}", weight='bold')
			#medelLoss = "Medel-loss: %.4f   \n" % InfoList[i].Loss
			trainTime = "TrainTime: %.2f sekunder    \n" % InfoList[i].trainTime
			testTime = "TestTime: %.3f sekunder    \n" % InfoList[i].testTime
			#textString = medelLoss + trainTime + testTime
			textString = trainTime + testTime
		
			# Plot settings
			plt.ylabel(textString, fontsize=10, rotation=0, ha='right', weight='bold')
		
			# Set limits for x and y axis
			plt.xlim((min_all,max_all))
			plt.ylim((min_all,max_all))
			plt.gca().set_aspect('equal', adjustable='box')
		
			# Plot
			plt.scatter(InfoList[i].predictedRenomalized, InfoList[i].target)
	
		# Set main plot title
		plt.suptitle(f"Epochs: {InfoList[0].n_epochs}      Layers: {InfoList[0].n_layers}\n Learning rate: {InfoList[0].learning_rate}      Decay: {InfoList[0].decay}", weight='bold')
		# Show plot
		plt.show()
