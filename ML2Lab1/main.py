import numpy as np
from sklearn.model_selection import train_test_split
import multiprocessing

from Program import Program
from Plotter import Plotter

# Available Datasets: boston, concrete, friedm, istanbul, laser, plastic, quakes, stock, wizmir

def get_integer(prompt=""):
	print(prompt)
	while True:
		action = input()
		try:
			action = int(action)
			return action

		except ValueError:
			print("Input must be an integer..")


def run_prog():
	choice = """
	Choose dataset:
	0. Exit
	1. boston
	2. concrete
	3. friedm
	4. istanbul
	5. laser
	6. plastic
	7. quakes
	8. stock
	9. wizmir
	10. All of the above (Will run all in parallel!)
	"""

	while True:
		action = get_integer(choice)
		if action == 0:
			return False
		elif action == 1:
			execute("boston")
			break
		elif action == 2:
			execute("concrete")
			break
		elif action == 3:
			execute("friedm")
			break
		elif action == 4:
			execute("istanbul")
			break
		elif action == 5:
			execute("laser")
			break
		elif action == 6:
			execute("plastic")
			break
		elif action == 7:
			execute("quakes")
			break
		elif action == 8:
			execute("stock")
			break
		elif action == 9:
			execute("wizmir")
			break
		elif action == 10:
			executeAllParallel()
			break
		else:
			print("Invalid number")

	return True


def execute(dataset):
	# Dataset to use
	DatasetList = []
	DatasetList.append(dataset)
	Index = [0]
	
	program = Program()

	a_pool = multiprocessing.Pool(processes = len(Index))
	InfoList = a_pool.starmap(program.Run, zip(DatasetList, Index))

	# Plot results in InfoList. (Containing results from one dataset)
	Plotter.Plot(InfoList)


def executeAllParallel():
	# Dataset to use
	DatasetList = ["boston", "concrete", "friedm", "istanbul", "laser", "plastic", "quakes", "stock", "wizmir"]
	Index = [0, 1, 2, 3, 4, 5, 6, 7, 8]

	program = Program()

	a_pool = multiprocessing.Pool(processes = len(Index))
	print("This App uses multiprocessing, computer might freeze for a while :)")
	InfoList = a_pool.starmap(program.Run, zip(DatasetList, Index))

	# Plot results in InfoList. (Containing results from all dataset)
	Plotter.Plot(InfoList)


if __name__ == '__main__':
	run_prog()
