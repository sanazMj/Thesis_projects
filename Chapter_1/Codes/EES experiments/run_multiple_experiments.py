from Main import ex
import sacred

Num = 5
num_epochs = 400

for i in range(Num):
     ex.run(config_updates={'num_epochs': num_epochs, 'mode_collapse': '', 'categorization': 2.3, 'Pixel_Full': 19})
for i in range(Num):
     ex.run(config_updates={'num_epochs': num_epochs, 'mode_collapse': '', 'categorization': 2, 'Pixel_Full': 9})

