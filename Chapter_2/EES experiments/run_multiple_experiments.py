from Main import ex
import sacred

Num = 5
for i in range(Num):
     ex.run(config_updates={'num_epochs':400, 'Model_type':'Vanilla'})
for i in range(Num):
     ex.run(config_updates={'num_epochs': 400,'Model_type':'Conditional' ,'mode_collapse':''})
for i in range(Num):
     ex.run(config_updates={'num_epochs': 400,'Model_type':'Conditional' ,'mode_collapse':'Minibatch'})
for i in range(Num):
     ex.run(config_updates={'num_epochs': 400, 'Model_type': 'Conditional', 'mode_collapse': 'DSGAN'})

