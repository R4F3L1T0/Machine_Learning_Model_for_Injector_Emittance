import numpy as np
import keras
import keras.backend as K                        #using Tensorflow
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras import metrics, optimizers
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import GridSearchCV		   #GridSearch for hyperparameters

sns.set_style('darkgrid')

reshuffle_data = False	 		#True if you want to reshuffle the data, and split them in train set and test set
add_noise = False				#True if you want to add some gaussian noise to the train set


#################
### FUNCTIONS ###
#################

def glorot_normal():
	return keras.initializers.glorot_normal()

def l2_reg(a = 0.0001):
	return keras.regularizers.l2(a)

def l1_reg(a = 0.0001):
	return keras.regularizers.l1(a)

#################
#################
#################


##############
### MODELS ###
##############


def build_simple_model(input_dim, model_parameters, ker_init = glorot_normal(), bias_init = 'zeros', ker_reg = l2_reg(), bias_reg = l2_reg()):
	model = Sequential()
	# Input layer
	model.add(Dense(model_parameters[0][0],input_shape=(input_dim,),activation='tanh',kernel_initializer = ker_init, bias_initializer = bias_init, kernel_regularizer = ker_reg, bias_regularizer = bias_reg))
	model.add(Dropout(model_parameters[0][1]))
	# Intermediate layers
	for i in range(1,len(model_parameters)-1):
		model.add(Dense(model_parameters[i][0],activation='tanh',kernel_initializer = ker_init, bias_initializer = bias_init, kernel_regularizer = ker_reg, bias_regularizer = bias_reg))
		model.add(Dropout(model_parameters[i][1]))
	# Output layer
	model.add(Dense(model_parameters[-1],activation='linear',kernel_initializer = ker_init, bias_initializer = bias_init))
	# Optimizer
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')
	return model


###### MODEL PARAMETERS #####

# The model parameters should be passed in this way: a list of tuples (where the list index correspond to layers one) 
# containing as first argument of each tuple the number of neurons and as second the dropout value of that layer
# the last instance of the list should be just a number since no dropout is expected for the last layer

best_model_parameters = [(50,0),(25,0.2),(10,0),2]

#############################

##############
##############
##############

###########################
##### DATA ACQUISITION ####
###########################


if (reshuffle_data==True):
	input_data = np.load('../data/OTRS_input.npy')
	emittance_data = np.load('../data/OTRS_emittance_output.npy')

	train_input, test_input, train_output, test_output = train_test_split(input_data,emittance_data,test_size=0.2, shuffle=True)
	np.save('../data/train_input.npy',train_input)
	np.save('../data/train_output.npy',train_output)
	np.save('../data/test_input.npy',test_input)
	np.save('../data/test_output.npy',test_output)

	#adding Gaussian noise to the training set, you can set the mean and the std_dev of the distribution by changing the parameters
	#the factor 0.02 is the range width of the normalized data  
	if (add_noise==True):
		mean = 0
		std_dev = 1 
		noise_amount = 1	
		for i in range(noise_amount):
			x_noise = train_input + np.random.normal(mean,std_dev,(train_input.shape)) * 0.02
			y_noise = train_output + np.random.normal(mean,std_dev,(train_output.shape)) * 0.02

			train_input = np.concatenate((train_input,x_noise),axis=0)
			train_output = np.concatenate((train_output,y_noise),axis=0)

			np.save('../data/train_input.npy',train_input)
			np.save('../data/train_output.npy',train_output)



else:
	train_input = np.load('../data/train_input.npy')
	train_output = np.load('../data/train_output.npy')
	test_input = np.load('../data/test_input.npy')
	test_output = np.load('../data/test_output.npy')


###########################
###########################
###########################


###########################
##### MODEL IN ACTION #####
###########################


results_on_train = []
results_on_test = []


size_of_the_ensable_of_neural_networks = 2

for i in range(size_of_the_ensable_of_neural_networks):
	#callback to the TensorBoard -> tensorboard --logdir path_to_current_dir/name_dir  #To use TensorBoard
	#tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=100, batch_size=150, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=True)
	tensorboard = keras.callbacks.TensorBoard(log_dir='../results/TensorBoard', histogram_freq=0, batch_size=150, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=False)

	#build and train the model
	model = build_simple_model(len(train_input[0,1:]), best_model_parameters,ker_reg=l2_reg(0),bias_reg=l2_reg(0))
	model.fit(train_input[:,1:], train_output, validation_data=(test_input[:,1:],test_output), epochs=2000, batch_size=150,callbacks=[tensorboard])
	
	#save the entire model
	model.save('../results/models/model_'+str(i+1)+'.h5',model)

	#here the results on the train set and the prediction on the test set are saved into a list
	#and at the same time they are ordered in magnitude to better understand the goodness of model 
	train_input_emittance_x_axis = train_input[train_output[:,0].argsort()]
	output_on_train = model.predict(train_input_emittance_x_axis[:,1:],verbose=1)
	results_on_train.append(output_on_train)

	test_input_emittance_x_axis = test_input[test_output[:,0].argsort()]
	output_on_test = model.predict(test_input_emittance_x_axis[:,1:],verbose=1)
	results_on_test.append(output_on_test)


results_on_train = np.asarray(results_on_train)
np.save('../results/ensamble_results_on_train.npy',results_on_train)
results_on_test = np.asarray(results_on_test)
np.save('../results/ensamble_predictions_on_test.npy',results_on_test)


###########################
###########################
###########################