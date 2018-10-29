import numpy as np
import scipy.io as sio
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr
from datetime import datetime,timedelta
from matplotlib.colors import ListedColormap

### This algorithm load the performance of the ensamble of neural networks trained (both on the train and on the test set) and compare them with the expected results
### The emittance values in the plots are ordered in magnitude to better compare them with the expected results


sns.set_style('darkgrid')


pred = np.load('../results/ensamble_results_on_train.npy')
y_train = np.load('../data/train_output.npy')

print(y_train.shape)
print(pred.shape)


mean_pred=np.mean(pred,axis=0)

y_train=y_train[y_train[:,0].argsort()]
y_1 = [a - 0.05 for a in y_train[:,0]]
y_2 = [a + 0.05 for a in y_train[:,0]]


plt.title('Emittance Values x-axis (train data)',fontsize=45)
plt.xlabel('samples number',fontsize=35)
plt.ylabel('emittance values',fontsize=35)
plt.plot(mean_pred[:,0],'o',c='k',label='mean prediction')
plt.plot(y_train[:,0],linewidth=3,linestyle='--',color='crimson',label='train data')
plt.fill_between(range(len(y_train[:,0])),y_1,y_2,color='crimson',interpolate=True,alpha=1)
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)  
plt.legend(loc='best',fontsize=30)
plt.show()


pred = np.load('../results/ensamble_predictions_on_test.npy')
y_test = np.load('../data/test_output.npy')


print(y_test.shape)
print(pred.shape)


mean_pred=np.mean(pred,axis=0)

y_test=y_test[y_test[:,0].argsort()]
y_1 = [a - 0.05 for a in y_test[:,0]]
y_2 = [a + 0.05 for a in y_test[:,0]]

print(y_test.shape)
print(mean_pred.shape)




plt.title('Emittance Values x-axis (test data)',fontsize=45)
plt.xlabel('samples number',fontsize=35)
plt.ylabel('emittance values',fontsize=35)
plt.plot(mean_pred[:,0],'o',c='k',label='prediction')
plt.plot(y_test[:,0],linewidth=3,linestyle='--',color='crimson',label='test data')
plt.fill_between(range(len(y_test[:,0])),y_1,y_2,color='crimson',interpolate=True,alpha=1)
plt.tick_params(axis = 'both', which = 'major', labelsize = 20)  
plt.legend(loc='best',fontsize=30)
plt.show()
