import numpy as np 
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.colors import LogNorm
from scipy.stats import norm

#Define and train 2D GMM 
#This one works badly...
def gmm_2D(FITC, cherry, confidence):
	gmm = mixture.GaussianMixture(n_components = 3, covariance_type='full')
	data = np.hstack([FITC,cherry])
	#print data
	gmm.fit(data)
	prob = gmm.predict_proba(data)
	label = np.argmax(prob, axis=1)
	label = label.reshape(label.size,1)
	
	zipped = np.hstack([data,label])
	#print zipped
	return zipped

def label_infection(data, model):
	labels = model.predict(data)
	return labels

def extract_infected(data, confidence):
	gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
	gmm.fit(data)
	#score = -gmm.score_samples(data)
	#print np.amin(score), np.amax(score)
	prob = gmm.predict_proba(data)
	zipped = np.concatenate((data, prob), axis=1)

	fig = plt.figure()
	plt.plot(data, prob[:,1],'o')
	filename = 'prob_test.png'
	#print('Processing position ' + str(pos) + ' frame ' + str(frame))
	plt.savefig('/Users/nicolasquach/Documents/stanford/covert_lab/deep_learning/plots/3.20.17/'+filename, format = 'png')
	plt.close(fig)

	infected = []

	for i in range(np.shape(zipped)[0]):
		if zipped[i,2] >= confidence:
			infected.append(zipped[i,0])

	return infected


def gaussian_classifier(no_infection, infection, confidence):
	#print infection
	mean = np.mean(no_infection)
	std = np.std(no_infection)

	prob = norm(mean, std).pdf(infection)
	zipped = np.concatenate((infection, prob), axis=1)


	
	fig = plt.figure()
	plt.plot(infection, prob,'o')
	filename = 'prob_test.png'
	#print('Processing position ' + str(pos) + ' frame ' + str(frame))
	plt.savefig('/Users/nicolasquach/Documents/stanford/covert_lab/deep_learning/plots/5.2.17/'+filename, format = 'png')
	plt.close(fig)
	

	infected = []

	for i in range(np.shape(zipped)[0]):
		if zipped[i,1] <= (1-confidence) and zipped[i,0] >= mean:
			infected.append(zipped[i,0])

	return infected




