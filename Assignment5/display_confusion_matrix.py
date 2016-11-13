import numpy as np
import matplotlib.pyplot as plt

def display_confusion_matrix(conf_arr):
	norm_conf = []
	for i in conf_arr:
		a = 0
		tmp_arr = []
		a = sum(i, 0)
		for j in i:
			tmp_arr.append(float(j)/float(a))
		norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
					interpolation='nearest')

	width, height = conf_arr.shape

	for x in xrange(width):
		for y in xrange(height):
			ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
						horizontalalignment='center',
						verticalalignment='center')

	#cb = fig.colorbar(res)
	alphabet = '0123456789'
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	#plt.savefig('confusion_matrix.png', format='png')
	plt.show()
	
