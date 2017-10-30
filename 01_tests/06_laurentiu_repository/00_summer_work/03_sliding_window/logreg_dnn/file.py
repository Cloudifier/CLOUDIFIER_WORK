for i in range(1):
	plt.matshow(slider.scenes[i][0].reshape(40, 80), cmap = 'gray')
	plt.title("Scena 40x80")

	plt.matshow(slider.correct_windows[i][0], cmap = 'gray')
	plt.title("Fereastra corecta")

	'''
	plt.matshow(slider.model[1:, slider.scenes[i][1]].reshape(28, 28) * slider.correct_windows[i][0], cmap = 'gray')
	plt.colorbar()
	plt.axis('off')
	plt.title("Matrice activare fereastra corecta/theta target corect")
	'''

	plt.matshow(slider.windows[i][0], cmap = 'gray')
	plt.title("Fereastra identificata")

	'''
	plt.matshow(slider.model[1:, slider.scenes[i][1]].reshape(28, 28) * slider.windows[i][0], cmap = 'gray')
	plt.colorbar()
	plt.axis('off')
	plt.title("Matrice activare fereastra identificata/theta target corect")

	plt.matshow(slider.model[1:, slider.windows[i][1]].reshape(28, 28) * slider.windows[i][0], cmap = 'gray')
	plt.colorbar()
	plt.axis('off')
	plt.title("Matrice activare fereastra identificata/theta target prezis")
	'''

	print("Pozitie:{}, Target:{}".format(slider.scenes[i][2], slider.scenes[i][1]))
	print("Valoare prezisa pe fereastra corecta:{}, probabilitate:{}".
		format(slider.correct_windows[i][1], slider.correct_windows[i][2]))
	print("Valoare prezisa:{}, probabilitate:{}, pozitie:{}".
		format(slider.windows[i][1], slider.windows[i][2], slider.windows[i][3]))
	print()