import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class MnistProcesser:
  def __init__(self):
    """
    Incarcati setul de date MNIST. Acesta este format din 70000 de imagini cu cifre scrise de mana.
    Dimensiunea imaginilor este 28x28.
    Datele pe care le veti gasi in mnist['data'] sunt efectiv pixelii imaginilor, iar target-ul
    fiecarei imagini este de fapt cifra.
    Inainte de a incepe lucrul, va indemnam sa explorati setul de date intr-un notebook pentru a 
    vedea 'forma' setului de date MNIST (la fel cum ati procedat si pentru IRIS).
    """
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home = '.')
  
    self.X = mnist['data']
    self.y = mnist['target'] 
    print('Loaded {} images which contain {} pixels'.format(self.X.shape[0], self.X.shape[1]))
  
  def PlotRandomImage(self, digit_list):
    """
    digit_list reprezinta o lista cu cifre pentru care voi trebuie sa faceti graficul folosind 
    imshow (plt.imshow). In MNIST, evident, exista mai multe imagini in care este reprezentata o
    anumita cifra. Voi trebuie sa alegeti una random. De exemplu:
    digit_list = [0, 5, 9]. Se va itera prin lista pentru a procesa fiecare cifra C. Din setul de date
    se aleg doar imaginile in care sunt desenate cifre C. Dupa asta, alegeti random o intrare din cele selectate.
    
    Salvati imaginile cu un nume sugestiv
    """
    import random
    for digit in digit_list:
      digit_idx = np.where(self.y == digit)
      selected_digits = self.X[digit_idx]
      print("We found {} images that represent digit {}".format(selected_digits.shape[0], digit))
      img = random.choice(selected_digits)
      plt.imshow(img.reshape(28, 28))
      plt.savefig('{}.png'.format(digit))
  
  def FindPredictions(self):
    """
    Selectati primele 13 imagini din setul de date (X). Incarcati fisierul cu extensia .npy (np.load).
    Inmultiti matricea formata din pixelii primelor 13 imagini cu matricea incarcata din fisierul
    'theta.npy'.
    """
    selected = self.X[:13, :]
    theta = np.load('theta.npy')
    _result = selected.dot(theta.T)
    return _result
  
  def CreateScene(self, size):
    """
    Alegeti orice imagine din setul de date (X). Creati un 2-D numpy array plin cu 0-uri de dimensiune
    size x size. Plasati imaginea selectata in scena creata la o pozitie aleatoare.
    """
    import random
    small_img = random.choice(self.X).reshape(28, 28)
    scene = np.zeros((size, size))
    
    pos_X = random.randint(0, size - 28)
    pos_Y = random.randint(0, size - 28)
    scene[pos_Y:pos_Y + 28, pos_X:pos_X + 28] = small_img
    return scene
  
  def GetFavoritePixels(self, config_file = 'config.txt'):
    """
    In fisierul de configurare exista un JSON care contine 7 perechi cheie: valoare. Valorile reprezinta
    efectiv pixelii pe care trebuie sa ii selectati pentru fiecare imagine din setul de date (X).
    Odata creat array-ul numpy cu acesti pixeli, va trebui sa creati un DataFrame Pandas care sa contina
    acesti pixeli (numele coloanelor trebuie sa fie acelasi cu cheile mentionate in JSON).
    """
    import json
    f = open(config_file, 'r')
    keys = []
    values = []
    config_data = json.load(f)
    for key, value in config_data.items():
      keys.append(key)
      values.append(value)
    print("JSON file contains the following pairs (key, value): {}".format(list(zip(keys, values))))
    
    selected = self.X[:, values]
    df = pd.DataFrame(selected, columns = keys)
    return df
  
if __name__ == '__main__':
  print("TASK #1")
  mnist_processer = MnistProcesser()
  
  print("TASK #2")
  dl1 = [0, 5, 9]
  dl2 = [4, 1, 2, 3, 8]
  mnist_processer.PlotRandomImage(digit_list = dl1)
  mnist_processer.PlotRandomImage(digit_list = dl2)
  
  print("TASK #3")
  magic_matrix = mnist_processer.FindPredictions()
  
  print("TASK #4")
  scene = mnist_processer.CreateScene(size = 100)
  
  print("TASK #5")
  df = mnist_processer.GetFavoritePixels()