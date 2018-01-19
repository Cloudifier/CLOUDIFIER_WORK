import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class MnistProcesser:
  def __init__(self):
    """
    Incarcati setul de date MNIST. Acesta este format din 70000 de imagini cu cifre scrise de mana.
    Datele pe care le veti gasi in mnist['data'] sunt efectiv pixelii imaginilor, iar target-ul
    fiecarei imagini este de fapt cifra.
    Inainte de a incepe lucrul, va indemnam sa explorati setul de date intr-un notebook pentru a 
    vedea 'forma' setului de date MNIST (la fel cum ati procedat si pentru IRIS).
    """

    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home = '.')
  
    #self.X = ....
    #self.y = ....
  
  def PlotRandomImage(self, digit_list):
    """
    digit_list reprezinta o lista cu cifre pentru care voi trebuie sa faceti graficul folosind 
    imshow (plt.imshow). In MNIST, evident, exista mai multe imagini in care este reprezentata o
    anumita cifra. Voi trebuie sa alegeti una random. De exemplu:
    digit_list = [0, 5, 9]. Se va itera prin lista pentru a procesa fiecare cifra C. Din setul de date
    se aleg doar imaginile in care sunt desenate cifre C. Dupa asta, alegeti random o intrare din cele selectate.

    Salvati imaginile cu un nume sugestiv
    """
    pass
  
  def FindPredictions(self):
    """
    Selectati primele 13 imagini din setul de date (X). Incarcati fisierul cu extensia .npy (np.load).
    Inmultiti matricea formata din pixelii primelor 13 imagini cu matricea incarcata din fisierul
    'theta.npy'.
    """
    return None
  
  def CreateScene(self, size):
    """
    Alegeti orice imagine din setul de date (X). Creati un 2-D numpy array plin cu 0-uri de dimensiune
    size x size. Plasati imaginea selectata in scena creata la o pozitie aleatoare.
    """
    return None
  
  def GetFavoritePixels(self, config_file = 'config.txt'):
    """
    In fisierul de configurare exista un JSON care contine 7 perechi cheie: valoare. Valorile reprezinta
    efectiv pixelii pe care trebuie sa ii selectati pentru fiecare imagine din setul de date (X).
    Odata creat array-ul numpy cu acesti pixeli, va trebui sa creati un DataFrame Pandas care sa contina
    acesti pixeli (numele coloanelor trebuie sa fie acelasi cu cheile mentionate in JSON).
    """
    return None
  
if __name__ == '__main__':
  """
  Incercati sa puneti print-uri sugestive pe masura ce rezolvati task-urile propuse
  """
  
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