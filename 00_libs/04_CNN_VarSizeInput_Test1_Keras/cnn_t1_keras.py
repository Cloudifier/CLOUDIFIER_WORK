# -*- coding: utf-8 -*-
"""
@Created on Thu Jul 20 17:24:28 2017

@lastmodified 2017-10-04

@author: High Tech Systems and Software

@project: Cloudifier.NET

@sub-project: Deep Convolutional Network for Variable Size Image Recognition

@description: 
  KERAS developed model based on Model API (non-sequencial). 
  Architecture and specifications developed by Cloudifier team
  Code implemented and tested by HTSS

@copyright: CLOUDIFIER SRL

Top architectures:
  
BEST:             32x2_64x2_512x2_GMP_512d_512d      7    0.992   0.988
             
  16x2_32x2_64x2_128x2_256x2_GMP_512d_512d     10    0.988   0.988
            16x2_32x2_128(1)x1_GMP_512_512      2    0.981   0.990  


TEST1:
  
                                      Layout  Model  TestAcc  VarAcc
5           32x2_64x2_512(1)x1_GMP_512d_512d      5    0.980   0.928
4       NOB_16x2_32x2_128(3)x1_GMP_512d_512d      4    0.986   0.940
3         NOB_16x2_32x2_128(1)x1_GMP_512_512      3    0.977   0.958
1       NOB_16x2_32x2_128(1)x1_GMP_512d_512d      1    0.986   0.960
0           16x2_32x2_128(1)x1_GMP_512d_512d      0    0.981   0.970
8               16x2_32x2_64x2_GMP_512d_512d      8    0.985   0.972
10  16x2_32x2_64x2_128x2_256x2_GMP_512d_512d     10    0.979   0.972
2             16x2_32x2_128(1)x1_GMP_512_512      2    0.982   0.974
9   16x2_32x2_64x2_128x2_256x1_GMP_512d_512d      9    0.984   0.978
6           32x2_64x2_512(3)x1_GMP_512d_512d      6    0.985   0.984
7              32x2_64x2_512x2_GMP_512d_512d      7    0.992   0.998


TEST2:
  
                                      Layout  Model  TestAcc  VarAcc
4       NOB_16x2_32x2_128(3)x1_GMP_512d_512d      4    0.979   0.898
5           32x2_64x2_512(1)x1_GMP_512d_512d      5    0.979   0.962
1       NOB_16x2_32x2_128(1)x1_GMP_512d_512d      1    0.982   0.966
8               16x2_32x2_64x2_GMP_512d_512d      8    0.986   0.980
6           32x2_64x2_512(3)x1_GMP_512d_512d      6    0.990   0.984
9   16x2_32x2_64x2_128x2_256x1_GMP_512d_512d      9    0.988   0.986
0           16x2_32x2_128(1)x1_GMP_512d_512d      0    0.986   0.988
3         NOB_16x2_32x2_128(1)x1_GMP_512_512      3    0.978   0.988
7              32x2_64x2_512x2_GMP_512d_512d      7    0.992   0.988
10  16x2_32x2_64x2_128x2_256x2_GMP_512d_512d     10    0.988   0.988
2             16x2_32x2_128(1)x1_GMP_512_512      2    0.981   0.990  


TEST3: 

4  16x2->32x2->64x2->128x2->256(3)x2->GMP->512d->...      4    0.963   0.964
2              32x2->64x2->512(1)x2->GMP->512d->512d      2    0.982   0.976
0                16x2->32x2->128(1)x1->GMP->512->512      0    0.983   0.978
1                16x2->32x2->128(3)x1->GMP->512->512      1    0.983   0.978
5  16x2->32x2->64x2->128x2->256(1)x2->GMP->512d->...      5    0.991   0.986
3              32x2->64x2->512(3)x2->GMP->512d->512d      3    0.985   0.992  



TEST4: 3ep

                                             Layout  Model  TestAcc  VarAcc
2           B0_32x2->64x2->512(3)x2->GMP->512d->512d      2    0.980   0.864
3  B0_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      3    0.985   0.932
1  B1_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      1    0.980   0.958
4           B2_32x2->64x2->512(3)x2->GMP->512d->512d      4    0.987   0.974
0           B1_32x2->64x2->512(3)x2->GMP->512d->512d      0    0.989   0.978
5  B2_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      5    0.989   0.980
  
  
TEST5: 3 ep
  
                                              Layout  Model  TestAcc  VarAcc
3  B0_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      3    0.973   0.918
2           B0_32x2->64x2->512(3)x2->GMP->512d->512d      2    0.980   0.950
5  B2_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      5    0.981   0.972
4           B2_32x2->64x2->512(3)x2->GMP->512d->512d      4    0.982   0.978
0           B1_32x2->64x2->512(3)x2->GMP->512d->512d      0    0.985   0.986
1  B1_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      1    0.988   0.988  

TEST6 20 EPOCHS:
  
                                              Layout  Model  TestAcc  VarAcc
2           B0_32x2->64x2->512(3)x2->GMP->512d->512d      2    0.987   0.886
3  B0_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      3    0.984   0.952
1  B1_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      1    0.983   0.966
0           B1_32x2->64x2->512(3)x2->GMP->512d->512d      0    0.985   0.970
4           B2_32x2->64x2->512(3)x2->GMP->512d->512d      4    0.984   0.978
5  B2_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      5    0.987   0.982  

TEST7 20 epochs LR_plateau

                                              Layout  Model  TestAcc  VarAcc
2           B0_32x2->64x2->512(3)x2->GMP->512d->512d      2    0.989   0.956
3  B0_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      3    0.993   0.982
1  B1_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      1    0.993   0.988
5  B2_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      5    0.994   0.990
4           B2_32x2->64x2->512(3)x2->GMP->512d->512d      4    0.993   0.992
0           B1_32x2->64x2->512(3)x2->GMP->512d->512d      0    0.994   0.996


TEST8 20 epochs LR_plateau
                                              Layout  Model  TestAcc  VarAcc
3  B0_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      3    0.990   0.982
2           B0_32x2->64x2->512(3)x2->GMP->512d->512d      2    0.991   0.988
1  B1_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      1    0.991   0.990
5  B2_16x2->32x2->64x2->128x2->256(1)x2->GMP->512...      5    0.993   0.990
0           B1_32x2->64x2->512(3)x2->GMP->512d->512d      0    0.994   0.994
4           B2_32x2->64x2->512(3)x2->GMP->512d->512d      4    0.995   0.996

"""

import numpy as np


from keras.utils import np_utils
from keras.datasets import mnist


from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from tqdm import tqdm

import matplotlib.pyplot as plt

from logger import Logger

from fcn_builder import BuildModel, BuildModelBlocks

import pandas as pd

from PIL import Image


if __name__=="__main__":
  
  
  log = Logger(lib_name = "FCNLIB")
  
  np.set_printoptions(formatter={'float':'{: 0.2f}'.format})
  
  use_pooling = False
  
  if log.GetMachineName() !="DAMIAN":
    nr_epochs = 20
    batch_size = 128
  else:
    nr_epochs = 1
    batch_size = 32
    
  img_h = 28
  img_w = 28
  nr_ch=1
  nr_classes = 10
  

  models_block_defs = [


      ("B1_32x2->64x2->512(3)x2->GMP->512d->512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":512, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),


       ("B1_16x2->32x2->64x2->128x2->256(1)x2->GMP->512d->512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK4", "TYPE":"CONV", "NRLY":2, "VALU":128, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK5", "TYPE":"CONV", "NRLY":2, "VALU":256, "KERN":(1,1), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIRAMID_LIKE", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),  


      ("B0_32x2->64x2->512(3)x2->GMP->512d->512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":0, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":0, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":512, "KERN":(3,3), "BATN":0, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),


       ("B0_16x2->32x2->64x2->128x2->256(1)x2->GMP->512d->512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":0, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":0, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":0, "ACTV":"elu"},
          {"NAME":"CBLOCK4", "TYPE":"CONV", "NRLY":2, "VALU":128, "KERN":(3,3), "BATN":0, "ACTV":"elu"},
          {"NAME":"CBLOCK5", "TYPE":"CONV", "NRLY":2, "VALU":256, "KERN":(1,1), "BATN":0, "ACTV":"elu"},
          {"NAME":"PIRAMID_LIKE", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),  


      ("B2_32x2->64x2->512(3)x2->GMP->512d->512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":512, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),


       ("B2_16x2->32x2->64x2->128x2->256(1)x2->GMP->512d->512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK4", "TYPE":"CONV", "NRLY":2, "VALU":128, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK5", "TYPE":"CONV", "NRLY":2, "VALU":256, "KERN":(1,1), "BATN":2, "ACTV":"elu"},
          {"NAME":"PIRAMID_LIKE", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),  


      ]

      
  block_defs = True
  
  models_log = [x for x,_ in models_block_defs]
  log.VerboseLog("Models:")
  for x in models_log:
    log.VerboseLog("  {}".format(x))
  
  if block_defs:
    models_list = models_block_defs
  else:
    models_list = []
  
  (X_train, y_tr), (X_test, y_ts) = mnist.load_data()
  X_train = X_train.astype("float32") / 255
  X_test  = X_test.astype("float32") / 255
  if X_train.ndim != 4:
    nr_train_obs = X_train.shape[0]
    nr_test_obs = X_test.shape[0]
    train_img_h = X_train.shape[1]
    train_img_w = X_train.shape[2]
    X_train = X_train.reshape(nr_train_obs, train_img_h, train_img_w, nr_ch)
    X_test = X_test.reshape(nr_test_obs, train_img_h, train_img_w, nr_ch)

  np.set_printoptions(suppress = True, edgeitems = 5, linewidth = 100)

  y_train = np_utils.to_categorical(y_tr, nr_classes)
  y_test = np_utils.to_categorical(y_ts, nr_classes)  
  
  # random image preparation
  nr_tests = 5 # 5000
  np_idx = np.random.randint(0,9000,size = nr_tests)
  test_images = list()  
  x_samples = X_test[np_idx]
  y_samples = y_test[np_idx]
  
  max_acc_value = 0
  max_acc_name = ""
  max_acc_test = 0

  QUICK_TEST = True
  
  FULL_DISPLAY = True
  
  log.VerboseLog("Preparing test dataset...")
  for i in range(np_idx.shape[0]):
    if FULL_DISPLAY:
      log.VerboseLog("Initial image with label {},{}".format(np.argmax(y_samples[i]),y_samples[i]))
      plt.figure()
      plt.matshow(x_samples[i,:,:,:].reshape(img_h,img_w), cmap="gray")
      plt.show()
      
    new_h = np.random.randint(150,200)
    new_w = np.random.randint(150,200)
    test_img = np.zeros(shape=(new_h,new_w))
    src_h = x_samples.shape[1]
    src_w = x_samples.shape[2]
    src_img = x_samples[i,:,:,0]
    new_scale = np.random.randint(1,8) / 2.5 # generate scaling factors between 40% and 300
    new_s = int(src_h * new_scale)
    pos_r = np.random.randint(0, new_h-new_s-1)
    pos_c = np.random.randint(0, new_w-new_s-1)

    img_new = np.array(Image.fromarray(src_img).resize((new_s,new_s)))      
    
    test_img[pos_r:(pos_r+new_s), pos_c:(pos_c+new_s)] = img_new
    
    test_img = test_img.reshape(1,new_h,new_w,1)
    test_images.append(test_img)
    if FULL_DISPLAY:
      log.VerboseLog("New image scaled with {:.1f}: on {}x{} scene".format(new_scale, new_h,new_w))
      plt.figure()
      plt.matshow(test_img.reshape(new_h,new_w), cmap = "gray")
      plt.show()
  
  # now train and predict !!!
  log.VerboseLog("Done preparing test dataset.")
  
 
  model_slice = None
  if (log.MACHINE_NAME == "DAMIAN") or QUICK_TEST:
    model_slice = 1
    
  
  selected_models = models_list[:model_slice]
  log.VerboseLog("Training/testing a total of {} models".format(len(selected_models)))
  
  nr_models = len(selected_models)
  i = 0
  model_name_list = []
  model_list = []
  test_list = []
  vtest_list = []
  model_ids = []
  for model_name, model_def in selected_models:
    model_name_list.append(model_name)
    model_ids.append(i)
    i += 1
    if block_defs:
      log.VerboseLog("\nPreparing FCN ({}/{}): {} using model blocks definition ".format(
          i,nr_models,model_name))
      model, desc = BuildModelBlocks(model_def)
    else:
      log.VerboseLog("\nPreparing FCN ({}/{}): {} using simple model definition ".format(
          i,nr_models,model_name))
      model, desc = BuildModel(model_def)
      
    model.compile(optimizer = "adam", loss = "categorical_crossentropy",
                  metrics=["accuracy"])
  
    str_model = log.GetKerasModelSummary(model, full_info=False)
    log.VerboseLog("Short description:\n{}".format(desc))
    log.VerboseLog(str_model)
  
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    lr_monitor = ReduceLROnPlateau(factor = 0.9, patience = 1, verbose = 1)
    
  
    final_nr_epochs = nr_epochs

    log.VerboseLog("Training FCN ({}/{})for {} epochs...".format(
        i,nr_models,final_nr_epochs))

    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs = final_nr_epochs, batch_size= batch_size, verbose = 1,
              callbacks=[
                          lr_monitor,
                          #early_stopping,
                         ],
              #validation_split=0.15
              )
    
    model_list.append(model)
  
    score = model.evaluate(X_test, y_test, verbose=1)
    log.VerboseLog('Test score:{:.3f}'.format(score[0]))
    log.VerboseLog('Test accuracy:{:.3f}'.format(score[1]))
    
    test_list.append(score[1])
   
    # now the real test !!!
  
  
    preds = []
    for im_idx in tqdm(range(len(test_images))):
      test_img = test_images[im_idx]
      pred = model.predict(test_img).ravel()
      preds.append(np.argmax(pred))
      y_t = np.argmax(y_samples[im_idx])
      p_t = np.argmax(pred)
      res = y_t == p_t
      if (not res):
        np.set_printoptions(formatter={'float':'{: 0.2f}'.format}) 
        log.Log("Label/Prediction: {}/{} Correct: {} Imagesize: {}".format(
            y_t,p_t,res, test_img.shape))
        log.Log("  Prediction: {}".format(pred))
        log.Log("  y_test:     {}".format(np.array(y_samples[im_idx],
                                             dtype=np.float32)))
        new_h = test_img.shape[1]
        new_w = test_img.shape[2]
        log.OutputImage(test_img.reshape(new_h,new_w), "{}_WRONG_LABEL_{}_{}".format(model_name,y_t, p_t))

    var_size_acc = np.sum(preds==np.argmax(y_samples,1)) / y_samples.shape[0]
    vtest_list.append(var_size_acc)
    log.VerboseLog("Variable size accuracy: {:.3f} (test {:.3f})for {}".format(
            var_size_acc, score[1], model_name), results = True)
    if max_acc_value < var_size_acc:
      max_acc_value = var_size_acc
      max_acc_name = model_name
      max_acc_test = score[1]
      best_model = model
           
  if False: 
    log.ShowNotPrinted()  
  
  log.VerboseLog("\nFinal results:")
  log.ShowResults()
  pd.set_option("display.precision", 3)
  df_res = pd.DataFrame({"Model"   : model_ids,
                         "Layout"  : model_name_list, 
                         "VarAcc"  : vtest_list, 
                         "TestAcc" : test_list}).sort_values("VarAcc")
  log.VerboseLog("Results table:\n{}".format(df_res))
  log.VerboseLog("\nBest accuracy {:.3f} for model {} with test acc: {:.3f}".format(
                  max_acc_value, max_acc_name, max_acc_test))
  
  
  
  
  