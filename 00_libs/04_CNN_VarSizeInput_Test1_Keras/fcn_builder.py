# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:24:28 2017

@module: Keras model builder (grid-search helper)

@author: High Tech Systems and Software

@project: Cloudifier.NET

@sub-project: Deep Convolutional Network for Variable Size Image Recognition

@description: 
  KERAS model design library based on Model API (non-sequencial). 
  Architecture and specifications developed by Cloudifier team
  Code implemented and tested by HTSS

@copyright: CLOUDIFIER SRL

"""

from keras.layers import  Conv2D, Dropout, Dense, Input, Activation
from keras.layers import GlobalMaxPooling2D, BatchNormalization
from keras.models import Model


def GetValMaybe(data,skey):
  res = None
  if skey in data.keys():
    res = data[skey]
  return res

def BuildModel(model_definition):
  """
  DEPRECATED: USE BLOCK BUILDER
  
  Model builder for Fully Convolutional Networks.
  Accepts list of layers. each layer is a dictionary with type, kernel, value
  
  "t": "inp" - input, "c2d" - conv2d, "gmp" - GlobalMaxPooling, 
       "drp" - Dropout, "dns" - Dense
  "k": (h,w) touple only for c2d
  "p": string padding for Conv layers
  "s": strides touple for Conv layers
  "v": single value or touple. for different types of layers could be depth
       nr hiddens or for drop = rate
  "a": string denoting activation (optional)
  "i": initialization procedure (string) for weights (optional)
  
  returns a touple containing the model and its architecture
  
  example:
        [
          {"t":"inp","v":(None, None,nr_ch)},
          {"t":"c2d","v":16,"k":(4,4), "a":"elu"},
          {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
          {"t":"drp","v":0.5},
          {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
          {"t":"drp","v":0.5},
          {"t":"c2d","v":512,"k":(4,4),"a":"elu"},
          {"t":"gmp"},
          {"t":"drp","v":0.5},
          {"t":"dns", "v":10, "a":"softmax"}
        ]    
  """
  layers = list()
  nr_layers = len(model_definition)
  ldesc = list()
  last_layer = None
  
  for i in range(nr_layers):
    layer_def = model_definition[i]
    layer_type = layer_def["t"]
    if i==0: # we deal with first layer
      assert layer_type == "inp", "Model is not Sequential, first layer must be Input"
      input_layer = Input(shape=layer_def["v"])
      last_layer = input_layer
      ldesc.append("Input {}".format(layer_def["v"]))
    else:
      val = GetValMaybe(layer_def,"v")
      kernels = GetValMaybe(layer_def,"k")
      activ = GetValMaybe(layer_def,"a")
      init = GetValMaybe(layer_def,"i")
      pad = GetValMaybe(layer_def,"p")
      stride = GetValMaybe(layer_def,"s")
      if init==None:
        init = "he_normal"
      if pad == None:
        pad = "same"
      if stride == None:
        stride = (1,1)
      if layer_type == "c2d":
        last_layer = Conv2D(filters = val, kernel_size=kernels, padding = pad,
                             strides = stride, kernel_initializer=init, 
                             activation=activ)(last_layer)
        ldesc.append("Conv2D  [depth:{} kernel:{} stride:{} pad:{} init:{} activ:{}]".format(
                              val, kernels, stride, pad, init, activ))
      elif layer_type == "gmp":
        last_layer = GlobalMaxPooling2D()(last_layer)
        ldesc.append("GlobalMaxPooling2D")
      elif layer_type == "drp":
        last_layer = Dropout(rate = val)(last_layer)
        ldesc.append("Dropout [rate: {:.2f}]".format(val))
      elif layer_type == "dns":
        last_layer = Dense(units = val, activation = activ, 
                           kernel_initializer=init,
                           )(last_layer)
        ldesc.append("Dense   [unit:{} activ:{} init:{}]".format(val,activ,init))
    layers.append(last_layer)
  
  output_layer = last_layer #layers[-1]
  str_nn = ""
  for l in ldesc:
    str_nn += l +"\n"
  model = Model(inputs = input_layer, outputs=output_layer)    
  return model, str_nn
      

def BuildModelBlocks(model_definition):
  """
  Builds a model from block definition.
  Keywords:
    "NAME": name of the block
    "TYPE": type of the block "INPUT","CONV","PIRAMID","FC", "READOUT"
    "NRLY": number of layers (either conv or fc etc)
    "INIT": optional init of layers
    "KERN": kernel/layer size (touple for conv only)
    "VALU": value of depth for convs or units touples for FC
    "DROP": optional rate of dropout (if DROP exists then dropout is added at 
            end of the block or between all layers for FC block)
    "ACTV": optional activation of each layer in block
    "BATN": 0/1/2 optional batch normalization after each layer (1 = before activation, 2= after)    
  example:
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":0, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":512, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]          
  """
  layers = list()
  ldesc = list()
  nr_blocks = len(model_definition)  
  for i in range(nr_blocks):
    block_def = model_definition[i]
    block_type = block_def["TYPE"]
    if i==0: # we deal with first layer
      assert block_type == "INPUT", "Model is not Sequential, first block must be Input"
      input_layer = Input(shape=block_def["VALU"])
      last_layer = input_layer
      ldesc.append("Input {}".format(block_def["VALU"]))
    else:
      block_name = GetValMaybe(block_def,"NAME")
      assert block_name != None, "Blocks must have names"
      nr_layers = GetValMaybe(block_def,"NRLY")
      if not(block_type in ["INPUT","PIRAMID", "READOUT"]):
        assert nr_layers != None, "Block {} has no layers nr".format(block_name)
      val = GetValMaybe(block_def,"VALU")
      kernels = GetValMaybe(block_def,"KERN")
      activ = GetValMaybe(block_def,"ACTV")
      init = GetValMaybe(block_def,"INIT")
      pad = GetValMaybe(block_def,"PADD")
      stride = GetValMaybe(block_def,"STRD")
      batchn = GetValMaybe(block_def,"BATN")
      drop = GetValMaybe(block_def,"DROP")
      if init==None:
        init = "he_normal"
      if pad == None:
        pad = "same"
      if stride == None:
        stride = (1,1)
        
      if block_type == "CONV":
        for l in range(nr_layers):
          lname = block_name+"_CONV{}_{}".format(val, l)
          last_layer = Conv2D(filters = val, kernel_size=kernels, padding = pad,
                               strides = stride, kernel_initializer=init,
                               name = lname)(last_layer)
          if batchn == 1:
            last_layer = BatchNormalization()(last_layer)
          if activ != None:
            last_layer = Activation(activ)(last_layer)            
          if batchn == 2:
            last_layer = BatchNormalization()(last_layer)
          ldesc.append("Conv2D  [depth:{} kernel:{} stride:{} pad:{} init:{} batchnorm:{} activ:{}]".format(
                                val, kernels, stride, pad, init, batchn, activ))
          layers.append(last_layer)
        # done all convs now check for dropout
        if drop != None:
          last_layer = Dropout(rate = drop)(last_layer)
          ldesc.append(" Dropout [rate: {:.2f}]".format(drop))
          layers.append(last_layer)
      # done with CONV layers
      elif block_type == "PIRAMID":
        ###
        ### must be implemented properly
        last_layer = GlobalMaxPooling2D()(last_layer)
        ldesc.append("GlobalMaxPooling2D")
        layers.append(last_layer)
      # done with PIRAMID layer
      elif block_type == "FC":
        for l in range(nr_layers):
          lname = block_name+"_FC{}_{}".format(val,l)
          last_layer = Dense(units = val, activation = activ, 
                             kernel_initializer=init,
                             )(last_layer)
          ldesc.append("Dense   [unit:{} activ:{} init:{}]".format(val,activ,init))
          layers.append(last_layer)
          if drop != None:
            last_layer = Dropout(rate = drop)(last_layer)
            ldesc.append(" Dropout [rate: {:.2f}]".format(drop))
            layers.append(last_layer)
      elif block_type =="READOUT":
        last_layer = Dense(units = val, activation = activ, 
                           kernel_initializer=init,
                           )(last_layer)
        ldesc.append("Readout [unit:{} activ:{} init:{}]".format(val,activ,init))
        layers.append(last_layer)
        
              
  output_layer = last_layer #layers[-1]
  str_nn = ""
  for l in ldesc:
    str_nn += l +"\n"
  model = Model(inputs = input_layer, outputs=output_layer)    
  return model, str_nn
    
    
    