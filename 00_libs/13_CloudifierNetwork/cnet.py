# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 08:38:57 2018

@author: Andrei Ionut Damian
"""

from keras.layers import (
                          Conv2D, 
                          Conv2DTranspose,
                          Dense,
                          BatchNormalization, 
                          Activation, 
                          Concatenate, 
                          SeparableConv2D, 
                          Input,
                          GlobalAveragePooling2D,
                          GlobalMaxPooling2D,
                          Lambda,
                          add,
                          )

from keras.models import Model
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

__version__   = "0.1"
__author__    = "Cloudifier"
__copyright__ = "(C) Cloudifier SRL"
__project__   = "Cloudifier"  
__module__    = "CloudifierNet"
__reference__ = "Based on InceptionV3, ResNet and Xception architecures"

import os

def load_module(module_name, file_name):
  """
  loads modules from _pyutils Google Drive repository
  usage:
    module = load_module("logger", "logger.py")
    logger = module.Logger()
  """
  from importlib.machinery import SourceFileLoader
  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "GoogleDrive"),
                 os.path.join("D:/", "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    logger_lib = None
    print("Logger library not found in shared repo.", flush = True)
    #raise Exception("Couldn't find google drive folder!")
  else:  
    utils_path = os.path.join(drive_path, "_pyutils")
    print("Loading [{}] package...".format(os.path.join(utils_path,file_name)),flush = True)
    logger_lib = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
    print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)),flush = True)

  return logger_lib

class SimpleLogger:
  def __init__(self):
    return
  def VerboseLog(self, _str, show_time):
    print(_str, flush = True)

def LoadLogger(lib_name, config_file):
  module = load_module("logger", "logger.py")
  if module is not None:
    logger = module.Logger(lib_name = lib_name, config_file = config_file)
  else:
    logger = SimpleLogger()
  return logger

class CloudifierNet:
  
  def __init__(self, n_classes, input_shape, 
               ver='v1', pooling='max',
               logger=None, DEBUG=False,
               direct_skip=False):
    """
    """
    self.ver = ver
    self.pooling = pooling
    self._block_count = 0
    self.DEBUG = DEBUG
    self.n_classes = n_classes
    self.input_shape = input_shape
    self.direct_skip = direct_skip
    self.short_desc = []
    if logger is None:
      self.logger = LoadLogger('CNET','config.txt')

    if K.image_data_format() == 'channels_first':
      self.channel_axis = 1
    else:
      self.channel_axis = 3    
    self.log("Init {}...".format(__module__))

    builders = {
        "v1" : self._build_network_v1,
        "v2" : self._build_network_v2,
        "v3" : self._build_network_v3,
        }
    
    if ver not in builders.keys():
      self.log("Architecture '{}' not known".format(ver))
      raise

    self.builder = builders[ver]
    
    self.log_builder()
       
    self.builder(input_shape=self.input_shape, 
                 n_classes=self.n_classes,
                 simple_compile=True,
                 pooling=self.pooling,
                 direct_skip=self.direct_skip
                 )
    self.log_architecture()
    if self.DEBUG:
      self.log(self.logger.GetKerasModelSummary(self.model))
    return
  
  def log_builder(self):
    self.log("Building model: {} pooling: {}".format(self.ver,self.pooling))
    return
  
  def log_architecture(self):
    self.log("Simple model architecure ({:,} params):".format(
        self.model.count_params()))
    for desc in self.short_desc:
      self.log("  "+desc)
    self.log("Model layers:")
    self.log("  Input_name:{} shape:{}".format(
             self.input_tensor.name,
             K.int_shape(self.input_tensor)))
    self.log("  Fin_conv_name:{} shape:{}".format(
             self.final_conv_layer.name,
             K.int_shape(self.final_conv_layer)))
    self.log("  Fin_pool_name:{} shape:{}".format(
             self.final_pool_layer.name,
             K.int_shape(self.final_pool_layer)))
    return

  
  def log(self, _s, _t=False):
    self.last_time = self.logger.VerboseLog(_s, show_time=_t)
    return
  
  def train(self, x, y, batch_size=64, epochs=1, 
                validation_data=None):
    self.log("Training...\n")
    self.model.fit(x, y, batch_size=batch_size, epochs=epochs, 
                   validation_data=validation_data)
    self.log("Training finished", _t=True)
    self.model_training_time = self.last_time
    return
    
  
  def conv2_bn(self, x, f, k=(1,1), s=(1,1), p='same', n='c2bn', activ='relu', use_bias=False):
    
    x = Conv2D(filters=f, 
               kernel_size=k, 
               strides=s, 
               padding=p, 
               use_bias=use_bias,
               name=n+'_cnv')(x)
    bn_axis = self.channel_axis
    x = BatchNormalization(axis=bn_axis, scale=False, name=n+'_bn')(x)
    x = Activation(activ, name=n+'_rel')(x)
    return x
    
  
  def SimpleInceptResBlock(self, input_tensor, 
                           n_maps=128, 
                           scale = 1.,
                           activ='relu', 
                           direct_skip=False,
                           name='CIncResB',
                           ):
    """
    inputs:  
      input_tensor:
      n_map: number of 3x3/5x5 maps within inception
        
    outputs: 
      output_tensor
    
    Uses basic Inception architecture with skip connection
    """
    self._block_count += 1
    name += '{:02d}'.format(self._block_count)
    self.short_desc.append('  {}: Simple_Incep_Res_Block, vol:{}'.format(name,n_maps))
    if direct_skip:
      # output volume equals input volume
      skip_tensor = input_tensor
      if self.channel_axis==3:
        output_volume = K.int_shape(input_tensor)[-1]
      else:
        output_volume = K.int_shape(input_tensor)[-3]
    else:
      # output volume equals n_maps
      output_volume = n_maps
      # "reshape" skip tensor to match volume
      skip_tensor = Conv2D(output_volume, 
                           kernel_size=(1,1), 
                           strides=(1,1), 
                           activation=None,
                           name=name+'_res_resh')(input_tensor)
      # batch norm probably not needed as we will do it in the next stage!
      #bn_axis = self.channel_axis
      #skip_tensor = BatchNormalization(axis=bn_axis, scale=False, 
      #                                 name=name+'_res_bn')(skip_tensor)
      
    # assume input is from a conv or concat or add with no BN and activation
    
    bn_axis = self.channel_axis
    x = BatchNormalization(axis=bn_axis, scale=False, 
                           name=name+'_bn_start')(input_tensor)
    x = Activation(activation=activ, name=name+'_start_'+activ)(x)
    
    n_btlneck1 = n_maps // 2
    n_btlneck2 = n_maps // 4
    
    tower_1 = self.conv2_bn(x, f=n_btlneck1, k=(1,1), n=name+'_t1', activ=activ)
    
    tower_2 = self.conv2_bn(x, f=n_btlneck2, k=(1,1), n=name+'_t2_c1', activ=activ)
    tower_2 = self.conv2_bn(tower_2, f=n_btlneck1, k=(3,3), n=name+'_t2_c31', activ=activ)    
    tower_2 = self.conv2_bn(tower_2, f=n_maps, k=(3,3), n=name+'_t2_c32', activ=activ)    
    
    tower_3 = self.conv2_bn(x, f=n_btlneck2, k=(1,1), n=name+'_t3_c1', activ=activ)
    tower_3 = self.conv2_bn(tower_3, f=n_maps, k=(5,5), n=name+'_t3_c5', activ=activ)    
    
    all_towers = [tower_1, tower_2, tower_3]
    x = Concatenate(axis=self.channel_axis, name=name+'_concat')(all_towers)
    
    x = Conv2D(filters=output_volume, 
               kernel_size=(1,1), strides=(1,1),
               padding='same',
               use_bias=True,
               activation=None,
               name=name+'_cnv_preadd')(x)
    
    # batch norm probably not needed as we will do it in the next stage!
    #bn_axis = self.channel_axis
    #x = BatchNormalization(axis=bn_axis, scale=False, 
    #                       name=name+'_bn_preadd')(x)
    
    # create AddScaleLayer: 
    # input[0]: skip tensor
    # input[1]: current output    
    AddScaleLayer = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                           output_shape=K.int_shape(x)[1:],
                           arguments={'scale': scale},
                           name=name+'_addscal')

    x = AddScaleLayer([skip_tensor, x])  #add([skip_tensor, x])
      
    # removed final activation  for gradient flow optimization  - no activations between skips !
    #x = Activation(activation=activ, name=name+'_fin_'+activ)(x)
    
    return x
  
  def SimpleSepResBlock(self,input_tensor, n_maps=128, activ='relu', 
                        gradual=True,
                        direct_skip=False,
                        name='CSepResB',
                        ):
    bn_axis = self.channel_axis
    self._block_count += 1
    name += '{:02d}'.format(self._block_count)
    self.short_desc.append('  {}: Simple_Separ_Res_Block, vol:{}'.format(name,n_maps))

    if gradual:
      maps = [n_maps // 4, 
              n_maps // 2, 
              n_maps]
    else:
      maps = [n_maps, n_maps, n_maps]

    if direct_skip:
      skip_tensor = input_tensor
      if self.channel_axis==3:
        last_volume = K.int_shape(input_tensor)[-1]
      else:
        last_volume = K.int_shape(input_tensor)[-3]
    else:
      last_volume = n_maps
      skip_tensorx = Conv2D(last_volume, 
                            kernel_size=(1,1), 
                            strides=(1,1), 
                            activation=None,
                            name=name+'_res_resh')(input_tensor)
      # batch norm probably not needed as we will do it in the next stage!      
      #bn_axis = self.channel_axis
      #skip_tensor = BatchNormalization(axis=bn_axis, scale=False, 
      #                                 name=name+'_res_bn')(skip_tensor)


    x = BatchNormalization(axis=bn_axis, name=name + '_scnv0_bn')(x)
    x = Activation(activ, name=name + '_scnv0_act')(x)
         
    x = SeparableConv2D(maps[0], (3, 3), 
                        padding='same', 
                        use_bias=False, 
                        name=name + '_scnv1')(input_tensor)
    x = BatchNormalization(name=name + '_scnv1_bn')(x)
    x = Activation(activ, name=name + '_scnv1_act')(x)
    
    x = SeparableConv2D(maps[1], (3, 3), 
                        padding='same', 
                        use_bias=False, 
                        name=name + '_scnv2')(x)
    x = BatchNormalization(name=name + '_scnv2_bn')(x)
    x = Activation(activ, name=name + '_scnv2_act')(x)
    
    x = SeparableConv2D(maps[2], (3, 3), 
                        padding='same', 
                        use_bias=False, 
                        name=name + '_scnv3')(x)
    # batch norm probably not needed as we will do it in the next stage!      
    #x = BatchNormalization(name=name + '_scnv3_bn')(x)
    
    if direct_skip and (last_volume != maps[2]):
      x = BatchNormalization(name=name + '_scnv3_bn')(x) # moved here BN from prev lines
      x = Activation(activ, name=name + '_scnv3_act')(x)
      x = Conv2D(last_volume, 
                 kernel_size=(1,1), 
                 strides=(1,1), 
                 activation=None,
                 name=name+'_out_resh')(x)
      # batch norm probably not needed as we will do it in the next stage!      
      #x = BatchNormalization(axis=bn_axis, scale=False, 
      #                       name=name+'_out_bn')(x)
    
    
    x = add([x, skip_tensor])
    
    # remove for gradiend flow optimization  - no activations between skips !
    #x = Activation(activ, name=name + '_fin_' + activ)(x) 
    
    return x
  
  def _start_block_v1(self, input_tensor):
    x = input_tensor
    x = self.conv2_bn(x, f=32, k=(3,3), p='same', n='block1_c1')
    x = self.conv2_bn(x, f=48, k=(3,3), s=(2,2), p='same', n='block1_c2')
    x = Conv2D(filters=64,
               kernel_size=(1,1),
               stride=(1,1),
               padding='valid',
               name='block1_c3')(x)
    return x
  
  
  def _add_head(self, head_type):    
    if head_type == 1:
      self.final_out_layer = Dense(n_classes, activation='softmax', 
                                   name='readout')(self.final_out_layer)

    elif head_type == 2:
      # deconv process
      out_deconved = self._get_dense_output()
      self.final_out_layer = Activation("softmax")(out_deconved)
  
  
  def _get_dense_output(self):
    deconv_list = []
    AddScaleLayer = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                           output_shape=K.int_shape(x)[1:],
                           arguments={'scale': scale},
                           name=name+'_fcn_as')
    result =  Conv2DTranspose(filters=,
                                 kernel_size=,
                                 strides=,
                                 padding=,)(self.downsamplers[-1][0]])
    n_layers = len(self.downsamplers)-1
    for i in range(n_layers):
      layer, factor, scale = self.downsamplers[i]
    
      deconved = Conv2DTranspose(filters=,
                                 kernel_size=,
                                 strides=,
                                 padding=,)(layer)
      deconv_list.append(deconved)
    return add(deconv_list, axis=self.channel_axis)
  
  def _build_network_v1(self, input_shape, n_classes, activ='relu', 
                        head_type=1, 
                        conv_head=False,
                        pooling='max', 
                        simple_compile=False):
    """
    
    head_type = 0: No head, 1: classification head; 2: dense softmax
    """
    input_tensor = Input(input_shape)
    self.downsamplers = []
    self.input_tensor = input_tensor
    
    x = self._start_block_v1(self.input_tensor)
    

    x = self.SimpleInceptResBlock(x, n_maps=128, direct_skip=True)
    x = self.SimpleInceptResBlock(x, n_maps=256, direct_skip=False)
    x = self.SimpleInceptResBlock(x, n_maps=384, direct_skip=True)
    self.downsamplers.append((x,0.5))

    x = self.conv2_bn(x, f=256, k=(3,3), s=(2,2), p='same', n='downsmpl1')    
    x = self.SimpleInceptResBlock(x, n_maps=512, direct_skip=False)
    self.downsamplers.append((x,0.25))

    
    self.final_conv_layer = x
    
    if pooling == 'max':
      x = GlobalMaxPooling2D()(x)
    elif pooling == 'avg':
      x = GlobalAveragePooling2D()(x)
    
    self.final_pool_layer = x
    self.final_out_layer = x   
    
    self._add_head(head_type)
                  
    
    self.model = Model(self.input_tensor, self.final_out_layer)
    if simple_compile:
      self.log("Compiling...")
      self.model.compile(optimizer='adam', loss='categorical_crossentropy', 
                         metrics=['accuracy'])
      self.log("Done compiling.")
    return self.model


  def _build_network_v2(self, 
                        input_shape, 
                        n_classes,                         
                        activ='relu', include_head=True, 
                        pooling='max', simple_compile=False):
    input_tensor = Input(input_shape)
    self.input_tensor = input_tensor
    
    ## When using SimpleSepResBlock:
    ## start with conv
    ## end with BN and activation !
    x = self._start_block(self.input_tensor)    
    
    # input must be simple conv
    x = self.SimpleSepResBlock(x, n_maps=128, direct_skip=..)
    x = self.SimpleSepResBlock(x, n_maps=256, gradual=True, direct_skip=..)
    x = self.SimpleSepResBlock(x, n_maps=384, gradual=True, direct_skip=..)
    x = self.SimpleSepResBlock(x, n_maps=512, gradual=True, direct_skip=..)
    x = self.SimpleSepResBlock(x, n_maps=1024, gradual=True, direct_skip=..)
    x = self.SimpleSepResBlock(x, n_maps=2048, gradual=True, direct_skip=..)
    # after output do BN and Activ

    
    self.final_conv_layer = x
    
    if pooling == 'max':
      x = GlobalMaxPooling2D()(x)
    elif pooling == 'avg':
      x = GlobalAveragePooling2D()(x)
    
    self.final_pool_layer = x
    
    if include_head:
      x = Dense(n_classes, activation='softmax', name='readout')(x)
    
    self.final_out_layer = x
        
    self.model = Model(self.input_tensor, self.final_out_layer)
    if simple_compile:
      self.log(" Compiling...")
      self.model.compile(optimizer='adam', loss='categorical_crossentropy', 
                         metrics=['accuracy'])
      self.log(" Done compiling.")
    return self.model
    
  def _build_network_v3(self, 
                        input_shape, 
                        n_classes,                         
                        activ='relu', include_head=True, 
                        pooling='max', simple_compile=False):
    input_tensor = Input(input_shape)
    self.input_tensor = input_tensor
    
    ## When using SimpleSepResBlock:
    ## start with conv
    ## end with BN and activation !
    x = self._start_block(self.input_tensor)    
    
    # input must be simple conv
    x = self.SimpleInceptResBlock(x, n_maps=128, direct_skip=..)
    x = self.SimpleInceptResBlock(x, n_maps=256, gradual=True, direct_skip=..)
    x = self.SimpleInceptResBlock(x, n_maps=384, gradual=True, direct_skip=..)
    x = self.SimpleSepResBlock(x, n_maps=512, gradual=True, direct_skip=..)
    x = self.SimpleSepResBlock(x, n_maps=1024, gradual=True, direct_skip=..)
    x = self.SimpleSepResBlock(x, n_maps=2048, gradual=True, direct_skip=..)
    # after output do BN and Activ

    
    self.final_conv_layer = x
    
    if pooling == 'max':
      x = GlobalMaxPooling2D()(x)
    elif pooling == 'avg':
      x = GlobalAveragePooling2D()(x)
    
    self.final_pool_layer = x
    
    if include_head:
      x = Dense(n_classes, activation='softmax', name='readout')(x)
    
    self.final_out_layer = x
        
    self.model = Model(self.input_tensor, self.final_out_layer)
    if simple_compile:
      self.log(" Compiling...")
      self.model.compile(optimizer='adam', loss='categorical_crossentropy', 
                         metrics=['accuracy'])
      self.log(" Done compiling.")
    return self.model


if __name__=='__main__':
  from keras.datasets import cifar100
  from keras.datasets import cifar10
  from keras.datasets import mnist
  from keras.utils import np_utils
  
  dataset = cifar100

  (X_train, y_train), (X_test, y_test) = dataset.load_data()
  if dataset == mnist:
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
  print("Showing sampled images...")
  nr_img = 2
  test_img = X_train[np.random.randint(0,1000,size=nr_img)]
  for i in range(test_img.shape[0]):
    plt.imshow(test_img[i])
    plt.show()
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255
  X_test /= 255    
  n_classes = np.unique(y_train).shape[0]
  Y_train = np_utils.to_categorical(y_train, n_classes)
  Y_test = np_utils.to_categorical(y_test, n_classes)    
  input_shape = X_train.shape[1:]  
  results = []
  epochs = 5
  for v in ['v1','v2', 'v3']:     
    for agg in ['max','avg']:
      cnet = CloudifierNet(n_classes=n_classes, 
                           input_shape=input_shape, 
                           ver=v, 
                           pooling=agg, 
                           )
      cnet.train(x=X_train, y=Y_train, batch_size=64, epochs=epochs, 
                 validation_data=(X_test,Y_test))
      model = cnet.model      
      res = model.evaluate(X_test,Y_test)
      results.append((v, agg, res,
                      model.metrics_names, 
                      model, 
                      cnet.model_training_time))
 
  for res in results:
    ir, v, agg, res, metrics, _model, train_time = res
    cnet.log("Model {}({}) {}: {:.2f} {}: {:.2f}% train: {:.1f}s".format(
        v, agg, metrics[0], res[0], metrics[1], res[1]*100, train_time))
    
     
      
    
    
    
    
    