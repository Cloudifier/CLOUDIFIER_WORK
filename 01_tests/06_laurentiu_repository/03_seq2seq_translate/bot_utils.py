# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:15:50 2018

@author: Andrei Ionut Damian
"""

import numpy as np
import nltk
import os

from tqdm import tqdm

__version__   = "0.1"
__author__    = "4E Software"
__copyright__ = "(C) 4E Software SRL"
__project__   = "TempRent"  
__module__    = "TempRentBotAlphaUtils"
__reference__ = ""


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


class DataLoaderV1:
  """
  basic data loader where input and target dictionary differs
  """
  def __init__(self, input_file, logger, max_samples=None, start_char='\t', end_char='\n',
               DEBUG=True):
    self.DEBUG=DEBUG
    self.START_CHAR = start_char
    self.END_CHAR = end_char
    self.max_samples = max_samples
    self.logger=logger
    self.max_inp_word_seq_len = 0
    self.max_out_word_seq_len = 0
    self.seps = [self.START_CHAR, self.END_CHAR,' ',',',';','.','!','?'] 
    self.inp_unk_word = 'UNKINP'
    self.out_unk_word = 'UNKOUT'
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    input_words = set()
    target_words = set()
    self.log("Loading data...")
    lines = open(input_file, 'r', encoding='utf-8').read().split('\n')
    self.log("Done loading data.", _t=True)
    self.log("Processing data ...")
    self.inp_data_char = []
    self.out_data_char = []
    self.inp_data_word = []
    self.out_data_word = []
    if self.max_samples is None:
      self.max_samples = len(lines)-1
    self.log('Generating dictionary...')      
    for line in tqdm(lines[: min(self.max_samples, len(lines) - 1)]):
      input_text, target_text = line.split('\t')
      input_texts.append(input_text)
      # We use "tab"+"space" as the "start sequence" character or word
      # for the targets, and "\n" as "end sequence" character.
      target_texts.append(self.START_CHAR + target_text + self.END_CHAR)
      
      
      
      #char level split
      for char in input_texts[-1]:
        if char not in input_characters:
          input_characters.add(char)
      for char in target_texts[-1]:
        if char not in target_characters:
          target_characters.add(char)    
          
      # word level split
      c_inp_words = self._tokenize_input(input_text)
      c_out_words = self._tokenize_output(target_text)
      
      self.max_inp_word_seq_len = max(self.max_inp_word_seq_len, len(c_inp_words))
      c_out_words = [self.START_CHAR] + c_out_words + [self.END_CHAR]
      self.max_out_word_seq_len = max(self.max_out_word_seq_len, len(c_out_words))      
      for word in c_inp_words:
        if word not in input_words:
          input_words.add(word)
      for word in c_out_words:
        if word not in target_words:
          target_words.add(word)
    
        
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    self.inp_char_vocab_size = len(input_characters)
    self.out_char_vocab_size = len(target_characters)      
    self.max_inp_char_seq_len = max([len(txt) for txt in input_texts])
    self.max_out_char_seq_len = max([len(txt) for txt in target_texts])          
    self.inp_char_to_id = dict(
        [(char, i) for i,char in enumerate(input_characters)])
    self.out_char_to_id = dict(
        [(char, i) for i,char in enumerate(target_characters)])
    self.inp_id_to_char = dict(
        [(i, char) for char, i in self.inp_char_to_id.items()])
    self.out_id_to_char = dict(
        [(i, char) for char, i in self.out_char_to_id.items()])
                      
    input_words.add(self.inp_unk_word)
    target_words.add(self.out_unk_word)


      
    self.inp_word_vocab_size = len(input_words)
    self.out_word_vocab_size = len(target_words)    
    self.inp_word_to_id = dict(
        [(word, i) for i,word in enumerate(sorted(list(input_words)))])
    self.out_word_to_id = dict(
        [(word, i) for i,word in enumerate(sorted(list(target_words)))])      
    self.inp_id_to_word = dict(
        [(i, word) for word, i in self.inp_word_to_id.items()])
    self.out_id_to_word = dict(
        [(i, word) for word, i in self.out_word_to_id.items()])


    self.log('Generating dataset...')      
    self.n_pairs = len(input_texts)
    for pair_index in tqdm(range(self.n_pairs)):
      self.inp_data_char.append(self.input_char_text_to_tokens(
                                          input_texts[pair_index]))
      self.out_data_char.append(self.output_char_text_to_tokens(
                                          target_texts[pair_index]))
      self.inp_data_word.append(self.input_word_text_to_tokens(
                                          input_texts[pair_index]))
      self.out_data_word.append(self.output_word_text_to_tokens(
                                          target_texts[pair_index]))
    
    self.log("Done processing data.",_t=True)

    self.log(" Sample pairs: {}".format(self.n_pairs))
    self.log(" Input char-level tokens  : {}".format(self.inp_char_vocab_size))
    self.log(" Output char-level tokens : {}".format(self.out_char_vocab_size))
    self.log(" Max chr-level in seq len : {}".format(self.max_inp_char_seq_len))
    self.log(" Max chr-level out seq len: {}".format(self.max_out_char_seq_len))
    self.log(" Input word-level tokens  : {}".format(self.inp_word_vocab_size))
    self.log(" Output word-level tokens : {}".format(self.out_word_vocab_size))
    self.log(" Max wrd-level in seq len : {}".format(self.max_inp_word_seq_len))
    self.log(" Max wrd-level out seq len: {}".format(self.max_out_word_seq_len))

    sample_id = np.random.randint( self.n_pairs // 4, self.n_pairs // 2)
    sample = input_texts[sample_id]
    self.log(" Inp sample  : '{}'".format(sample))
    sample_tokens = self.inp_data_char[sample_id]
    self.log(" Char tokens: {}".format(self._tokendebug(sample_tokens)))
    sample_reconstructed = self.input_char_tokens_to_text(sample_tokens)
    self.log(" Char tok2txt: '{}'".format(sample_reconstructed))

    sample_tokens = self.inp_data_word[sample_id]
    self.log(" Word tokens: {}".format(sample_tokens))
    sample_reconstructed = self.input_word_tokens_to_text(sample_tokens)
    self.log(" Word tok2txt: '{}'".format(sample_reconstructed))
    

    #sample_id = np.random.randint( self.n_pairs // 4, self.n_pairs // 2)
    sample = target_texts[sample_id]
    self.log(" Out sample  : '{}'".format(sample[1:-1]))
    sample_tokens = self.out_data_char[sample_id]
    self.log(" Char tokens: {}".format(self._tokendebug(sample_tokens)))
    sample_reconstructed = self.output_char_tokens_to_text(sample_tokens, convert_start_end=True)
    self.log(" Char tok2txt: '{}'".format(sample_reconstructed))

    sample_tokens = self.out_data_word[sample_id]
    self.log(" Word tokens: {}".format(sample_tokens))
    sample_reconstructed = self.output_word_tokens_to_text(sample_tokens, convert_start_end=True)
    self.log(" Word tok2txt: '{}'".format(sample_reconstructed))
    self.log("Data prepared. Use .get_train_test_data()")
    return
  
  
  def get_train_test_data(self, as_words=True, test_size=0.1, random_state=33, debug=False):
    """
    inputs: 
      as_words: if True then return data as words seq else return as char seq
    returns
        train_encoder_input, train_decoder_input, train_decoder_output
        test_encoder_input, test_decoder_input, test_decoder_output
    """
    indices = list(range(self.n_pairs))
    train_size = int(self.n_pairs * (1-test_size))
    test_size = self.n_pairs - train_size
    r = np.random.RandomState(random_state)
    train_indices = r.choice(indices, size=train_size, replace=False)
    test_indices = np.array(list(set(indices)-set(train_indices)))
    if as_words:
      trn_enc_inp = [self.inp_data_word[x] for x in train_indices]
      trn_dec_inp = [self.out_data_word[x][:-1] for x in train_indices]
      trn_dec_out = [self.out_data_word[x][1:] for x in train_indices]
      tst_enc_inp = [self.inp_data_word[x] for x in test_indices]
      tst_dec_inp = [self.out_data_word[x][:-1] for x in test_indices]
      tst_dec_out = [self.out_data_word[x][1:] for x in test_indices]
    else:
      trn_enc_inp = [self.inp_data_char[x] for x in train_indices]
      trn_dec_inp = [self.out_data_char[x][:-1] for x in train_indices]
      trn_dec_out = [self.out_data_char[x][1:] for x in train_indices]
      tst_enc_inp = [self.inp_data_char[x] for x in test_indices]
      tst_dec_inp = [self.out_data_char[x][:-1] for x in test_indices]
      tst_dec_out = [self.out_data_char[x][1:] for x in test_indices]
    
    if debug:
      self.log("Got {} train_indices --> {}".format(train_size, train_indices[:10]))
      self.log("Got {} test_indices --> {}".format(test_size, test_indices[:10]))
      
      train_index = 100
      test_index = 10
      trn_encin = trn_enc_inp[train_index]
      trn_decou = trn_dec_out[train_index]
      trn_decin = trn_dec_inp[train_index]
      tst_encin = tst_enc_inp[test_index]
      tst_decou = tst_dec_out[test_index]
      tst_decin = tst_dec_inp[test_index]
      
      self.log("trn_enc_inp: '{}'".format(trn_encin))
      self.log("trn_enc_inp: '{}'".format(self.tokens_to_text(trn_encin, is_input=True, as_words=as_words)))
    
      self.log("trn_dec_inp: '{}'".format(trn_decin))
      self.log("trn_dec_inp: '{}'".format(self.tokens_to_text(trn_decin, is_input=False, as_words=as_words)))
    
      self.log("trn_dec_out: '{}'".format(trn_decou))
      self.log("trn_dec_out: '{}'".format(self.tokens_to_text(trn_decou, is_input=False, as_words=as_words)))
    
      self.log("tst_enc_inp: '{}'".format(tst_encin))
      self.log("tst_enc_inp: '{}'".format(self.tokens_to_text(tst_encin, is_input=True, as_words=as_words)))
    
      self.log("tst_dec_inp: '{}'".format(tst_decin))
      self.log("tst_dec_inp: '{}'".format(self.tokens_to_text(tst_decin, is_input=False, as_words=as_words)))
    
      self.log("tst_dec_out: '{}'".format(tst_decou))
      self.log("tst_dec_out: '{}'".format(self.tokens_to_text(tst_decou, is_input=False, as_words=as_words)))
      
    
    return (trn_enc_inp, trn_dec_inp, trn_dec_out, 
            tst_enc_inp, tst_dec_inp, tst_dec_out)
  

  def _tokenize_input(self, _str):
    res = nltk.word_tokenize(_str)
    return res
  
  
  def _tokenize_output(self, _str):
    res = nltk.word_tokenize(_str)
    return res
    
  
  
  def _tokendebug(self, token_list):
    if self.DEBUG:
      res = token_list
    else:
      s_list = [str(x) for x in token_list]
      res = "#".join(s_list)
    return res
  
  def input_char_tokens_to_text(self, token_list):
    out_str = ""
    for token in token_list:
      char = self.inp_id_to_char[token]
      out_str += char
    return out_str

  def output_char_tokens_to_text(self, token_list, convert_start_end=True):
    out_str = ""
    text = ""
    for token in token_list:
      text = self.out_id_to_char[token]
      if text == self.START_CHAR:
        text = ''
        if convert_start_end:
          text = '[start]'
      elif text == self.END_CHAR:
        text = ''
        if convert_start_end:
          text = '[end]'
      out_str += text
    return out_str


  def input_word_tokens_to_text(self, token_list):
    out_str = ""
    for i, token in enumerate(token_list):
      word = self.inp_id_to_word[token]
      if (word not in self.seps) and (i>=1) and (i < (len(token_list)-1)):
        out_str += ' '
      out_str += word
    return out_str

  def output_word_tokens_to_text(self, token_list, convert_start_end=True):
    out_str = ""    
    text_started = False
    for i, token in enumerate(token_list):
      text = self.out_id_to_word[token]
      if ((text not in self.seps) and text_started and
          (i <= (len(token_list)-2)) and (i>=1)):
        out_str += ' '
      if (not text_started) and (text not in self.seps):
        text_started = True
      if text == self.START_CHAR:
        text = ''
        if convert_start_end:
          text = '[start]'
      elif text == self.END_CHAR:
        text = ''
        if convert_start_end:
          text = '[end]'
      out_str += text
    return out_str
  

  def input_char_text_to_tokens(self, text):
    tokens = []
    for char in text:
      tokens.append(self.inp_char_to_id[char])
    return tokens

  def output_char_text_to_tokens(self, text):
    tokens = []
    for char in text:
      tokens.append(self.out_char_to_id[char])
    return tokens
  
  def input_word_text_to_tokens(self, text):
    tokens = []
    words = nltk.word_tokenize(text)
    for word in words:
      tokens.append(self.inp_word_to_id[word])
    return tokens

  def output_word_text_to_tokens(self, text):
    tokens = []
    words = nltk.word_tokenize(text)
    words = [self.START_CHAR] + words + [self.END_CHAR]
    for word in words:
      tokens.append(self.out_word_to_id[word])
    return tokens
  
  
  def tokens_to_text(self, tokens, is_input=True, as_words=True):
    """
    """
    if is_input:
      if as_words:
        res = self.input_word_tokens_to_text(tokens)
      else:
        res = self.input_char_tokens_to_text(tokens)        
    else:
      if as_words:
        res = self.output_word_tokens_to_text(tokens)
      else:
        res = self.output_char_text_to_tokens(tokens)
    return res
  
      
  def log(self,_s,_t=False):
    return self.logger.P(_s, show_time=_t)
