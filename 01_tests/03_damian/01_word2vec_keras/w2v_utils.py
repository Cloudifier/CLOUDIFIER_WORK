# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:14:58 2018

@author: Andrei Ionut Damian
"""

import numpy as np


def cosim_vec_mat(emb_vector, emb_matrix, normed_vector=True,normed_matrix=True):
  """
  compute cosine similarity between a vector and a whole matrix
  inputs:
    normed_vector, normed_matrix: true if vector and/or matrix are already normed
  """
  emb_vector = emb_vector.ravel()
  assert emb_vector.shape[0]==emb_matrix.shape[1], "vector and matrix must have same embedding size"
  if not normed_vector:
    emb_vector /= np.sqrt(np.sum(emb_vector**2))
  if not normed_matrix:
    emb_matrix /= np.sqrt(np.sum(emb_matrix**2, axis=1, preserve_dims=True))
  
  cos = emb_matrix.dot(emb_vector)  
  return cos


def compute_biconcept_vector(embeds_1, embeds_2):
  """
  inputs:
      embeds_1, embeds_2: array of [M, EMBEDS] with selected pairs:
        example:
          embeds_1 = ['man','father','grandfather']
          embeds_2 = ['woman', 'mother', 'grandmother']
  output:
    concept vector embeddings
  """
  assert embeds_1.shape == embeds_2.shape, "Both embeddings matrices must have same shape"
  diffs = embeds_1 - embeds_2
  avg  = diffs.mean(axis = 0)
  return avg


def analyze_concept(embeds, embeds_1, embeds_2):
  """
  inputs:
    embeds: a matrix of embeddings that will be analysed as being either part of
    embedds_1, embeds_2: paired matrices for compute_concept_vector
  output:
    list of "belonging" - posizive for embeds_1 and negative for embeds_2
  """
  biconcept = compute_biconcept_vector(embeds_1,embeds_2)
  res = cosim_vec_mat(biconcept, embeds)
  return res
  