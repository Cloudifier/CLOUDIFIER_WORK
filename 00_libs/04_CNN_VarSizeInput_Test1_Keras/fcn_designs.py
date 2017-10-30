# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:24:28 2017

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







"""

  
models_list_simple_def =[
       ("16_16_d_24_d_128e(1)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4),"a":"elu"},
                  {"t":"c2d","v":16,"k":(4,4),"a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":24,"k":(4,4),"a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":128,"k":(1,1),"a":"elu"},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
                  ]),

       ("16_16_d_24_d_128n(1)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4),"a":"elu"},
                  {"t":"c2d","v":16,"k":(4,4),"a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":24,"k":(4,4),"a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":128,"k":(1,1)},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
                  ]),

       ("16_16_d_24_d_128n(4)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4),"a":"elu"},
                  {"t":"c2d","v":16,"k":(4,4),"a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":24,"k":(4,4),"a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":128,"k":(4,4)},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
                  ]),
                        
       ("16_16_d_32_d_256n(1)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4),"a":"elu"},
                  {"t":"c2d","v":16,"k":(4,4),"a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":32,"k":(4,4),"a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":256,"k":(1,1)},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),
                  

       ("16_16_d_32_d_256e(1)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":16,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":256,"k":(1,1),"a":"elu"},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),

       ("16_16_d_32_d_256e(4)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":16,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":256,"k":(4,4),"a":"elu"},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),

       ("16_32_d_64_d_512n(1)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":512,"k":(1,1),},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),
                  
       ("16_32_d_64_d_512e(1)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":512,"k":(1,1),"a":"elu"},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),

       ("16_32_d_64_d_512e(1)_G_d_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":512,"k":(1,1),"a":"elu"},
                  {"t":"gmp"},
                  {"t":"drp","v":0.5},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),

       ("16_32_d_64_d_512n(4)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":512,"k":(4,4),},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),

       ("16_32_d_64_d_512e(4)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":16,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":512,"k":(4,4),"a":"elu"},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),

       ("16_32_d_64_d_512e(4)_G_d_SM",[
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
              ]),

       ("32_32_d_64_64_d_128_128_d_1024e(4)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":128,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":128,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":1024,"k":(4,4),"a":"elu"},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),

       ("32_32_d_64_64_d_128_128_d_1024e(4)_G_d_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":128,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":128,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":1024,"k":(4,4),"a":"elu"},
                  {"t":"gmp"},
                  {"t":"drp","v":0.5},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),
  
       ("32_32_d_64_64_d_128_128_d_1024e(1)_G_d_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":128,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":128,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":1024,"k":(1,1),"a":"elu"},
                  {"t":"gmp"},
                  {"t":"drp","v":0.5},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),

       ("32_32_d_64_64_d_128_128_d_1024e(1)_G_SM",[
                  {"t":"inp","v":(None, None,nr_ch)},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":32,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":64,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":128,"k":(4,4), "a":"elu"},
                  {"t":"c2d","v":128,"k":(4,4), "a":"elu"},
                  {"t":"drp","v":0.5},
                  {"t":"c2d","v":1024,"k":(1,1),"a":"elu"},
                  {"t":"gmp"},
                  {"t":"dns", "v":10, "a":"softmax"}
              ]),  
          ]
  
  
  models_block_defs1 = [

      ("B1_16x2_32x2_128(1)x1_GMP_512d_512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":128, "KERN":(1,1), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":256, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),


      ("B2_16x2_32x2_128(1)x1_GMP_512d_512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":128, "KERN":(1,1), "BATN":2, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":256, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),

      ("B0_16x2_32x2_128(1)x1_GMP_512d_512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":0, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":0, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":128, "KERN":(1,1), "BATN":0, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":256, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),


      ("B1_16x2_32x2_128(1)x1_GMP_512_512",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":128, "KERN":(1,1), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":256, "ACTV":"elu"},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),

      ("B2_16x2_32x2_128(1)x1_GMP_512_512",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":128, "KERN":(1,1), "BATN":2, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":256, "ACTV":"elu"},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),



      ("B1_32x2_64x2_512(1)x1_GMP_512d_512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":512, "KERN":(1,1), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),

      ("B2_32x2_64x2_512(1)x1_GMP_512d_512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":2, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":512, "KERN":(1,1), "BATN":2, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),

      ("32x2_64x2_512(3)x1_GMP_512d_512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":512, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),

      ("32x2_64x2_512x2_GMP_512d_512d",
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

      ("16x2_32x2_64x2_GMP_512d_512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIRAMID_LIKE", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),



      ("16x2_32x2_64x2_128x2_256x1_GMP_512d_512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK4", "TYPE":"CONV", "NRLY":2, "VALU":128, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK5", "TYPE":"CONV", "NRLY":1, "VALU":256, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIRAMID_LIKE", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),
   
       ("16x2_32x2_64x2_128x2_256x2_GMP_512d_512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK4", "TYPE":"CONV", "NRLY":2, "VALU":128, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK5", "TYPE":"CONV", "NRLY":2, "VALU":256, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIRAMID_LIKE", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),  

      ]


  models_block_defs = [



      ("16x2->32x2->128(1)x1->GMP->512->512",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":128, "KERN":(1,1), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":256, "ACTV":"elu"},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),

      ("16x2->32x2->128(3)x1->GMP->512->512",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":1, "VALU":128, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":256, "ACTV":"elu"},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),
      ("32x2->64x2->512(1)x2->GMP->512d->512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":512, "KERN":(1,1), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIR", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),



      ("32x2->64x2->512(3)x2->GMP->512d->512d",
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


   
       ("16x2->32x2->64x2->128x2->256(3)x2->GMP->512d->512d",
          [
          {"TYPE":"INPUT", "VALU":(None,None,nr_ch)},
          {"NAME":"CBLOCK1", "TYPE":"CONV", "NRLY":2, "VALU":16, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK2", "TYPE":"CONV", "NRLY":2, "VALU":32, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK3", "TYPE":"CONV", "NRLY":2, "VALU":64, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK4", "TYPE":"CONV", "NRLY":2, "VALU":128, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"CBLOCK5", "TYPE":"CONV", "NRLY":2, "VALU":256, "KERN":(3,3), "BATN":1, "ACTV":"elu"},
          {"NAME":"PIRAMID_LIKE", "TYPE":"PIRAMID"},
          {"NAME":"FINAL","TYPE":"FC","NRLY":2,"VALU":512, "ACTV":"elu","DROP":0.5},
          {"NAME":"OUT","TYPE":"READOUT","VALU":10,"ACTV":"softmax"}
          ]       
       ),  

       ("16x2->32x2->64x2->128x2->256(1)x2->GMP->512d->512d",
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

      ]

