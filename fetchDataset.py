# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:40:36 2019

@author: binbi
"""

import numpy as np
from numpy import genfromtxt

import os




def fetchDataset(dataset='yanzu_5.8'):
    
    #os.chdir("D:\\data_aquisition\\yazu_05_08")
    #os.chdir('C:\\Users\\binbi\\Desktop\\data_aquisition\\yixing')
    #os.chdir('C:\\Users\\binbi\\Desktop\\data_aquisition\\marcus')
    
    
    # subject yazhou
    if 'yanzu' in dataset:
        
        #change directory
        os.chdir("D:\\data_aquisition\\yazu_05_08")
        
        if dataset=='yanzu':
        
            X1 = genfromtxt('yanzu_3.6_X.txt', delimiter='')
            y1 = genfromtxt('yanzu_3.6_Y.txt',delimiter=',')
            X2 = genfromtxt('yanzu_4.2_X.txt', delimiter='')
            y2 = genfromtxt('yanzu_4.2_Y.txt',delimiter=',')       
            X3 = genfromtxt('yanzu_4.7_X.txt', delimiter='')
            y3 = genfromtxt('yanzu_4.7_Y.txt',delimiter=',')
            X4 = genfromtxt('yanzu_5.3_X.txt', delimiter='')
            y4 = genfromtxt('yanzu_5.3_Y.txt',delimiter=',')
            X5 = genfromtxt('yanzu_5.8_X.txt', delimiter='')
            y5 = genfromtxt('yanzu_5.8_Y.txt',delimiter=',')
            
            X = np.concatenate((X1,X2,X3,X4,X5))
            y = np.concatenate((y1,y2,y3,y4,y5))

        
        
        
        
        elif dataset == 'yanzu_3.6':
            X = genfromtxt('yanzu_3.6_X.txt', delimiter='')
            y = genfromtxt('yanzu_3.6_Y.txt',delimiter=',')
            
        elif dataset == 'yanzu_4.2':
            X = genfromtxt('yanzu_4.2_X.txt', delimiter='')
            y = genfromtxt('yanzu_4.2_Y.txt',delimiter=',')
            
        elif dataset == 'yanzu_4.7':
            X = genfromtxt('yanzu_4.7_X.txt', delimiter='')
            y = genfromtxt('yanzu_4.7_Y.txt',delimiter=',')
            
        elif dataset == 'yanzu_5.3':
            X = genfromtxt('yanzu_5.3_X.txt', delimiter='')
            y = genfromtxt('yanzu_5.3_Y.txt',delimiter=',')
            
        elif dataset == 'yanzu_5.8':
            X = genfromtxt('yanzu_5.8_X.txt', delimiter='')
            y = genfromtxt('yanzu_5.8_Y.txt',delimiter=',')
        else:
            print("Please specify the correct velocity!")
            X = np.zeros(0)
            y = np.zeros(0)
            
        
    # Subject yixing       
        
    elif 'yixing' in dataset:
        #change directory
        os.chdir("D:\\data_aquisition\\yixing")
        
        # load data from all velocity
        if dataset == 'yixing':
            
            X1 = genfromtxt('yixing_3.8_X.txt', delimiter='')
            y1 = genfromtxt('yixing_3.8_Y.txt',delimiter=',')
            X2 = genfromtxt('yixing_4.2_wideHeelOff_X.txt', delimiter='')
            y2 = genfromtxt('yixing_4.2_wideHeelOff_Y.txt',delimiter=',')       
            X3 = genfromtxt('yixing_5.0_X.txt', delimiter='')
            y3 = genfromtxt('yixing_5.0_Y.txt',delimiter=',')
            X4 = genfromtxt('yixing_5.5_X.txt', delimiter='')
            y4 = genfromtxt('yixing_5.5_Y.txt',delimiter=',')
            X5 = genfromtxt('yixing_6.1_X.txt', delimiter='')
            y5 = genfromtxt('yixing_6.1_Y.txt',delimiter=',')
            
            X = np.concatenate((X1,X2,X3,X4,X5))
            y = np.concatenate((y1,y2,y3,y4,y5))
        
        elif dataset == 'yixing_3.8':
            X = genfromtxt('yixing_3.8_X.txt', delimiter='')
            y = genfromtxt('yixing_3.8_Y.txt',delimiter=',')
            
        elif dataset == 'yixing_4.2':
            X = genfromtxt('yixing_4.2_269s_X.txt', delimiter='')
            y = genfromtxt('yixing_4.2_269s_Y.txt',delimiter=',')
            
        elif dataset == 'yixing_5.0':
            X = genfromtxt('yixing_5.0_X.txt', delimiter='')
            y = genfromtxt('yixing_5.0_Y.txt',delimiter=',') 
            
        elif dataset == 'yixing_5.5':
            X = genfromtxt('yixing_5.5_X.txt', delimiter='')
            y = genfromtxt('yixing_5.5_Y.txt',delimiter=',')
            
        elif dataset == 'yixing_6.1':
            X = genfromtxt('yixing_6.1_188s_X.txt', delimiter='')
            y = genfromtxt('yixing_6.1_188s_Y.txt',delimiter=',')
            
        else:
            print("Please specify the correct velocity!")
            X = np.zeros(0)
            y = np.zeros(0)
        
# Subject Marcus
    elif 'marcus' in dataset:
        #change directory
        os.chdir("D:\\data_aquisition\\marcus")
        
        if dataset == 'marcus':
            X1 = genfromtxt('Marcus_4.0_X.txt', delimiter='')
            y1 = genfromtxt('Marcus_4.0_Y.txt',delimiter=',')
            X2 = genfromtxt('Marcus_4.6_X.txt', delimiter='')
            y2 = genfromtxt('Marcus_4.6_Y.txt',delimiter=',')       
            X3 = genfromtxt('Marcus_5.2_X.txt', delimiter='')
            y3 = genfromtxt('Marcus_5.2_Y.txt',delimiter=',')
            X4 = genfromtxt('Marcus_5.8_X.txt', delimiter='')
            y4 = genfromtxt('Marcus_5.8_Y.txt',delimiter=',')
            X5 = genfromtxt('Marcus_6.4_X.txt', delimiter='')
            y5 = genfromtxt('Marcus_6.4_Y.txt',delimiter=',')
            
            X = np.concatenate((X1,X2,X3,X4,X5))
            y = np.concatenate((y1,y2,y3,y4,y5))
    
        
        elif dataset == 'marcus_4.0':
            X = genfromtxt('Marcus_4.0_X.txt', delimiter='')
            y = genfromtxt('Marcus_4.0_Y.txt',delimiter=',') 
        
        elif dataset == 'marcus_4.0_all':
            X = genfromtxt('Marcus_4.0_all_X.txt', delimiter='')
            y = genfromtxt('Marcus_4.0_all_Y.txt',delimiter=',') 
        
        
        elif dataset == 'marcus_4.6_all':
            X = genfromtxt('Marcus_4.6_all_X.txt', delimiter='')
            y = genfromtxt('Marcus_4.6_all_Y.txt',delimiter=',') 
            
        elif dataset == 'marcus_4.6':
            X = genfromtxt('Marcus_4.6_X.txt', delimiter='')
            y = genfromtxt('Marcus_4.6_Y.txt',delimiter=',') 
            
        elif dataset == 'marcus_4.6_filtered':
            X = genfromtxt('Marcus_4.6_all_filtered_X.txt', delimiter='')
            y = genfromtxt('Marcus_4.6_all_filtered_Y.txt',delimiter=',') 
            
        elif dataset == 'marcus_5.2':
            X = genfromtxt('Marcus_5.2_X.txt', delimiter='')
            y = genfromtxt('Marcus_5.2_Y.txt',delimiter=',') 
            
        elif dataset == 'marcus_5.8':
            X = genfromtxt('Marcus_5.8_X.txt', delimiter='')
            y = genfromtxt('Marcus_5.8_Y.txt',delimiter=',') 
            
        elif dataset == 'marcus_6.4':
            X = genfromtxt('Marcus_6.4_X.txt', delimiter='')
            y = genfromtxt('Marcus_6.4_Y.txt',delimiter=',') 
        
        else:
            print("Please specify the correct velocity!")
            X = np.zeros(0)
            y = np.zeros(0)
        
# Subject Gunnar
        
    elif 'gunnar' in dataset:
        os.chdir("D:\\data_aquisition\\gunnar")
        
        if dataset=='gunnar':
#            X1 = genfromtxt('gunnar_3.6_X.txt', delimiter='')
#            y1 = genfromtxt('gunnar_3.6_Y.txt',delimiter=',')
            X2 = genfromtxt('gunnar_4.2_X.txt', delimiter='')
            y2 = genfromtxt('gunnar_4.2_Y.txt',delimiter=',')       
#            X3 = genfromtxt('gunnar_4.7_X.txt', delimiter='')
#            y3 = genfromtxt('gunnar_4.7_Y.txt',delimiter=',')
#            X4 = genfromtxt('gunnar_5.2_X.txt', delimiter='')
#            y4 = genfromtxt('gunnar_5.2_Y.txt',delimiter=',')
            X5 = genfromtxt('gunnar_5.8_X.txt', delimiter='')
            y5 = genfromtxt('gunnar_5.8_Y.txt',delimiter=',')
            
            X = np.concatenate((X2,X5))
            y = np.concatenate((y2,y5))
            
        elif dataset == 'gunnar_3.6':
            X= genfromtxt('gunnar_3.6_X.txt', delimiter='')
            y= genfromtxt('gunnar_3.6_Y.txt',delimiter=',')
            
        elif dataset == 'gunnar_4.2':
            X= genfromtxt('gunnar_4.2_X.txt', delimiter='')
            y= genfromtxt('gunnar_4.2_Y.txt',delimiter=',')
            
        elif dataset == 'gunnar_4.7':
            X= genfromtxt('gunnar_4.7_X.txt', delimiter='')
            y= genfromtxt('gunnar_4.7_Y.txt',delimiter=',')
            
        elif dataset == 'gunnar_5.2':
            X= genfromtxt('gunnar_5.2_X.txt', delimiter='')
            y= genfromtxt('gunnar_5.2_Y.txt',delimiter=',')
            
        elif dataset == 'gunnar_5.8':
            X= genfromtxt('gunnar_5.8_X.txt', delimiter='')
            y= genfromtxt('gunnar_5.8_Y.txt',delimiter=',')
            
        else:
            print("Please specify the correct velocity!")
            X = np.zeros(0)
            y = np.zeros(0)
            
    elif 'hui' in dataset:
        os.chdir("D:\\data_aquisition\\hui")
        
        if dataset=='hui':
            X1 = genfromtxt('hui_3.8_X.txt', delimiter='')
            y1 = genfromtxt('hui_3.8_Y.txt',delimiter=',')
            X2 = genfromtxt('hui_4.4_X.txt', delimiter='')
            y2 = genfromtxt('hui_4.4_Y.txt',delimiter=',')       
            X3 = genfromtxt('hui_4.9_X.txt', delimiter='')
            y3 = genfromtxt('hui_4.9_Y.txt',delimiter=',')
            X4 = genfromtxt('hui_5.5_X.txt', delimiter='')
            y4 = genfromtxt('hui_5.5_Y.txt',delimiter=',')
            X5 = genfromtxt('hui_6.1_X.txt', delimiter='')
            y5 = genfromtxt('hui_6.1_Y.txt',delimiter=',')
            
            X = np.concatenate((X1,X2,X3,X4,X5))
            y = np.concatenate((y1,y2,y3,y4,y5))
            
        elif dataset == 'hui_3.8':
            X= genfromtxt('hui_3.8_X.txt', delimiter='')
            y= genfromtxt('hui_3.8_Y.txt',delimiter=',')
            
        elif dataset == 'hui_4.4':
            X= genfromtxt('hui_4.4_X.txt', delimiter='')
            y= genfromtxt('hui_4.4_Y.txt',delimiter=',')
            
        elif dataset == 'hui_4.9':
            X= genfromtxt('hui_4.9_X.txt', delimiter='')
            y= genfromtxt('hui_4.9_Y.txt',delimiter=',')
            
        elif dataset == 'hui_5.5':
            X= genfromtxt('hui_5.5_X.txt', delimiter='')
            y= genfromtxt('hui_5.5_Y.txt',delimiter=',')
            
        elif dataset == 'hui_5.5_feature_12_KalF':
            X= genfromtxt('hui_5.5_KalF_X.txt', delimiter='')
            y= genfromtxt('hui_5.5_Y.txt',delimiter=',')
            
        elif dataset == 'hui_6.1':
            X= genfromtxt('hui_6.1_X.txt', delimiter='')
            y= genfromtxt('hui_6.1_Y.txt',delimiter=',')
            
        else:
            print("Please specify the correct velocity!")
            X = np.zeros(0)
            y = np.zeros(0)
            
    elif 'snorri' in dataset:
        os.chdir("D:\\data_aquisition\\snorri")
        
        if dataset=='snorri':
            X1 = genfromtxt('snorri_4.1_X.txt', delimiter='')
            y1 = genfromtxt('snorri_4.1_Y.txt',delimiter=',')
            X2 = genfromtxt('snorri_4.8_X.txt', delimiter='')
            y2 = genfromtxt('snorri_4.8_Y.txt',delimiter=',')       
            X3 = genfromtxt('snorri_5.4_X.txt', delimiter='')
            y3 = genfromtxt('snorri_5.4_Y.txt',delimiter=',')
            X4 = genfromtxt('snorri_6.0_X.txt', delimiter='')
            y4 = genfromtxt('snorri_6.0_Y.txt',delimiter=',')
            X5 = genfromtxt('snorri_6.6_X.txt', delimiter='')
            y5 = genfromtxt('snorri_6.6_Y.txt',delimiter=',')
            
            X = np.concatenate((X1,X2,X3,X4,X5))
            y = np.concatenate((y1,y2,y3,y4,y5))
            
        elif dataset == 'snorri_4.1':
            X= genfromtxt('snorri_4.1_X.txt', delimiter='')
            y= genfromtxt('snorri_4.1_Y.txt',delimiter=',')
            
        elif dataset == 'snorri_4.1_all':
            X= genfromtxt('snorri_4.1_all_X.txt', delimiter='')
            y= genfromtxt('snorri_4.1_all_Y.txt',delimiter=',')
            
        elif dataset == 'snorri_4.8':
            X= genfromtxt('snorri_4.8_X.txt', delimiter='')
            y= genfromtxt('snorri_4.8_Y.txt',delimiter=',')
            
        elif dataset == 'snorri_5.4':
            X= genfromtxt('snorri_5.4_X.txt', delimiter='')
            y= genfromtxt('snorri_5.4_Y.txt',delimiter=',')
            
        elif dataset == 'snorri_6.0':
            X= genfromtxt('snorri_6.0_X.txt', delimiter='')
            y= genfromtxt('snorri_6.0_Y.txt',delimiter=',')
            
        elif dataset == 'snorri_6.6':
            X= genfromtxt('snorri_6.6_X.txt', delimiter='')
            y= genfromtxt('snorri_6.6_Y.txt',delimiter=',')
            
        else:
            print("Please specify the correct velocity!")
            X = np.zeros(0)
            y = np.zeros(0)
            
    elif 'longbin' in dataset:
        os.chdir("D:\\data_aquisition\\longbin")
        
        
        if dataset=='longbin':
            X1 = genfromtxt('longbin_3.8_frontal_X.txt', delimiter='')
            y1 = genfromtxt('longbin_3.8_frontal_Y.txt',delimiter=',')
            X2 = genfromtxt('longbin_4.4_frontal_X.txt', delimiter='')
            y2 = genfromtxt('longbin_4.4_frontal_Y.txt',delimiter=',')       
            X3 = genfromtxt('longbin_4.9_frontal_X.txt', delimiter='')
            y3 = genfromtxt('longbin_4.9_frontal_Y.txt',delimiter=',')
            X4 = genfromtxt('longbin_5.5_frontal_X.txt', delimiter='')
            y4 = genfromtxt('longbin_5.5_frontal_Y.txt',delimiter=',')
            X5 = genfromtxt('longbin_6.1_frontal_X.txt', delimiter='')
            y5 = genfromtxt('longbin_6.1_frontal_Y.txt',delimiter=',')
            
            X = np.concatenate((X1,X2,X3,X4,X5))
            y = np.concatenate((y1,y2,y3,y4,y5))
        
        
        
        if dataset == 'longbin_3.8':
            
            X= genfromtxt('longbin_3.8_lateral_bad_X.txt', delimiter='')
            y= genfromtxt('longbin_3.8_lateral_bad_Y.txt',delimiter=',')
            
        if dataset == 'longbin_frontal_3.8':
            
            X= genfromtxt('longbin_3.8_frontal_X.txt', delimiter='')
            y= genfromtxt('longbin_3.8_frontal_Y.txt',delimiter=',')
            
        if dataset == 'longbin_frontal_4.4':
            
            X= genfromtxt('longbin_4.4_frontal_X.txt', delimiter='')
            y= genfromtxt('longbin_4.4_frontal_Y.txt',delimiter=',')
            
        if dataset == 'longbin_frontal_4.9':
            
            X= genfromtxt('longbin_4.9_frontal_X.txt', delimiter='')
            y= genfromtxt('longbin_4.9_frontal_Y.txt',delimiter=',')             
            
        if dataset == 'longbin_frontal_5.5':
            
            X= genfromtxt('longbin_5.5_frontal_X.txt', delimiter='')
            y= genfromtxt('longbin_5.5_frontal_Y.txt',delimiter=',')

        if dataset == 'longbin_frontal_6.1':
            
            X= genfromtxt('longbin_6.1_frontal_X.txt', delimiter='')
            y= genfromtxt('longbin_6.1_frontal_Y.txt',delimiter=',') 
            
    elif 'song' in dataset:
        os.chdir("D:\\data_aquisition\\song")
        
        
        if dataset=='song':
            X1 = genfromtxt('song_frontal_3.8_X.txt', delimiter='')
            y1 = genfromtxt('song_frontal_3.8_Y.txt',delimiter=',')
            X2 = genfromtxt('song_frontal_4.4_X.txt', delimiter='')
            y2 = genfromtxt('song_frontal_4.4_Y.txt',delimiter=',')       
            X3 = genfromtxt('song_frontal_5.0_X.txt', delimiter='')
            y3 = genfromtxt('song_frontal_5.0_Y.txt',delimiter=',')
            X4 = genfromtxt('song_frontal_5.5_X.txt', delimiter='')
            y4 = genfromtxt('song_frontal_5.5_Y.txt',delimiter=',')
            X5 = genfromtxt('song_frontal_6.1_X.txt', delimiter='')
            y5 = genfromtxt('song_frontal_6.1_Y.txt',delimiter=',')
            
            X = np.concatenate((X1,X2,X3,X4,X5))
            y = np.concatenate((y1,y2,y3,y4,y5))        
        
        
        
        if dataset == 'song_frontal_3.8':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'song_frontal_4.4':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'song_frontal_5.0':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'song_frontal_5.5':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'song_frontal_6.1':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
            
            
    elif 'jun' in dataset:
        os.chdir("D:\\data_aquisition\\jun")
        
        
        if dataset=='jun':
            X1 = genfromtxt('jun_frontal_3.9_X.txt', delimiter='')
            y1 = genfromtxt('jun_frontal_3.9_Y.txt',delimiter=',')
            X2 = genfromtxt('jun_frontal_5.1_X.txt', delimiter='')
            y2 = genfromtxt('jun_frontal_5.1_Y.txt',delimiter=',')       
            X3 = genfromtxt('jun_frontal_5.7_X.txt', delimiter='')
            y3 = genfromtxt('jun_frontal_5.7_Y.txt',delimiter=',')
            X4 = genfromtxt('jun_frontal_6.3_X.txt', delimiter='')
            y4 = genfromtxt('jun_frontal_6.3_Y.txt',delimiter=',')

            X = np.concatenate((X1,X2,X3,X4))
            y = np.concatenate((y1,y2,y3,y4))
        
        
        
        
        if dataset == 'jun_frontal_3.9':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'song_frontal_4.4':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'jun_frontal_5.1':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'jun_frontal_5.7':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'jun_frontal_6.3':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
         
    elif 'hao' in dataset:
        os.chdir("D:\\data_aquisition\\hao")
        
        if dataset=='hao':
            X1 = genfromtxt('hao_frontal_4.0_X.txt', delimiter='')
            y1 = genfromtxt('hao_frontal_4.0_Y.txt',delimiter=',')
            X2 = genfromtxt('hao_frontal_4.6_X.txt', delimiter='')
            y2 = genfromtxt('hao_frontal_4.6_Y.txt',delimiter=',')       
            X3 = genfromtxt('hao_frontal_5.2_X.txt', delimiter='')
            y3 = genfromtxt('hao_frontal_5.2_Y.txt',delimiter=',')
            X4 = genfromtxt('hao_frontal_5.8_X.txt', delimiter='')
            y4 = genfromtxt('hao_frontal_5.8_Y.txt',delimiter=',')
            X5 = genfromtxt('hao_frontal_6.4_X.txt', delimiter='')
            y5 = genfromtxt('hao_frontal_6.4_Y.txt',delimiter=',')
            
            X = np.concatenate((X1,X2,X3,X4,X5))
            y = np.concatenate((y1,y2,y3,y4,y5))
        
        if dataset == 'hao_frontal_4.0':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'hao_frontal_4.6':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'hao_frontal_5.2':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'hao_frontal_5.8':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'hao_frontal_6.4':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
    elif 'qing' in dataset:
        os.chdir("D:\\data_aquisition\\qing")
        
        if dataset=='qing':
            X1 = genfromtxt('qing_frontal_3.7_X.txt', delimiter='')
            y1 = genfromtxt('qing_frontal_3.7_Y.txt',delimiter=',')
            X2 = genfromtxt('qing_frontal_4.2_X.txt', delimiter='')
            y2 = genfromtxt('qing_frontal_4.2_Y.txt',delimiter=',')       
            X3 = genfromtxt('qing_frontal_4.7_X.txt', delimiter='')
            y3 = genfromtxt('qing_frontal_4.7_Y.txt',delimiter=',')
            X4 = genfromtxt('qing_frontal_5.2_X.txt', delimiter='')
            y4 = genfromtxt('qing_frontal_5.2_Y.txt',delimiter=',')
            X5 = genfromtxt('qing_frontal_5.7_X.txt', delimiter='')
            y5 = genfromtxt('qing_frontal_5.7_Y.txt',delimiter=',')
            
            X = np.concatenate((X1,X2,X3,X4,X5))
            y = np.concatenate((y1,y2,y3,y4,y5))
        
        
        
        if dataset == 'qing_frontal_3.7':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'qing_frontal_4.2':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'qing_frontal_4.7':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'qing_frontal_5.2':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'qing_frontal_5.7':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')   
            
    elif 'yue' in dataset:
        os.chdir("D:\\data_aquisition\\yue")
        
        if dataset=='yue':
            X1 = genfromtxt('yue_frontal_4.0_X.txt', delimiter='')
            y1 = genfromtxt('yue_frontal_4.0_Y.txt',delimiter=',')
            X2 = genfromtxt('yue_frontal_4.6_X.txt', delimiter='')
            y2 = genfromtxt('yue_frontal_4.6_Y.txt',delimiter=',')       
            X3 = genfromtxt('yue_frontal_5.2_X.txt', delimiter='')
            y3 = genfromtxt('yue_frontal_5.2_Y.txt',delimiter=',')
            X4 = genfromtxt('yue_frontal_5.8_X.txt', delimiter='')
            y4 = genfromtxt('yue_frontal_5.8_Y.txt',delimiter=',')
            X5 = genfromtxt('yue_frontal_6.4_X.txt', delimiter='')
            y5 = genfromtxt('yue_frontal_6.4_Y.txt',delimiter=',')
            
            X = np.concatenate((X1,X2,X3,X4,X5))
            y = np.concatenate((y1,y2,y3,y4,y5))
        
        if dataset == 'yue_frontal_4.0':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'yue_frontal_4.6':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'yue_frontal_5.2':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'yue_frontal_5.8':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')
            
        if dataset == 'yue_frontal_6.4':
            
            X= genfromtxt(dataset+'_X.txt', delimiter='')
            y= genfromtxt(dataset+'_Y.txt',delimiter=',')         

            
            
    else:
        
        print("No such subject!")
        X = np.zeros(0)
        y = np.zeros(0)
        

        
        
    return X,y

