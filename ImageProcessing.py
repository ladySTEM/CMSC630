import cv2
import numpy as np
import matplotlib.pyplot as plt
import random 
import math
import os 
import argparse
import time
from collections import Counter
import json
from scipy.signal import convolve2d
from scipy.signal import find_peaks


class ImageProcessing():
    def __init__(self,height,width):
        #super().__init__('image_processing')
        self.statement = "myworld"
        self.width = width
        self.height = height
        self.pixel_range = 255

    def rgb2gray(self,img):
        new_img = np.zeros((self.height,self.width),np.uint8)
        for i in range(self.height):
            for j in range(self.width):
                blue,green,red = img[i,j]
                bw = int((0.299*blue+0.587*green+0.114*red))
                new_img[i,j] = bw
        return new_img

    def calc_histogram(self,img): 
        h = np.zeros((self.pixel_range+1)) 
        for i in range(self.height):
            for j in range(self.width):
                h[img[i,j]] = h[img[i,j]]+1        
        return h

    def equalize_histogram(self,img,h):
        e_hist = np.zeros((h.size))
        cdf_n = 0
        new_img = np.zeros((self.height,self.width),np.uint8)
        for index,p_intensity in enumerate(h):
            p_n = p_intensity/(self.height * self.width)
            cdf_n = p_n+cdf_n
            e_hist[index] = round(self.pixel_range*cdf_n)
        for i in range(self.height):
            for j in range(self.width):
                pix_val = img[i,j]
                new_img[i,j] = e_hist[pix_val]

        return new_img,e_hist 

    def salt_and_pepper(self,img,strength): 
        new_img = img
        coverage_w = round(strength*(self.height*self.width))
        coverage_b = round(strength*(self.height*self.width))
        for i in range(coverage_w):
            x_w = random.randint(0,self.width-1)
            y_w = random.randint(0,self.height-1)
            if img[y_w,x_w] != 0:
             new_img[y_w,x_w] = 255
        for i in range(coverage_b):
            x_b = random.randint(0,self.width-1)
            y_b = random.randint(0,self.height-1)
            if img[y_b,x_b] != 255:
             new_img[y_b,x_b] = 0
            
           
        return new_img


    def gauss_noise(self,img,strength,var):
        new_img = img
        sigma = np.sqrt(var)
        n = np.random.normal(loc=0,
                             scale=sigma,
                             size=(self.height,self.width))
        new_img = img+strength*n

        return new_img

    def save_img(self,img,string):
       #newimg = img[0:self.height-1,0:self.width-1]
       cv2.imwrite(string,img)

    def printmyworld(self):
        print(self.statement)
    
    def create_mask(self,w,h,sigma):
        pi = math.pi
        hy = np.array(range(int(-(h-1)/2),int((h-1)/2+1))) 
        hx = np.array(range(int(-(w-1)/2),int((w-1)/2+1)))
        gy = np.zeros(hy.shape)
        gx = np.zeros(hx.shape)
        for i_val_hy,j_val_hx in zip(enumerate(hy),enumerate(hx)):
            i,val_hy = i_val_hy
            j,val_hx = j_val_hx
            gx[i] = (((2*pi)**0.5)*sigma)**(-1)*math.exp(-(val_hx**2)/(2*sigma**2))
            gy[j] = (((2*pi)**0.5)*sigma)**(-1)*math.exp(-(val_hy**2)/(2*sigma**2))
        
        return gx,gy,hx,hy

    def linear_filter(self,img,w,h,sigma):
        
        gx,gy,hx,hy = self.create_mask(w,h,sigma)
        radius_w = int(math.floor(w/2))
        radius_h = int(math.floor(h/2))
        new_img = np.zeros((self.height,self.width))

        for i in range(radius_h,self.height-radius_h):
            for j in range(self.width):
                sum = 0
                for index,y in enumerate(hy):
                   sum = img[i+y,j]*gy[index]+sum
                sum = sum/(index+1)
                new_img[i - radius_h,j] = sum
        for i in range(self.height):
            for j in range(radius_w,self.width-radius_w):
                sum = 0
                for index,x in enumerate(hx):
                    sum = img[i,j+x]*gx[index] + sum
                new_img[i,j-radius_w] = sum

        return new_img

    def median_filter(self,img,pixel_weights):
        h,w = pixel_weights.shape
        new_img = np.zeros((self.height,self.width))
        radius_w = int(math.floor(w/2))
        radius_h = int(math.floor(h/2))
    
        for i in range(radius_h,self.height-radius_h):
            for j in range(radius_w,self.width-radius_w):
                array = []
                for a in range(i-radius_h,i+radius_h+1):
                    for b in range(j-radius_w,j+radius_w+1):                 
                        pw_index_y= int(a-(i-radius_h))
                        pw_index_x =int(b-(j-radius_w))
                        val = img[a,b]
                        times = pixel_weights[pw_index_y,pw_index_x]
                        temp_array = np.repeat(val,times)
                        array = np.append(array,temp_array)
                array.sort()
                middle_element = array[len(array) //2]
                new_img[i,j] = middle_element
                
    def find_edges(self,img):
        """
        This function uses compass method to find the edges of the image.
        
        Inputs: 
            img - 2D array of image
        Outputs: 
            mydisk - array of structure element
        """
        
        new_img = np.zeros((self.height,self.width))
        mat1 = []
        mat2 = []
        
        
        H4 = np.array([[3, 0, -3],
                       [10, 0, -10],
                       [3, 0, -3]])
        
        H5 = np.array([[10,3,0],
                       [3,0,-3],
                       [0,-3,-10]])
        
        H6 = np.array([[3, 10, 3],
                       [0, 0, 0],
                       [-3, -10, -3]])
        
        H7 = np.array([[0, 3, 10],
                       [-3, 0, 3],
                       [-10, -3, 0]])
        
        grad0 = convolve2d(img,H4,boundary='symm', mode='same')
        grad1 = convolve2d(img,H5,boundary='symm', mode='same')
        grad2 = convolve2d(img,H6,boundary='symm', mode='same')
        grad3 = convolve2d(img,H7,boundary='symm', mode='same')
        grad4 = convolve2d(img,-1*H4,boundary='symm', mode='same')
        grad5 = convolve2d(img,-1*H5,boundary='symm', mode='same')
        grad6 = convolve2d(img,-1*H6,boundary='symm', mode='same')
        grad7 = convolve2d(img,-1*H7,boundary='symm', mode='same')
        
        new_img = np.ones_like(grad0)
        
        for i in range(grad0.shape[0]):
            for j in range(grad0.shape[1]):
                new_img[i,j] = np.sqrt(grad0[i,j]**2+grad1[i,j]**2+grad2[i,j]**2+grad3[i,j]**2+grad4[i,j]**2+grad5[i,j]**2+grad6[i,j]**2+grad7[i,j]**2)

        threshold = 200
        edges = np.zeros_like(new_img)
        edges[new_img > threshold] = 255

        return edges
    
    def create_disk(self,r):
        """
        This function creates the structure element used to erode or dilate the objects in the image
        
        Inputs: 
            r - radius of structure element (element is set to disk shape)
        Outputs: 
            mydisk - array of structure element
        
        """
        mat_size = 2*r+1
        mydisk = np.ones((mat_size,mat_size),dtype=int)
        mydisk[0,0]=0
        mydisk[0,mat_size-1] = 0
        mydisk[mat_size-1,0] = 0
        mydisk[mat_size-1,mat_size-1] = 0
        return mydisk
    
    def erosion(self,img,r,iterations):
        """
        This function uses erosion to shrink the objects in the image, getting rid of small noise 
        
        Inputs: 
            img - 2D array containig pixel values
            r - radius of structure element (element is set to disk shape)
            iterations - depicts number of iterations that dilation should be set to
        Outputs: 
            dilated - image array of dilated image
        
        """
        disk = self.create_disk(r)
        new_img = convolve2d(img,disk,boundary='symm', mode='same')
        eroded = np.zeros_like(new_img)
        value = np.sum(255*disk)
        eroded[new_img >= value] = 255
        
        if iterations > 1:
            for i in range(iterations-2):
                new_img = convolve2d(eroded,disk,boundary='symm', mode='same')
                eroded = np.zeros_like(new_img)
                value = np.sum(255*disk)
                eroded[new_img >= value] = 255
        
        
        return eroded
    
    def dilation(self,img,r,iterations):
        """
        This function uses dilation to grow the objects in the image based on a given structure element
        
        Inputs: 
            img - 2D array containig pixel values
            r - radius of structure element (element is set to disk shape)
            iterations - depicts number of iterations that dilation should be set to
        Outputs: 
            dilated - image array of dilated image
        
        """
        disk = self.create_disk(r)
        new_img = convolve2d(img,disk,boundary='symm', mode='same')
        dilated = np.zeros_like(new_img)
        dilated[new_img > 0] = 255
        
        if iterations > 1:
            for i in range(iterations-2):
                new_img = convolve2d(dilated,disk,boundary='symm', mode='same')
                dilated = np.zeros_like(new_img)
                dilated[new_img > 0] = 255
        
        return dilated
                    
        
    def gray_threshold(self,img):
        """
        This function uses gray thresholding to split the input image into background and foreground portions
        
        Inputs: 
            img - 2D array containing pixel values
            
        Outputs: 
            foreground - new image array containing only foreground pixels
            background - new image array containing only background pixels
        """
        
        h = self.calc_histogram(img)+1
        N = self.width*self.height
        var_within = []
        for T in range(256): #for threshold T
            p_o = p_b = mean_o = mean_b = var_o = var_b = 0
            
            for i in range(256): #for intensity i
                P_i = h[i]/N
                if i<=T:
                    p_o = p_o + P_i
                    mean_o = mean_o+i*P_i/p_o
                    var_o = var_o +((i-mean_o)**2)*P_i/p_o
                else:
                    p_b = p_b+P_i
                    mean_b = mean_b+i*P_i/p_b
                    var_b = var_b+((i-mean_b)**2)*P_i/p_b
            var_w = var_o*p_o+var_b*p_b
            var_within.append(var_w)
        var_min = np.amin(var_within)
        threshold = np.where(var_within == var_min)[0]
        
        foreground = img.copy()
        background = img.copy()
        foreground[img > threshold] = 0
        background[img <= threshold] = 0
        
      
        
        return foreground,background
    
    def clustering(self,img, k):
        """
        This function uses k_means clustering to separate the image's pixels into k different intensities
        
        Inputs:
            img - 2D array containing pixel grayscale values
            k - constant that determines how many pixel value clusters the image should be organized into
        Outputs: 
            new_img - 2D array containing clustered image values
            
        """

        k_arr = []
        error = 10 #initialize error (just needs to be greater than while value)
        
        h = self.calc_histogram(img)
        
        peaks = find_peaks(h)
        
        x = np.linspace(0, len(peaks[0])-1,k,dtype=int)
        print("peaks: {} length: {}\n x:{}\n".format(peaks,len(peaks[0]),x))
        
        #Find k number of random pixel values
        #initialize classes which will hold cluster data
        clusters = {}
        for iter_k in range(k):
            k_arr.append(peaks[0][x[iter_k]])#img[rand_i,rand_j])
            clusters.setdefault(iter_k,[]) #initialize cluster class

        #Find distance between pixels and centroid
        #Pixels are organized into clusters based on the centroid that is closest 
        while error > 1:
            for i in range(self.height-1):
                for j in range(self.width-1):   #for first 5 pixels
                    pixel = img[i,j]
                    distance = []
                    for iter_k in range(k):
                        myk = float(k_arr[iter_k])
                        new_distance = np.linalg.norm(myk-pixel)
                        distance.append(new_distance) 

                    cluster_index = np.argmin(distance)
                    clusters[cluster_index].append(pixel)
            #Find mean value of clusters, and reassign centroid to this mean. 
            # Error is the magnitude difference between new K value and old K value
            for iter_k in range(k):
                #if clusters[iter_k].length == 0:
               # print("K-Value: {}\nCluster {} Arr: {}\n".format(k_arr[iter_k],iter_k,clusters[iter_k]))   
                error = np.linalg.norm(k_arr[iter_k] - np.mean(clusters[iter_k],axis=0))
                k_arr[iter_k] = int(np.floor(np.mean(clusters[iter_k],axis=0))) #new k values

        new_img = img.copy()
        
        #Assign new image values based on which cluster their old values are a part of 
        for iter_k in range(k):
            mask = np.apply_along_axis(np.in1d,axis=1, arr= img, ar2= clusters[iter_k])
            new_img = np.where(mask, k_arr[iter_k], new_img)
     
  
        return new_img