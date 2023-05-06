"""

CMSC 630 - Part 3
Gabriella Graziani

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random 
import math
from collections import Counter
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
        size = img.shape
        height = size[0]
        width = size[1]
        h = np.zeros((self.pixel_range+1)) 
        for i in range(height-1):
            for j in range(width-1):
                h[img[i,j]] = h[img[i,j]]+1        
        return h

    def equalize_histogram(self,img):
        
        r,g,b = cv2.split(img)
        
        hr = self.calc_histogram(r)
        hg = self.calc_histogram(g)
        hb = self.calc_histogram(b)
        e_histr = np.zeros((hr.size))
        e_histg = np.zeros((hg.size))
        e_histb = np.zeros((hb.size))
        cdf_n = 0
        
        new_img_r = r.copy()
        new_img_g = g.copy()
        new_img_b = b.copy()
        
        for index,p_intensity in enumerate(hr):
            p_n = p_intensity/(self.height * self.width)
            cdf_n = p_n+cdf_n
            e_histr[index] = round(self.pixel_range*cdf_n)
            
        cdf_n = 0
            
        for index,p_intensity in enumerate(hg):
            p_n = p_intensity/(self.height * self.width)
            cdf_n = p_n+cdf_n
            e_histg[index] = round(self.pixel_range*cdf_n)
            
        cdf_n = 0
        
        for index,p_intensity in enumerate(hb):
            p_n = p_intensity/(self.height * self.width)
            cdf_n = p_n+cdf_n
            e_histb[index] = round(self.pixel_range*cdf_n)
            
        for i in range(self.height):
            for j in range(self.width):
                r_pix_val = r[i,j]
                g_pix_val = g[i,j]
                b_pix_val = b[i,j]
                new_img_r[i,j] = e_histr[r_pix_val]
                new_img_g[i,j] = e_histg[g_pix_val]
                new_img_b[i,j] = e_histb[b_pix_val]
                
        new_img = cv2.merge((new_img_r,new_img_g,new_img_b))

        return new_img

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
        """
        This functions performs a Gaussia filter using a given input kernel
        
        Inputs: 
            img - rgb array of image
        Outputs: 
            new_img - rgb array with applied gaussian filter
        """
        r,g,b = cv2.split(img)
        pixel_weights = np.array(pixel_weights)

        new_img_r = convolve2d(r,pixel_weights,boundary='symm', mode='same')
        new_img_g = convolve2d(g,pixel_weights,boundary='symm', mode='same')
        new_img_b = convolve2d(b,pixel_weights,boundary='symm', mode='same')
        
        
        new_img = cv2.merge((new_img_r,new_img_g,new_img_b))/np.sum(pixel_weights)
        new_img = new_img.astype(int)
    
        return new_img
                
    def find_edges(self,img):
        """
        This function uses compass method to find the edges of the image.
        
        Inputs: 
            img - 2D array of image
        Outputs: 
            mydisk - array of structure element
        """
        height, width = img.shape
        
        #print("Img height: {} Img Width: {} \n".format(height,width))
        
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
        print("Threshold: {}\n".format(threshold))
        foreground = img.copy()
        background = img.copy()
        foreground[img > threshold] = 255
        foreground[img<= threshold] = 0
        background[img <= threshold] = 255
        background[img > threshold] = 0
        
      
        
        return foreground,background
    
    def calc_range(self,mean,iter_k,scale):
        """
        This function is used in the clustering algorithm to determine 
        the pixels ranges the centroids should be initialized within
        
        Inputs: 
            mean - center point of the initialization
            iter_k - centroid that the range is being applied for
            scale - the deviation from the mean value 
            
        Outputs: 
            beg - begining index of range
            end - ending index of range
        """
        beg = mean+(iter_k*scale-scale)
        end = mean+(iter_k*scale)
        
        if beg <0: 
            beg = 0
        if end > 255:
            end = 255
            
        return beg,end
   
    def clustering(self,img,k):
        """
        This function is the k_means clustering algorithm and separate a rgb image into k many rgb colors
        
        Inputs: 
            img - rgb input image array
            k - value which determines the number of rgb clusters
        Outputs: 
            rgb - resulting clustere image
        """
        r,g,b = np.split(img,3,axis=2)
        
        
        
        r_mean = 150#self.mean_h(r)
        g_mean = 150#self.mean_h(g)
        b_mean = 150#self.mean_h(b)
       
        
        centroid = [0,0,0]
        label = []
        error = 10
        #clusters = {}
        
        error = 1000
        error_diff = 1000
       # img = np.interp(img,(img.min(),img.max()),(0,1))
        
        scale = 50
        
        for iter_k in range(k):
            beg_r,end_r = self.calc_range(r_mean,iter_k,scale)
            beg_g,end_g = self.calc_range(g_mean,iter_k,scale)
            beg_b,end_b = self.calc_range(b_mean,iter_k,scale)
      
            
           # print("Beg r: {} End r: {} mean: {} ".format(beg_r,end_r,r_mean))
            r_val =random.randint(beg_r,end_r)
            g_val = random.randint(beg_g,end_g)
            b_val = random.randint(beg_b,end_b)
            centroid = np.vstack([centroid,[r_val,g_val,b_val]])
        centroid = np.delete(centroid,0,axis=0)
       # print("OG Centroid: {}".format(centroid))
        label = np.zeros_like(r)
        
        while error_diff > 40: 

            for i in range(self.height-1):
                for j in range(self.width-1):
                    pixel = np.array([r[i,j],g[i,j],b[i,j]])
                    distance = []
                    for iter_k in range(k):
                        new_distance = np.linalg.norm(pixel-centroid[iter_k])
                        distance = np.append(distance,new_distance)
                    cluster_index = np.argmin(distance)
                  #  print("Distance: {} Cluster Index: {}".format(distance,cluster_index))
                    label[i,j] = cluster_index
            
            for iter_k in range(k):
                indices = np.where(label == iter_k)
            
                mean_r = int(np.mean(r[indices]))
                mean_g = int(np.mean(g[indices]))
                mean_b = int(np.mean(b[indices]))
                                      
                new_centroid = np.array([mean_r,mean_g,mean_b])
                old_error = error
                error = np.linalg.norm(new_centroid-centroid)
                
                error_diff = np.linalg.norm(old_error-error)
                centroid[iter_k] = new_centroid
            
        new_r = np.zeros_like(r)
        new_g = np.zeros_like(b)
        new_b = np.zeros_like(g)
        rgb = np.zeros_like(img)
        
       
        
        for iter_k in range(k):
            new_r[label == iter_k] = centroid[iter_k][0]
            new_g[label == iter_k] = centroid[iter_k][1]
            new_b[label == iter_k] = centroid[iter_k][2]
            
            
 
        rgb = np.stack((new_r[:, :, 0], new_g[:, :, 0], new_b[:, :, 0]), axis=2)
    
        return rgb#,binary_img
                                      
 
    
    def cluster_threshold(self,img):
        """
        This function thresholds the input grayscale image into a foreground and background binary image
        
        Inputs: 
            img - 2d grayscale image array
        Outputs: 
            new_img - binary image with foreground containing white pixels and background containing black pixels
        """
        
        
        h = self.calc_histogram(img)
        
        indexmax = np.argmax(h)
        
        indexmin = np.argmin(h)
        
        new_img = 255*np.ones_like(img)
        new_img[img == indexmax] =  0
 
        
        return new_img 
    
    def extract_features(self,edge_img,binary_img,img):
        """
        This function extracts features from the binary edge image, binary foreground image, and color image. 
        
        Inputs: 
            edge_img - 2d binary edge image
            binary_img - 2d binary foreground/background image
            img - 2d rgb image
        Outputs: 
            features - an array of features extracted from image. Includes perimeter,
            area, mean rgb, and max histogram intensity value
        """
        #Find Perimeter
        perimeter = int(np.sum(img[10:self.height-1,10:self.width-10])/255)
        
        #Find Area
        area = int(np.sum(binary_img[10:self.height-1,10:self.width-10])/255)
        col_measure = []
        row_measure = []
        
        #Find mean color from most dense foreground of image
        my_range  = 50
        for j in range(60,self.width-60):
            
            col_sum = np.sum(binary_img[:,j])
            col_measure= np.append(col_measure,col_sum)
            
        for i in range(60,self.height-60):
            row_sum = np.sum(binary_img[i,:])
            row_measure= np.append(row_measure,row_sum)
        max_x = np.argmax(col_measure)+60
        max_y = np.argmax(row_measure)+60
        
        window = img[max_y-my_range:max_y+my_range,max_x-my_range:max_x+my_range,:]
        
    
        r,g,b = np.split(window,3,axis=2) 
        
        mean_r = int(np.mean(r))
        mean_g = int(np.mean(g))
        mean_b = int(np.mean(b))
        
    
        
        #Find histogram from most dense foreground of image
        window = img[max_y-my_range:max_y+my_range,max_x-my_range:max_x+my_range]
        h = self.calc_histogram(window)
        
        max_h_val = np.argmax(h)
        
      
        features = [perimeter,area,mean_r,mean_g,mean_b,max_h_val]
        

        
        return features
        
        
                
                
    