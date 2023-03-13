"""
CMSC 630
Semester Project Part 1
Gabriella Graziani

"""

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

        return new_img
 

def main(input_file,sp_strength,gauss_noise_strength,gauss_noise_var,median_array,
         lin_fil_w,lin_fil_h,lin_fil_sigma,img_height,img_width):

    image_processing = ImageProcessing(img_height,img_width)
    parent_dir = os.getcwd() #get current working directory
    path = os.path.join(parent_dir,input_file) #add input file
    all_images = os.listdir(path) #collect name of all images in file

    #Create output files
    file_list = ['Grayscale','SPNoise','GaussianNoise','Histograms',
            'EqualizedHistograms','LinearFilter','MedianFilter','AveragedHistograms']
    for item in file_list:
        os.makedirs(item,exist_ok = True)

    #make all images b&w
    print("BEGINNING: RGB to Grayscale \n")
    st_batch = time.time()
    img_tm = []
    for item in all_images:
        st = time.time()
        img_path = 'Cancerous cell smears/'+item
        img = cv2.imread(img_path)
        bw_img = image_processing.rgb2gray(img)
        image_processing.save_img(bw_img,"Grayscale/"+item)
        et = time.time()
        exec = et-st
        img_tm.append(exec)
    avg_tm = sum(img_tm)/len(img_tm)
    et_batch = time.time()
    bw_tm_batch = et_batch-st_batch

    print("RGB to Grayscale - Batch Execution time: {} Time per image: {}\n".format(bw_tm_batch,avg_tm))
   
    #find histograms of all images and equalize
    print("BEGINNING: Find histograms\n")

    st_batch = time.time()
    img_tm = []
    for item in all_images:
        st = time.time()
        img_path = 'Grayscale/'+item
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        histogram = image_processing.calc_histogram(img)
        file = "Histograms/"+item[:-4]+".txt"
        np.savetxt(file,histogram)
        x = np.arange(256)
        plt.bar(x,histogram)
        plt.title(item[:-4]+" Histogram")
        plt.savefig("Histograms/"+item[:-4]+".png")
        plt.clf()
        et = time.time()
        exec = et-st
        img_tm.append(exec)
    avg_tm = sum(img_tm)/len(img_tm)
    et_batch = time.time()
    hist_tm_batch = et_batch-st_batch
    print("Find histograms - Batch Execution time: {} Time per image: {}\n".format(hist_tm_batch,avg_tm))

    #find equalized histogram
    print("BEGINNING: Equalize histograms\n")

    st_batch = time.time()
    img_tm = []
    for item in all_images:
        st = time.time()
        img_path = 'Grayscale/'+item
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        hist = np.loadtxt("Histograms/"+item[:-4]+".txt",dtype=float)
        eq_img,eq_hist = image_processing.equalize_histogram(img,hist)
        file = "EqualizedHistograms/"+item[:-4]+".txt"
        np.savetxt(file,histogram)
        x = np.arange(256)
        plt.bar(x,eq_hist)
        plt.title(item[:-4]+" Histogram")
        plt.savefig("EqualizedHistograms/"+item[:-4]+".png")
        plt.clf()
        image_processing.save_img(eq_img,"EqualizedHistograms/"+item)
        et = time.time()
        exec = et-st
        img_tm.append(exec)
    avg_tm = sum(img_tm)/len(img_tm)
    et_batch = time.time()
    equalized_tm_batch = et_batch-st_batch

    print("Equalize histograms - Batch Execution time: {} Time per image: {}\n".format(equalized_tm_batch,avg_tm))

    #average histogram for each class
    print("BEGINNING: Average histogram for class process\n")

    st_batch = time.time()
    classes = {}
    for item in all_images:
        prefix = item[:3]
        classes.setdefault(prefix,[]).append(item) #create dictionary containing image names pertaining to class
    for a_class in classes:
        new_hist = []
        hist_array = np.empty((len(classes[a_class]),256))
        for index,item in enumerate(classes[a_class]):
            hist = np.loadtxt("Histograms/"+item[:-4]+".txt",dtype=float)
            hist_array[index,:] = hist
        rows,cols = hist_array.shape
        for col in range(cols):
            myavg = sum(hist_array[:,col])/(rows)
            new_hist.append(myavg)
        file = "AveragedHistograms/Class_"+a_class+".txt"
        np.savetxt(file,new_hist)
        x = np.arange(256)
        plt.bar(x,new_hist)
        plt.title("Class "+a_class+" Histogram")
        plt.savefig("AveragedHistograms/Class_"+a_class+".png")
        plt.clf()
    et_batch = time.time()
    avhist_tm_batch = et_batch-st_batch
  
    print("Average histogram for class process - Batch Execution time:{}\n".format(avhist_tm_batch))

    #add salt and pepper noise
    print("BEGINNING: Add salt and pepper noise \n")

    st_batch = time.time()
    img_tm = []
    for item in all_images:
        st = time.time()
        img_path = 'Grayscale/'+item
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        sp_img = image_processing.salt_and_pepper(img,sp_strength) 
        image_processing.save_img(sp_img,"SPNoise/"+item)
        et = time.time()
        exec = et-st
        img_tm.append(exec)
    avg_tm = sum(img_tm)/len(img_tm)
    et_batch = time.time()
    sp_tm_batch = et_batch-st_batch

    print("Add salt and pepper noise - Batch Execution time: {} Time per image: {}\n".format(sp_tm_batch,avg_tm))

    #add gaussian noise 
    print("BEGINNING: Add gaussian noise \n")

    st_batch = time.time()
    img_tm = []
    for item in all_images:
        st = time.time()
        img_path = 'Grayscale/'+item
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        gn_img = image_processing.gauss_noise(img,gauss_noise_strength,gauss_noise_var)
        image_processing.save_img(gn_img,"GaussianNoise/"+item)
        et = time.time()
        exec = et-st
        img_tm.append(exec)
    avg_tm = sum(img_tm)/len(img_tm)
    et_batch = time.time()
    gn_tm_batch = et_batch-st_batch

    print("Add gaussian noise - Batch Execution time: {} Time per image: {}\n".format(gn_tm_batch,avg_tm))

    #median filter
    print("BEGINNING: Apply median filter to sp and gaussian noise images \n")

    st_batch = time.time()
    img_tm = []
    for item in all_images:
        st = time.time()
        sp_img_path = 'SPNoise/'+item
        gn_img_path = 'GaussianNoise/'+item
        sp_img = cv2.imread(sp_img_path,cv2.IMREAD_GRAYSCALE)
        gn_img = cv2.imread(gn_img_path,cv2.IMREAD_GRAYSCALE)    
        array = np.array(median_array) 
        medf_sp_img = image_processing.median_filter(sp_img,array)
        medf_gn_img = image_processing.median_filter(gn_img,array)
        image_processing.save_img(medf_sp_img,"MedianFilter/sp_noise_"+item)
        image_processing.save_img(medf_gn_img,"MedianFilter/gauss_noise_"+item)
        et = time.time()
        exec = et-st
        img_tm.append(exec)
    avg_tm = sum(img_tm)/len(img_tm)
    et_batch = time.time()
    med_tm_batch = et_batch - st_batch

    print("Apply median filter to sp and gaussian noise images - Batch Execution time: {} Time per image: {}\n".format(med_tm_batch,avg_tm))

    #linear filter
    print("BEGINNING: Apply linear filter to sp and gaussian noise images \n")
    st_batch = time.time()
    img_tm = []
    for item in all_images:
        st = time.time()
        sp_img_path = 'SPNoise/'+item
        gn_img_path = 'GaussianNoise/'+item
        sp_img = cv2.imread(sp_img_path,cv2.IMREAD_GRAYSCALE)
        gn_img = cv2.imread(gn_img_path,cv2.IMREAD_GRAYSCALE)  
        linf_sp_img = image_processing.linear_filter(sp_img,lin_fil_w,lin_fil_h,lin_fil_sigma)
        linf_gn_img = image_processing.linear_filter(gn_img,lin_fil_w,lin_fil_h,lin_fil_sigma)
        image_processing.save_img(linf_sp_img,"LinearFilter/sp_noise_"+item)
        image_processing.save_img(linf_gn_img,"LinearFilter/gauss_noise_"+item)
        et = time.time()
        exec = et-st
        img_tm.append(exec)
    avg_tm = sum(img_tm)/len(img_tm)
    et_batch = time.time()
    linf_tm_batch = et_batch - st_batch

    print("Apply linear filter to sp and gaussian noise images - Batch Execution time: {} Time per image: {}\n".format(linf_tm_batch,avg_tm))
   
    

if __name__ == '__main__':
    with open('config.json','r') as f:
        config = json.load(f)
    main(**config)