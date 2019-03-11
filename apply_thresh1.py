# Convert training images to filtered versions
import cv2
import glob
from PIL import ImageOps, Image
import os, sys, shutil, os.path
import numpy

try:
    os.chdir("Datasets\osd\Dataset")
except:
    pass
chdir = 1
while chdir:
    try:
        cur_dir = os.getcwd()
        print(" Current directory is " + cur_dir)
        print("Please enter the path to the directory of training data you wish to filter. This directory should only contain unfiltered image files.")
        dirname = input()
        os.chdir(dirname)
    except:
        print("invalid directory!")
    else:
        print("Chosen directory is " + os.getcwd())
        print("Is this correct? [yes/no]")
        yes = input()
        if yes=="yes":
            chdir = 0

images = []
save_dir = os.getcwd() + "\\filtered"
print("Save directory set as " + save_dir)
print("Press [ENTER] to continue...")
x = input()
try:
    os.makedirs(save_dir)
except Exception as e:
    print(e)

for filename in glob.glob(os.getcwd() + '\*.png'):
    #print(filename)
    #img = numpy.asarray(Image.open(filename))
    img = cv2.imread(filename)
    #img = img.convert('1')
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    save_file = filename.split('\\')
    print("\nFiltered: ", save_file[-1])
    cv2.imwrite(os.getcwd() + "\\filtered\\" + save_file[-1], thresh1)

