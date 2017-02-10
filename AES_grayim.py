# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:44:47 2017

@author: Niramai_Siva
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 00:16:07 2017

@author: Niramai_Siva
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import PIL
import math
import numpy as np

from PIL import Image
from Crypto.Cipher import AES
import hashlib
import binascii

global password 
password = hashlib.sha256("dummy is the".encode('utf-8')).digest()
# initialize variables
plaintext = list()
plaintextstr = ""
imagename='C:\\Users\\Niramai Siva\\Downloads\\images.png';
# load the image
im = Image.open(imagename).convert('L')
pix = im.load()
#print im.size   # print size of image (width,height)
width = im.size[0]
height = im.size[1]
# break up the image into a list, each with pixel values and then append to a string
for y in range(0,height):
    #print("Row: %d") %y  # print row number
    for x in range(0,width):
        #print pix[x,y]  # print each pixel RGB tuple
        plaintext.append(pix[x,y])
 
# add 100 to each tuple value to make sure each are 3 digits long.  being able to do this is really just a PoC 
# that you'll be able to use a raw application of RSA to encrypt, rather than PyCrypto if you wanted.
for i in range(0,len(plaintext)):
        plaintextstr = plaintextstr + "%d" %(int(plaintext[i])+100)


# length save for encrypted image reconstruction
relength = len(plaintext)

# append dimensions of image for reconstruction after decryption
plaintextstr += "h" + str(height) + "h" + "w" + str(width) + "w"

# make sure that plantextstr length is a multiple of 16 for AES.  if not, append "n".  not safe in theory
# and i should probably replace this with an initialization vector IV = 16 * '\x00' at some point.  In practice
# this IV buffer should be random.
while (len(plaintextstr) % 16 != 0):
    plaintextstr = plaintextstr + "n"

# encrypt plaintext
obj = AES.new(password, AES.MODE_CBC, 'This is an IV456')
ciphertext = obj.encrypt(plaintextstr)
#ciphertext = binascii.hexlify(ciphertext)

imagedata=[int.from_bytes(ciphertext[i:i+3],byteorder='big') for i in range(0, 3*int(relength), 3)]

ct=0
newim=np.zeros([width,height],dtype='object')
for y in range(0,height):
    #print("Row: %d") %y  # print row number
    for x in range(0,width):
        #print pix[x,y]  # print each pixel RGB tuple
        newim[x,y] = imagedata[ct]
        ct=ct+1

#image.show()        # create image
#image.save('C:\\Users\\Niramai Siva\\Downloads\\image.jpg')      






####### decryption
#
#
#imagedata1=[imagedata[i].to_bytes(3,byteorder='big') for i in range(0, int(relength))]
#dplaintext=imagedata1[0]
#for i in range(1,len(imagedata1)):
#    dplaintext=dplaintext + ((imagedata1[i]))
#dplain=(dplaintext)
#while (len(dplain) % 16 != 0):
#    dplain = dplain + imagedata1[0]
#obj2 = AES.new(password, AES.MODE_CBC, 'This is an IV456')
#decrypted = obj2.decrypt((dplain))              # save to a file
### parse the decrypted text back into integer string
#
#step = 3
#finaltextone=[decrypted[i:i+step] for i in range(0, len(decrypted), step)]
#
#
#while (int(relength) != len(finaltextone)):
#    finaltextone.pop()           
#for i in range(0, len(finaltextone)) :
#    if finaltextone[i].isdigit():
#        c=1
#    else:
#        finaltextone[i]=100;
#        
#
#finaltexttwo=[(int(finaltextone[int(i)])-100) for i in range(0, len(finaltextone)) ]  
#
## reconstruct image from list of pixel RGB tuples
#newim = Image.new("L", (int(width), int(height)))
#newim.putdata((finaltexttwo))
#newim.show()
##    

