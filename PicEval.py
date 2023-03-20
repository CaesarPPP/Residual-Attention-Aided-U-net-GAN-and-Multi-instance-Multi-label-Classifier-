# -*- coding: utf-8 -*-
"""
This part is used to evaluate the super-resolution performance of GANs


"""


import tensorflow as tf

 # Read images from file.
gen_path = 'E:/1.jpg'
hr_path = 'E:/2.jpg'

gen_img_data = tf.gfile.FastGFile(gen_path, "rb").read()
hr_img_data = tf.gfile.FastGFile(hr_path, "rb").read()

gen_img = tf.image.decode_png(gen_img_data)
hr_img = tf.image.decode_png(hr_img_data)

# Compute PSNR over tf.uint8 Tensors.
psnr1 = tf.image.psnr(gen_img, hr_img, max_val=255.)
ssim1 = tf.image.ssim(gen_img, hr_img, max_val=255.)
print(psnr1)
print(ssim1)
with tf.Session() as sess:
     psnr = psnr1.eval()
     ssim = ssim1.eval()
     print(psnr)
     print(ssim)
print("_______________________________________")

 