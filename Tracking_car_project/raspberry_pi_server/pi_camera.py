# from picamera import PiCamera
# import picamera.array
# import time

# class PiC :
# 	def __init__(self) :
# 		self.camera = PiCamera()
# 		self.camera.resoution = (160, 160)
# 		self.img = np.zeros((160 , 160, 3), dtype=np.Int32)

# 	def get_pic(self) :
# 		self.camera.capture(self.img, 'bgr')

# 		return self.img

# from PIL import Image
# import select
# import v4l2capture
# import time

# class PiC :
# 	def __init__(self) :
# 		self.video - v4l2capture.Video_device('/dev/video0')
# 		self.size_x, self.size_y = video.set_format(160, 160)

# 		self.video.create_buffers(1)


# 	def get_img(self):
# 		self.video.start()

# 		self.video.queue_all_buffers()

# 		select.select((self.video, ), (), ())

# 		img = self.video.read()
# 		self.video.close()

# 		return img
	

import pygame, sys

from pygame.locals import *
import pygame.camera
import numpy as np

class PiC :
	def __init__(self) :
		self.w = 100
		self.h = 100

		pygame.init()
		pygame.camera.init()

		self.cam = pygame.camera.Camera("/dev/video0", (self.w, self.h))

	def get_img(self) :
		self.cam.start()

		img = self.cam.get_image()

		image = pygame.surfarray.array3d(img)


		self.cam.stop()

		return image










