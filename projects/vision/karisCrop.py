from PIL import Image

im = Image.open('karis1.ppm')

width = im.width
height = im.height

im_crop = im.crop((width//2, 0, width, int(height*.4)))
im_crop.save('karis1_x.png', quality=95)

