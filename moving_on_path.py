import numpy as np


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


import matplotlib.image as mpimg
import matplotlib.animation as animation
from scipy import ndimage
import cv2
import io

class render:
  def __init__(self,path,position):
    src= "https://raw.githubusercontent.com/vahidseydi/ReinforcementLearning/master/mc.png"
    self.Object= mpimg.imread(src)
    self.path=path
    self.position=position

  def set_object_list(self):
    im=self.Object
    get_angle = lambda slope:np.rad2deg(np.arctan(slope))
    normal =lambda im:(im * 255).astype(np.uint8)
    self.obj_list=[ normal(ndimage.rotate(im, get_angle(v),cval=1)) for v in self.position.df]
    
  def set_image_list(self):
    ims=[]   
    for i in range(len(self.position.x)):
      fig= plt.figure()
      ax = fig.gca()
      #ax.axes(xlim=(-1, 1), ylim=(-1, 1))
      ax.plot(self.path.x,self.path.f,'-b')
      imagebox = OffsetImage(self.obj_list[i], zoom=0.1)
      
      ax.axis('off')
      ab = AnnotationBbox(imagebox, (self.position.x[i], self.position.f[i]),frameon=False)
      ax.add_artist(ab)

      buf = io.BytesIO()
      fig.savefig(buf, format="png", dpi=180)
      
      buf.seek(0)
      img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
      buf.close()
      img = cv2.imdecode(img_arr, 1)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
      ims.append(img)
      plt.close()
    self.ims=ims

  def init(self):
    self.img.set_data(self.ims[0])
    return (self.img,)

  def animate(self,i):    
    self.img.set_data(self.ims[i])
    return (self.img,)
    
  def show(self):
    self.set_object_list()
    self.set_image_list()
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    self.img = ax.imshow(self.ims[0])
    anim = animation.FuncAnimation(fig, self.animate, init_func=self.init,
                                 frames=len(self.ims), interval=40, blit=True)
    return anim
class path:
  def __init__(self,x,grad=False):
    f= lambda x:np.sin(3 * x)*.45+.55
    df = lambda x:0.45* 3*np.cos(3*x)
    self.x = x
    self.f = f(x)
    if grad:
      self.df = df(x)
