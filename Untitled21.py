#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.io import fits


# In[31]:


dat=fits.open("C:/Users/ANANDHU/Downloads/modtest_image.fits")[0].data


# In[32]:


plt.figure()
plt.imshow(dat)
plt.colorbar()
plt.plot(dat)
plt.show()


# In[33]:


plt.plot(dat)
plt.show()


# In[34]:


x = np.arange(-1, 1, 2/dat.shape[1])
y = np.arange(-1, 1, 2/dat.shape[0])
xx, yy = np.meshgrid(x, y)


# In[35]:


fit = fitting.LevMarLSQFitter()


# In[36]:


mymod = models.Sersic2D()


# In[37]:


p = fit(mymod,xx,yy,dat)
p


# In[38]:


plt.imshow(p(xx,yy))


# In[39]:


plt.plot(p(xx,yy))
plt.show()


# In[40]:


print(p)


# In[41]:


mymod = models.Gaussian2D()


# In[42]:


q = fit(mymod,xx,yy,dat)
q


# In[43]:


plt.imshow(q(xx,yy))


# In[44]:


plt.plot(q(xx,yy))
plt.show()


# In[45]:


print(q)


# In[46]:


print(p)


# In[ ]:




