# artwork-generator
generate artworks with neural style transfer

### demo

![]('misc/demo.png')

### install
pip install .

### use
```
from AGen import *
import matplot.pyplot as plt
try:
	%matplotlib notebook
except:
	pass
res = VGG19Gen(style_file,content_file,size,denoiser)
plt.imshow(res)
```