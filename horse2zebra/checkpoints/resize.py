
import glob
from PIL import Image

mgs_path = glob.glob('./booknoiseTrain2/web/images/*.png')

for path in mgs_path:
    
    img1 = Image.open(path)
    img1 = img1.resize((2137,3107))

    print(path)
    
    img1.save("./booknoiseTrain2/web/images_resize/"+path.split('/')[-1])
