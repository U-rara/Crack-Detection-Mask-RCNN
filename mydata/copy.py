import os 
import shutil

for dir_name in os.listdir('./labelme_json'):
    pic_name = dir_name[:-5] + '.png'
    from_dir = './labelme_json/'+dir_name+'./label.png'
    to_dir = './cv2_mask/'+pic_name
    shutil.copyfile(from_dir, to_dir)
        
    print (from_dir)
    print (to_dir)
    
    
    