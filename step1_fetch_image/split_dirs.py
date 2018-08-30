import os
import shutil
root_dir='/home/jennifer/ImageFeature/image_feature_pipeline/step1_fetch_image/start_end'
dirs = os.listdir(root_dir)

for d in dirs:
    print (d)
    if d.startswith('0') and d != '000' and d!= '001' and d!= '028' and d != '003':
        try: os.mkdir(d+'0')
        except: pass
        pictures = os.listdir(d)
        for i in pictures[15000:]:
            #print (os.path.join(root_dir,d,i))
            #print (os.path.join(root_dir,d+'0',i))
            shutil.move(os.path.join(root_dir,d,i), os.path.join(root_dir,d+'0',i))
