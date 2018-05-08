# -*- coding: utf-8 -*-
import os,io,shutil
import urllib.request
import PIL.Image as Image
import multiprocessing
import time

# fetch videoid-videourl from video.pictures.
def fetch_file(subdir):
    cur_path = os.path.abspath(os.curdir)
    list_of_url_files = os.listdir(os.path.join(cur_path,subdir))
    return len(list_of_url_files), iter([os.path.join(cur_path, subdir, i) for i in list_of_url_files ])

def _save_images(inputs, outputs):
    """ save images from image-url files
    Args:
        inputs: a directory that has <imagenames, imageurls>
            |--inputs
               |--inputs/image_file.001
               |--inputs/image_file.002
               |-- ...
        outputs: a directory that save images.
            |--outpus
               |--outputs/imagename1.jpg
               |--outputs/imagename2.jpg
               |--outputs/...
    """
    height, width = 299, 299
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    with open(inputs, 'r') as f:
        for line in f:
            line = line.strip().split()
            image_url, image_name = line[1], line[0] + '.jpg'
            _outputs = '{}{}{}'.format(outputs, os.sep, image_name)

            if image_url.startswith('http://'):
                with urllib.request.urlopen(image_url) as url:
                    image_bytes = url.read()
                    data_stream = io.BytesIO(image_bytes)
                    Image.open(data_stream).convert('RGB').resize((height, width), Image.ANTIALIAS).save(_outputs)
            elif image_url.endswith('.jpg'):
                Image.open(image_url).resize((height, width), Image.ANTIALIAS).save(_outputs)
            else:
                continue

def save_images(inputs = "/url/data/", outputs = 'output', num_threads = 12):
    input_size, file_it = fetch_file(inputs)
    num_threads = num_threads if num_threads < input_size else input_size
    start_time = time.time()
    # multi-process
    force_exit = False
    while force_exit == False:
        records = []; threads = num_threads
        while threads != 0:
            try:
                file_in = next(file_it)
            except StopIteration:
                force_exit = True
                break
            # when input is XXX.000, then output will be outputs/000/
            dirname = file_in.split('.')[-1]
            file_out = os.path.join(os.curdir, outputs, dirname)
            print ('input: {}, output: {}'.format(file_in, file_out))
            process = multiprocessing.Process(target = _save_images, args = (file_in, file_out))
            process.start()
            records.append(process)
            threads -= 1
        for process in records:
            process.join()
    end_time = time.time()
    print("save images\ntime consuming: %0.2f"%(end_time - start_time))

def _tar_files(inputs, outputs):
    print ("zip from %s to %s.zip" % (inputs, outputs))
    shutil.make_archive(outputs, 'zip', inputs)

def tar_files(num_threads = 12, inputs = 'output', outputs = 'zip'):
    input_size, file_it = fetch_file(inputs)
    num_threads = num_threads if num_threads < input_size else input_size
    start_time = time.time()
    # multi-process
    force_exit = False
    while force_exit == False:
        records = []; threads = num_threads
        while threads != 0:
            try:
                file_in = next(file_it)
                file_out = os.path.join(outputs, file_in.split('/')[-1])
            except StopIteration:
                force_exit = True
                break
            print ('input: {}, output: {}'.format(file_in, file_out))
            process = multiprocessing.Process(target = _tar_files, args = (file_in, file_out))
            process.start()
            records.append(process)
            threads -= 1
        for process in records:
            process.join()
    end_time = time.time()
    print("tar files\ntime consuming: %0.2f"%(end_time - start_time))

