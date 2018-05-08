# -*- coding:utf-8 -*-
import PIL.Image as Image
import os, sys
import multiprocessing

def _virtualize_image(save_path='test.jpg', image_paths = [],
        ms=20, mw = 299, resize_mw = 256):
    """
    save image virtualization file.
    Args:
        ms: output image will show ms*ms pictures.
            For example, if ms = 20, then there will be 400 pictures shown on the output image.
        mw: original size of input images.
        resized_mw: resized weight
    """
    if resize_mw == None:
        msize = mw * ms
    else:
        msize = resize_mw * ms
    toImage = Image.new('RGB', (msize, msize))
    for y in range(0,ms):
        for x in range(0,ms):
            fname = image_paths[ms*y+x]
            if resize_mw != None:
                fromImage = Image.open(fname).resize((resize_mw, resize_mw), Image.ANTIALIAS)
                toImage.paste(fromImage, (x*resize_mw, y*resize_mw))
            else:
                fromImage = Image.open(fname)
                toImage.paste(fromImage, (x*mw, y*mw))
    toImage.save(save_path)

def virtualize_image(inputs = 'video_clusters.txt',
        outputs = 'output',
        num_threads = 16):
    """
    multi processing function
    Args:
        inputs: format is <video cover path> <clusterid>
        outputs: directory that to put output images.
        num_threads: threads number
    """
    # load cluster information
    cluster_dict = {}
    with open(inputs, 'r') as f:
        for line in f:
            path = line.strip().split('\t')[0]
            cluster = line.strip().split('\t')[1]
            if cluster not in cluster_dict:
                cluster_dict[cluster] = [path]
            else:
                cluster_dict[cluster].append(path)
    image = iter(cluster_dict.items())

    # make dirs for output
    try: os.makedirs(outputs)
    except: pass

    # multi-process
    force_exit = False
    while force_exit == False:
        records = []; threads = num_threads
        while threads != 0:
            try:
                cluster_id, imgs = next(image)
            except StopIteration:
                force_exit = True
                break
#            print (cluster_id, len(imgs))
            if len(imgs) >= 400:
                ms = 20
            elif len(imgs) >= 100:
                ms = 10
            elif len(imgs) >= 25:
                ms = 5
            else:
                continue
            save_path = os.path.join(outputs, cluster_id + '.jpg')
            print ('save into: {}'.format(save_path))
            process = multiprocessing.Process(target = _virtualize_image, args = (save_path, imgs, ms))
            process.start()
            records.append(process)
            threads -= 1
        for process in records:
            process.join()

if __name__ == '__main__':
    virtualize_image(inputs='video_cluster.txt', outputs='output', num_threads = 16)
