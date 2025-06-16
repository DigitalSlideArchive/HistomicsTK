from parallelization_utilities import (ProgressHelper, compute_mask, compute_mask_novips, 
                                           get_region_polygons, get_region_dict, first_order, 
                                           tile_grid, tile_grid_w_mask,
                                           get_ancestor_tileids, get_trim_dict, plot_tasks,
                                           process_tiles,
                                           Mask,tilejob)
from parallelization_utilities import write_to_tiff_vips, write_to_tiff_zarr

import os
import argparse
import large_image

# from job_func_wo_mask_nostriplocal import tilejob
import time
import json
import math
import numpy
import scipy
import skimage
import concurrent.futures
import itertools
import matplotlib.pyplot as plt
from collections import deque
# from multiprocessing import Manager
# from multiprocessing import Array
# from multiprocessing import shared_memory

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from simple_triton.config import TensorflowConfig
from simple_triton.model import TritonModel

assert len(tf.config.list_physical_devices("GPU")) == 0

import matplotlib.patches as patches
import concurrent.futures
import pyvips
import time
from pathlib import Path
import matplotlib.patches as patches
import hashlib
import h5py 
import large_image
import tempfile
import numpy as np
from simple_triton.feature_extraction import inference, study
from simple_triton.tile_iterators import TiffPrefetch
from simple_triton.utils import analyze
import time

name = "uni"
model = TritonModel(name, "localhost:8001")
model.load(block=True, timeout=1.)


# # load tensorflow model - set maximum batch size
# model = TritonModel(name, "localhost:8001")
# model.load(config=basic.json())
# check if model is loaded
assert model.is_loaded()

# check if model is idle
assert model.is_idle()

model.get_config()


project_dir = '/data/abhi/al_project_110824/' #'/data2/abhi/dsa_offline/' #/data/abhi/
svs_dir = 'val'
image_list = os.listdir(project_dir+svs_dir)
dataset_id = svs_dir+'_'

tiff_dir = os.listdir(os.path.join(project_dir,f'{dataset_id}tiffs'))

print(len(image_list))
print(len(tiff_dir))

def crop_center(img, _h, _w, cropx, cropy):
    startx = _w // 2 - (cropx // 2)
    starty = _h // 2 - (cropy // 2)    
    return img[starty:starty + cropy, startx:startx + cropx, :]

def get_strip_region(idx, pos, tile, tile40, tilemask, data, bbox, bbox_local, magnification=40):
    t1 = time.time()

    offset = 0 # offset to bounding box on all 4 edges

    scale = dict(magnification=magnification)
    mask_scale_factor = int(scale['magnification']//5) # mask magnification = 5
    
    region=dict(
        left=bbox[0]-offset*8, 
        top=bbox[1]-offset*8,  
        width=int(bbox[2]-bbox[0])+offset*8*2, 
        height=int(bbox[3]-bbox[1])+offset*8*2 
    )


    t2 = time.time()
    # print('tile_slicing:', t2-t1)

    img = ts.getRegion(region=region, scale=scale, format=large_image.constants.TILE_FORMAT_NUMPY)
    img_c = img[0].copy()
    _h,_w,_c = img_c.shape

    if _c in (2, 4):
        img_c = img_c[:, :, :-1]

    t3 = time.time()
    # print('region superpixel read:', t3-t2)



    mask = data[int(bbox_local[1]):int(bbox_local[3]),int(bbox_local[0]):int(bbox_local[2]),:] 
    mask_img = mask

    if mask_img.shape[2] == 4:
        mask_img = mask_img[:, :, :-1]

    
    maskvals = [[val % 256, val // 256 % 256, val // 65536 % 256]
                for val in [idx * 2, idx * 2 + 1]]
    

    zeroed = ((mask_img != maskvals[0]).any(axis=-1) &
            (mask_img != maskvals[1]).any(axis=-1))
    
    zeroed = numpy.repeat(zeroed, mask_scale_factor, axis=0).repeat(mask_scale_factor, axis=1)


    img_c[zeroed] = [0, 0, 0]

    cropped_img = crop_center(img_c, _h, _w, 224, 224)
    cropped_img = cropped_img#.astype(numpy.float32)/255.0

    t6 = time.time()

    return numpy.expand_dims(cropped_img, axis=0), t3-t2, t6-t1


def get_tile_for_feature(idx, bbox, source_zarr, magnification=40):
    offset = 0 # offset to bounding box on all 4 edges

    scale = dict(magnification=magnification)
    mask_scale_factor = int(scale['magnification']//5) # mask magnification = 5
    
    region=dict(
        left=bbox[0]-offset*8, 
        top=bbox[1]-offset*8,  
        width=max(224,int(bbox[2]-bbox[0])+offset*8*2), 
        height=max(224,int(bbox[3]-bbox[1])+offset*8*2) 
    )

    mask_region=dict(
        left=bbox[0]//8-offset,
        top=bbox[1]//8-offset, 
        width=max(224//8,int(bbox[2]-bbox[0])//8+offset*2), 
        height=max(224//8,int(bbox[3]-bbox[1])//8+offset*2)
    ) 

    img = ts.getRegion(region=region, scale=scale, format=large_image.constants.TILE_FORMAT_NUMPY)
    img_c = img[0].copy()
    _h,_w,_c = img_c.shape

    if _c in (2, 4):
        img_c = img_c[:, :, :-1]

    mask = source_zarr.getRegion(region=mask_region, format=large_image.constants.TILE_FORMAT_NUMPY)    
    mask_img = mask[0]

    if mask_img.shape[2] == 4:
        mask_img = mask_img[:, :, :-1]

    maskvals = [[val % 256, val // 256 % 256, val // 65536 % 256]
                for val in [idx * 2, idx * 2 + 1]]

    #upscale mask to image scale
    zeroed = numpy.repeat(((mask_img != maskvals[0]).any(axis=-1) &
            (mask_img != maskvals[1]).any(axis=-1)), mask_scale_factor, axis=0).repeat(mask_scale_factor, axis=1)

    if zeroed.shape[0] < 224:
        zeroed = np.concatenate((zeroed, np.zeros((224-zeroed.shape[0],zeroed.shape[1]),dtype=zeroed.dtype)), axis=0)
    elif zeroed.shape[1] < 224:
        zeroed = np.concatenate((zeroed, np.zeros((zeroed.shape[0], 224-zeroed.shape[1]),dtype=zeroed.dtype)), axis=1)

    print('mask shape', mask_img.shape,
        'zeroed shape', zeroed.shape,
        'region shape:', 
        max(224,int(bbox[2]-bbox[0])+offset*8*2),
        max(224,int(bbox[3]-bbox[1])+offset*8*2), 
        )
    
    img_c[zeroed] = [0, 0, 0]

    cropped_img = crop_center(img_c, _h, _w, 224, 224)
    cropped_img = cropped_img#.astype(numpy.float32)/255.0
    print('image shapes:', img_c.shape, 
        cropped_img.shape)
    # sad
    return numpy.expand_dims(cropped_img, axis=0)

def generate_batch(source_zarr):
    tilelist = []
    read_kwargs = []
    tilelist.append((get_tile_for_feature(idx, elem['user']['bbox'][idx * 4: idx * 4 + 4], 
                                        source_zarr
                                    ), [{f'bbox': elem['user']['bbox']
                [idx * 4: idx * 4 + 4]}]
                    )) #for idx, _ in enumerate(elem['values'])]# if idx < 10]
    read_kwargs.append([{f'bbox': elem['user']['bbox']
                [idx * 4: idx * 4 + 4]}])
    
    return tilelist, read_kwargs


class tileiterator(object):
    def __init__(self,
        # tilelist,
        # read_kwargs,
        dtype=numpy.uint8,
    ):
        self.read_kwargs = []#read_kwargs
        # self.tilelist = tilelist
        self.batch = max_batch_size
        # self._initialize()
        self.queue = deque([]) 
        self.pos = 0

    # def _initialize(self,):
    #     self.queue = deque([]) 
    #     self.pos = 0  # position in read_kwargs
        # self._fill()

    def __iter__(self):
        return self

    def __next__(self):
        if (self.pos == len(self.read_kwargs)) and not len(self.queue):
            raise StopIteration

        try:
            tiles, read_kwargs = self.queue.pop()
            # self._fill()
            print(self.pos, ':', len(read_kwargs), read_kwargs[0], read_kwargs[-1])
        except:
            raise

        self.pos+=1

        # last batch may only have partial size
        if self.pos == len(self.read_kwargs) and not len(self.queue):
            tiles.shape = [len(read_kwargs), *tiles.shape[1:]]
        

        return tiles, read_kwargs

    def _fill(self, tilelist, read_kwargs):
        try:
            self.queue.append(tilelist)
            self.read_kwargs.append(read_kwargs) #+= read_kwargs
        except Exception as error:
            raise

class singletileiterator(object):
    def __init__(self,
        tilelist,
        read_kwargs,
        dtype=numpy.uint8,
    ):
        self.read_kwargs = read_kwargs
        self.batch = max_batch_size
        self.data = tilelist
        self.pos = 0


    def __iter__(self):
        return self

    def __next__(self):
        if (self.pos > 0):
            raise StopIteration

        try:
            tiles, read_kwargs = self.data
            # self._fill()
            # print(self.pos, ':', len(read_kwargs), read_kwargs[0], read_kwargs[-1])
        except:
            raise

        self.pos+=1
        

        return tiles, read_kwargs


for image in image_list:
    image = image.split('.')[0]
    if f'superpixel_vips_{image}.tiff' not in tiff_dir:
        try:
            opts = argparse.Namespace(
                inputImageFile= os.path.join(project_dir,svs_dir,f'{image}.svs'),
                outputImageFile=os.path.join(project_dir,f'{dataset_id}tiffs',f'superpixel_vips_{image}.tiff'),
                outputAnnotationFile=os.path.join(project_dir,f'{dataset_id}annotations',f'{image}.json'),
                inputMaskFile=os.path.join(project_dir,f'{dataset_id}masks',f'{image}.svs_1.25_mask.png'),
                roi=[-1, -1, -1, -1],
                tileSize=1000, #4096
                superpixelSize=100,
                magnification=5,
                overlap=True,
                boundaries=True,
                bounding='Internal',
                slic_zero=True,
                compactness=0.1,
                sigma=1,
                default_category_label='default',
                default_fillColor='rgba(0, 0, 0, 0)',
                default_strokeColor='rgba(0, 0, 0, 1)'
                )

            num_workers = 20
            max_batch_size = 128

            print(opts.inputImageFile, '\n',
                opts.outputImageFile, '\n', 
                opts.outputAnnotationFile, '\n', 
                opts.inputMaskFile)

            large_image.__version__

            print(os.getcwd())
            print(os.path.exists(opts.inputImageFile))
            ts = large_image.open(opts.inputImageFile)
            mask_file = opts.inputMaskFile
            # tracer = VizTracer()

            averageSize = opts.superpixelSize ** 2
            overlap = opts.superpixelSize * 4 * 2 if opts.overlap else 0
            tileSize = opts.tileSize + overlap

            print(f'(averageSize, overlap, tileSize): {(averageSize, overlap, tileSize)}')


            meta = ts.getMetadata()
            found = 0
            bboxes = []
            bboxesUser = []
            tiparams = {}
            tiparams = get_region_dict(opts.roi, None, ts)

            print(meta, '\n', get_region_dict(opts.roi, None, ts))


            mask = Mask(mask_file, ts)


            start = time.time()
            task_ids, tasks, h, w, grid, scale, coordx, coordy, alltile_metadata = tile_grid_w_mask(ts, mask, opts, averageSize, overlap, tileSize, tiparams,verbose=False)
            # task_ids, tasks, h, w, grid, coordx, coordy = tile_grid(ts, opts, averageSize, overlap, tileSize, tiparams,verbose=True)
            stop = time.time()

            ancestors, dependents = first_order(grid)

            trim_dict = get_trim_dict(dependents)

            ancestor_taskids = get_ancestor_tileids(ancestors, coordx, coordy)


            xs = set() 
            ys = set()
            for _x,_y in alltile_metadata.keys():
                xs.add(_x)
                ys.add(_y)
            last_x = max(xs)
            last_y = max(ys)

            wsi_w = last_x + alltile_metadata[(last_x,0)][1]
            wsi_h = last_y + alltile_metadata[(0,last_y)][0]


            tile_mask = mask


                
            if __name__ == '__main__':


                start = time.time()
                # _ , tasks, _, _, _, _, _ = tile_grid_w_mask(ts, mask, opts, averageSize, overlap, tileSize, tiparams,verbose=True)
                #tile_grid(ts, opts, averageSize, overlap, tileSize, tiparams)

                
                results = {}
                strips = {}
                strips_found = {}
                bboxes_dict = {}
                bboxesUser_dict = {}
                img = pyvips.Image.black(
                    tiparams.get('region', {}).get('width', meta['sizeX']) / scale,
                    tiparams.get('region', {}).get('height', meta['sizeY']) / scale,
                    bands=4)
                img = img.copy(interpretation=pyvips.Interpretation.RGB)
                found = 0

                job_submission_times = []
                vip_times = []
                accessmetadata_times = []
                send_to_triton = []
                shift_id = 0
                submitted_region = {}

                if True: #opts.outputImageFile not in tiff_dir: TO BE IMPLEMENTED. NEED JSON FOR BBOX.

                    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
                    
                    # submit all tasks with no ancestors 

                        submitted = {
                            pool.submit(tilejob, task, tasks[(coordx[task[0]], coordy[task[1]])], trim_dict, coordx, coordy, 
                                        numpy.array(tasks[(coordx[task[0]], coordy[task[1]])][4]), opts): 
                                        task for task in [t for t in grid if not len(ancestors[t])]
                        }
                        
                        # as tasks complete, update dependencies and submit new tasks
                        for i in range(len(grid)):

                            
                            completed, waiting = concurrent.futures.wait(       
                                submitted, return_when=concurrent.futures.FIRST_COMPLETED
                            )

                            print(len(completed))

                            for future in completed:
                                start1 = time.time()

                                results[submitted[future]] = future.result() 

                                strips[results[submitted[future]][0]] = results[submitted[future]][1]
                                strips_found[results[submitted[future]][0]] = results[submitted[future]][2]
                                bboxes_dict[results[submitted[future]][0]] = results[submitted[future]][3]
                                bboxesUser_dict[results[submitted[future]][0]] = results[submitted[future]][4]

                                for dependent in dependents[submitted[future]]:
                                    try:
                                        ancestors[dependent].remove(submitted[future])
                                    except Exception as e:
                                        print(e)
                                        print(dependent,'\n', submitted[future])
                                        raise Exception

                                    if not len(ancestors[dependent]):
                                        mask = compute_mask_novips(dependent, tasks[(coordx[dependent[0]], coordy[dependent[1]])], overlap, strips, coordx, coordy, ancestor_taskids)

                                        submitted[pool.submit(tilejob, dependent, tasks[(coordx[dependent[0]], coordy[dependent[1]])], trim_dict, coordx, coordy, mask, opts)] = dependent

                                stop = time.time()
                                
                                job_submission_times.append(stop-start1)
                                
                                # write tile to vips    
                                data = results[submitted[future]][1][1]

                                d1 = numpy.where(data[:,:,0]>=0, data[:,:,0]+found, data[:,:,0]).astype(int)

                                data = numpy.dstack((
                                    ((d1) % 256).astype(int),
                                    ((d1) / 256).astype(int) % 256,
                                    ((d1) / 65536).astype(int) % 256,
                                    data[:,:,1])).astype('B')

                                vimg = pyvips.Image.new_from_memory(
                                        numpy.ascontiguousarray(data).data,
                                        data.shape[1], data.shape[0], data.shape[2],
                                        large_image.constants.dtypeToGValue[data.dtype.char])

                                vimg = vimg.copy(interpretation=pyvips.Interpretation.RGB)
                                vimgTemp = pyvips.Image.new_temp_file('%s.v')
                                vimg.write(vimgTemp)
                                vimg = vimgTemp

                                img = img.composite([vimg], pyvips.BlendMode.OVER, x=results[submitted[future]][0][0], y=int(results[submitted[future]][0][0]))

                                found += strips_found[results[submitted[future]][0]]

                                stop2 = time.time()
                                vip_times.append(stop2-stop)
                                # Send to triton

                                task_id = (coordx[submitted[future][0]], coordy[submitted[future][1]])
                                (tx0, ty0) = task_id
                                (x0, y0, scale, tile, tilemask) = tasks[(coordx[submitted[future][0]], coordy[submitted[future][1]])]


                                bboxuser = results[submitted[future]][4]


                                stop3 = time.time()
                                accessmetadata_times.append(stop3-stop2)
                                
                                up_times = 0
                                read_times = 0
                                total_times = 0
                                
                                # submitted_region = {}
                                # for idx in range(0,len(bboxuser)//4):
                                #     bbox = bboxuser[4*idx:4*idx+4]
                                #     bbox_local = ((bbox[0]-x0)//scale - tx0,
                                #                   (bbox[1]-y0)//scale - ty0,
                                #                   (bbox[2]-x0)//scale - tx0,
                                #                   (bbox[3]-y0)//scale - ty0)

                                #     up_start = time.time()
                                #     # _ , read_time, total_time = get_strip_region(shift_id + idx, (tx0,ty0), tile['tile'], None, tilemask, data, bbox, bbox_local, magnification=40)


                                #     submitted_region[
                                #         pool.submit(get_strip_region, 
                                #                     shift_id + idx, (tx0,ty0), 
                                #                     tile['tile'], None, tilemask, data, 
                                #                     bbox, bbox_local, magnification=40)
                                #                     ] = (shift_id + idx, bbox)
                                    



                                #     up_stop = time.time()

                                #     up_times += up_stop-up_start
                                #     # read_times+=read_time#t3-t2
                                #     # total_times+=total_time#t6-t1

                                #     if idx == len(bboxuser)//4-1:
                                #         shift_id += len(bboxuser)//4


                                # for future_region in concurrent.futures.as_completed(submitted_region):
                                #     bbox_id, bbox_info = submitted_region[future_region]
                                #     try:
                                #         data, read_time, total_time = future_region.result()
                                #     except Exception as exc:
                                #         print('%r generated an exception: %s' % (bbox_id, bbox_info))
                                #     else:
                                #         print('%r cropped image shape is %s' % (bbox_id, data.shape))
                                    
                                #     read_times+=read_time#t3-t2
                                #     total_times+=total_time#t6-t1


                                # stop4 = time.time()
                                # print(f'found {strips_found[results[submitted[future]][0]]}; send {len(bboxuser)//4} to triton:', stop4-stop3, 'read times:', read_times, 'total times:', total_times, 'up times:', up_times)
                                
                                # send_to_triton.append(stop4-stop3)
                                send_to_triton.append(0)

                                del submitted[future]

                    stop = time.time()
                                
                    print(f'time elapsed {stop - start}')


                    print(sum(job_submission_times),
                        sum(vip_times),
                        sum(accessmetadata_times),
                        sum(send_to_triton)/found*2)

                    scale = 8
                    if opts.magnification:
                        tiparams['scale'] = {'magnification': opts.magnification}

                    bboxes = []
                    bboxesUser = []

                    for cx, cy in grid:        
                        stripidx = (coordx[cx], cy)

                        bboxes += bboxes_dict[stripidx]
                        bboxesUser += bboxesUser_dict[stripidx]



                    # opts.outputImageFile=os.path.join('/data/abhi/superpixel_tiff2',f'superpixel_vips_{image}.tiff')
                    opts.outputImageFile=os.path.join(f'/data/abhi/al_project_110824/{dataset_id}tiffs',f'superpixel_vips_{image}.tiff')
                    found = write_to_tiff_vips(opts, grid, strips, strips_found, meta, scale, tiparams, coordx)
                    print(found)

                else:
                    print("skipping superpixel generation. Already found.")

                source_zarr = large_image.open(opts.outputImageFile)#os.path.join('/data/abhi/superpixel_tiff2',f'superpixel_vips_{image}.tiff'))
                # view metadata
                print(source_zarr.getMetadata())


                if opts.outputAnnotationFile:
                    print('>> Generating annotation file')
                    categories = [
                        {
                            'label': opts.default_category_label,
                            'fillColor': opts.default_fillColor,
                            'strokeColor': opts.default_strokeColor,
                        },
                    ]
                    annotation_name = 'Superpixel Epoch 0' #os.path.splitext(os.path.basename(opts.outputAnnotationFile))[0]
                    region_dict = get_region_dict(opts.roi, None, ts)
                    annotation = {
                        'name': annotation_name,
                        'elements': [{
                            'type': 'pixelmap',
                            'girderId': 'outputImageFile',
                            'transform': {
                                'xoffset': region_dict.get('region', {}).get('left', 0) / scale,
                                'yoffset': region_dict.get('region', {}).get('top', 0) / scale,
                                'matrix': [[scale, 0], [0, scale]],
                            },
                            'values': [0] * (found // (2 if opts.boundaries else 1)),
                            'categories': categories,
                            'boundaries': opts.boundaries,
                        }],
                        'attributes': {
                            'params': {k: v for k, v in vars(opts).items() if not callable(v)},
                            'cli': None,#Path(__file__).stem,
                            'version': None,#histomicstk.__version__,
                        },
                    }
                    if len(bboxes) and str(opts.bounding).lower() != 'separate':
                        annotation['elements'][0]['user'] = {'bbox': bboxesUser}
                    if len(bboxes) and str(opts.bounding).lower() != 'internal':
                        bboxannotation = {
                            'name': '%s bounding boxes' % os.path.splitext(
                                os.path.basename(opts.outputAnnotationFile))[0],
                            'elements': [{
                                'type': 'rectangle',
                                'center': [bcx, bcy, 0],
                                'width': bw,
                                'height': bh,
                                'rotation': 0,
                                'label': {'value': 'Region %d' % bidx},
                                'fillColor': 'rgba(0,0,0,0)',
                                'lineColor': opts.default_strokeColor,
                            } for bidx, (bcx, bcy, bw, bh) in enumerate(bboxes)],
                            'attributes': {
                                'params': {k: v for k, v in vars(opts).items() if not callable(v)},
                                'cli': None,#Path(__file__).stem,
                                'version': None,#histomicstk.__version__,
                            },
                        }
                        annotation = [annotation, bboxannotation]

                        opts.outputAnnotationFile = opts.outputAnnotationFile.split('.')[0]+'vips.'+opts.outputAnnotationFile.split('.')[1]
                    with open(opts.outputAnnotationFile, 'w') as annotation_file:
                        try:
                            json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)
                        except Exception:
                            print('Failed to serialize annotation')
                            print(repr(annotation))
                            raise
                    if hasattr(opts, 'callback'):
                        opts.callback('file', 2, 2)


                elem = annotation['elements'][0]

                bbox = elem['user']['bbox']
                hashval = repr(dict(
                    itemId=f'{annotation_name}', bbox=[int(v) for v in bbox], patchSize=224))
                hashval = hashlib.new('sha256', hashval.encode()).hexdigest()
                feature_fileName = 'feature-%s.h5' % (hashval)
                #'feature-0d77a56f18be4bbfa02da996f5cf68874626cd8b4d404866ee6fac186777c08d.h5'
                #'feature-c1b7307e5087d15b337e28f3ef04c52b54d00d5057b6f2454c2a30cd37c0ca0d.h5' #'feature-%s.h5' % (hashval)

                for idx, _ in enumerate(elem['values']):

                    bbox = elem['user']['bbox'][idx * 4: idx * 4 + 4]
                    # if bbox[0] > 87000 and bbox[2] < 101000 and bbox[1]>64000 and bbox[1]<69000:
                    img = get_tile_for_feature(idx, bbox, source_zarr)

                    # img.dtype()
                    if idx == 1:
                        print(bbox)
                        break

                print(feature_fileName)

                os.chdir('/home/atc5426/superpixel-classification/')

                limit = 1  # limit on number of pending requests per worker
                verbose = True  # display inference statistics and debugging information

                tilelist = [[],[]]
                read_kwargs = []
                config = model.get_config()
                config["input"][0]["dataType"] = None
                # create tile iterator
                dtype = np.float32 if config["input"][0]["dataType"] == "TYPE_FP32" else np.uint8

                print('Create feature', feature_fileName)
                lastlog = starttime = time.time()
                patchSize = 224
                ds = None

                features_dir = os.path.join('/data/abhi/al_project_110824',f'{dataset_id}features',image)
                os.makedirs(features_dir,exist_ok=True)

                filePath = os.path.join(features_dir, feature_fileName)

                with h5py.File(filePath, 'w') as fptr:

                    for idx, _ in enumerate(elem['values']):


                        tilelist[0].append(get_tile_for_feature(idx, elem['user']['bbox'][idx * 4: idx * 4 + 4], source_zarr)) 
                        
                        tilelist[1].append({'id':idx, f'bbox': elem['user']['bbox'][idx * 4: idx * 4 + 4]}) 
                                                                                #for idx, _ in enumerate(elem['values'])]# if idx < 10]

                        read_kwargs.append({'id':idx, f'bbox': elem['user']['bbox'][idx * 4: idx * 4 + 4]}) 
                                                                                #for idx, _ in enumerate(elem['values'])]# if idx < 10]


                        if (idx+1)%max_batch_size == 0 or (idx+1) == len(elem['values']):

                            print('processing:', idx+1, end='; ')

                            print(tilelist[0][0].shape, tilelist[0][1].shape, tilelist[0][2].shape)

                            tilelist[0] = np.concatenate(tilelist[0])

                            singleiterator = singletileiterator(tilelist, read_kwargs)

                            # inference

                            features, metadata, times, failures = inference(
                            singleiterator, name, url="localhost:8001", limit=limit, rest=0.0
                            )
                            tilelist = [[],[]]
                            read_kwargs = []

                        
                            # TODO: ensure this is uint8
                            if not ds:
                                # print(features[0][0].shape,':',features[0][0].shape)

                                ds = fptr.create_dataset(
                                    'images', features[0][0].shape, maxshape=(None,features[0][0].shape[1]),
                                    dtype=features[0][0].dtype, chunks=True)
                            else:

                                # print(features[0][0].shape,':',ds.shape[0] ,':',(ds.shape[0] + features[0][0].shape[0],features[0][0].shape[1]))

                                ds.resize((ds.shape[0] + features[0][0].shape[0], features[0][0].shape[1]))

                            # print(ds.shape, ':', features[0][0].shape)   
                            ds[ds.shape[0] - features[0][0].shape[0]:,:] = features[0][0]
                            if time.time() - lastlog > 5:
                                lastlog = time.time()
                                print(ds.shape, len(elem['values']),
                                        '%5.3f' % (time.time() - starttime),
                                        '%5.3f' % ((len(elem['values']) - idx - 1) / (idx + 1) *
                                                    (time.time() - starttime)),
                                        image)
                            # print(ds.shape, len(elem['values']), '%5.3f' % (time.time() - starttime),
                            #         image)
                # feature-c1b7307e5087d15b337e28f3ef04c52b54d00d5057b6f2454c2a30cd37c0ca0d.h5

                fptr.close()
        except Exception as e:
            print(f"skipping {image}. {e}" )
            with open("skipped_files.txt", "a") as f:
                f.write(image+"\n")

    else:
        print(f"skipping {image}. Already rendered.")
        
    print('completed extraction')