# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:30:06 2019

@author: tageldim
"""

from io import BytesIO
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import numpy as np

#%% ===========================================================================

def get_image_from_htk_response(resp):
    """Given a histomicsTK girder response, get np array image.
    Arguments:
        * resp - response from server request
    Returns:
        a pillow Image object
        
    Example:
        gc= girder_client.GirderClient(apiUrl = APIURL)
        gc.authenticate(interactive=True)
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" % (
          slide_id, left, right, top, bottom)
        resp  = gc.get(getStr, jsonResp=False)
        rgb = get_image_from_htk_response(resp)
    """        
    image_content = BytesIO(resp.content)
    image_content.seek(0)
    Image.open(image_content)
    image = Image.open(image_content)
    return np.uint8(image)

#%% ===========================================================================
    
