# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:19:18 2019

@author: tageldim
"""
import girder_client


def connect_to_api(apiurl, apikey=None, interactive=True):
    """Connect to a specific girder API"""
    assert interactive or (apikey is not None)
    gc = girder_client.GirderClient(apiUrl=apiurl)
    if apikey is not None:
        interactive = False
    if interactive:
        gc.authenticate(interactive=True)
    else:
        gc.authenticate(apiKey=apikey)
    return gc

