# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from scipy.misc import imread
from keras.models import model_from_json
import numpy as np
from api import settings
import time
import os
from keras import backend as K


def index(request):
    return HttpResponse("Hello, world. You're at the mnist index.")

@csrf_exempt
def json(request):
    K.clear_session()
    if request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(myfile.name, myfile)
        arr = np.expand_dims(imread(os.path.join(settings.MEDIA_ROOT, filename)), axis=0).astype(float)
        arr = arr/np.max(arr)
        y=vanilla(arr)
        result = []
        for i in range(y[0].shape[0]):
            result.append({'name': i, 'value': int(y[0][i])})

        os.remove(os.path.join(settings.MEDIA_ROOT, filename))
        return JsonResponse(result, safe=False)
    return JsonResponse([{'name': 'Erreur', 'value': 0}], safe=False)


def vanilla(arr):
    json_file = open(os.path.join(settings.BASE_DIR, 'mnist', 'vanilla_model', 'model.json'), 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(settings.BASE_DIR, 'mnist', 'vanilla_model', 'weights.hdf5'))
    json_file.close()
    y = model.predict(arr)
    y = (y-np.min(y))/(np.max(y) - np.min(y))
    y *= 100/np.sum(y)
    return y
