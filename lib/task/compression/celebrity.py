import face_recognition
import numpy as np
import os
from joblib import Parallel, delayed
# dataset http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


def load_and_get_emb(im_path):
    img = face_recognition.load_image_file(im_path)
    try:
        return [im_path,face_recognition.face_encodings(img)[0]]
    except:
        return None


def get_face_embeding_matrix(imgs_path, num_cores = 10,num_face = 1000):

    imgs_names = os.listdir(imgs_path)[:num_face]

    matrix = Parallel(n_jobs = num_cores)(delayed(load_and_get_emb)(imgs_path + name) for name in imgs_names)

    return np.array(matrix)
