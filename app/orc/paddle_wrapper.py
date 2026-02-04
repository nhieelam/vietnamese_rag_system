from typing import Union, List, Tuple, Optional
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import io

class PaddleOCRWrapper:
    def __init__(self, lang='vi', use_angle_cls=True, use_gpu=False, **kwargs):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=use_gpu, **kwargs)

    def read_image(self, image, cls=True):
        return self._ocr.predict(image, cls=cls)