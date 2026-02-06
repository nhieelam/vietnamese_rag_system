from paddleocr import PaddleOCR

class PaddleOCRWrapper:
    def __init__(self, lang='vi', use_angle_cls=True, **kwargs):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang,  **kwargs)
    
    def read_image(self, image):
        return self._ocr.predict(image)
    