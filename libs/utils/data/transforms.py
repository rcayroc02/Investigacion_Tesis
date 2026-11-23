from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math
import numpy as np

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class StripeDirectedErasing(object):
    """
    Oclusión sintética dirigida por franjas multigranulares solapadas.

    Opera sobre tensores normalizados (C, H, W), igual que RandomErasing,
    y reemplaza una región alineada con la geometría de franjas del modelo
    con ruido aleatorio del mismo tipo que usa RandomErasing.

    Parámetros
    ----------
    probability   : prob. de aplicar la aumentación (default 0.5, igual que RE)
    img_h, img_w  : resolución de entrada tras Resize+Crop (384, 128)
    granularities : splits de las dos ramas, debe coincidir con cfg.MODEL.GRANULARITIES
    overlap_ratio : solapamiento entre franjas (debe coincidir con OverlappedPartPool2d)
    severity_low  : fracción mínima del área de la región objetivo a ocluir (0.20)
    severity_high : fracción máxima del área de la región objetivo a ocluir (0.40)
    mean          : valor de relleno (igual que RandomErasing, se usa ruido ≈ mean)

    Regiones candidatas (5 opciones del documento, §4.2):
        0 – franja superior  del 3-split
        1 – franja media     del 3-split
        2 – franja inferior  del 3-split
        3 – zona de transición entre franja 0 y 1 (solapamiento)
        4 – zona de transición entre franja 1 y 2 (solapamiento)
    """

    def __init__(self,
                 probability=0.5,
                 img_h=384, img_w=128,
                 granularities=(2, 3),
                 overlap_ratio=0.2,
                 severity_low=0.20,
                 severity_high=0.40,
                 mean=(0.3525, 0.3106, 0.3140)):   # LUPerson stats
        self.probability    = probability
        self.img_h          = img_h
        self.img_w          = img_w
        self.granularities  = granularities
        self.overlap_ratio  = overlap_ratio
        self.severity_low   = severity_low
        self.severity_high  = severity_high
        self.mean           = mean

        # Pre-calcular coordenadas (y0, y1) de cada región candidata
        self._regions = self._build_regions()

    # ------------------------------------------------------------------
    def _build_regions(self):
        """
        Calcula (y_start, y_end) en píxeles para cada región candidata
        usando el 3-split (granularities[-1]) con overlap_ratio.
        Espejo exacto de la lógica de OverlappedPartPool2d.
        """
        H     = self.img_h
        g     = self.granularities[-1]   # 3-split es el de mayor granularidad
        sh    = H // g                   # stripe_h base
        ov    = int(self.overlap_ratio * sh)

        regions = []

        # Franjas completas con solapamiento (0, 1, 2)
        for i in range(g):
            y0 = max(0, i * sh - ov)
            y1 = min(H, (i + 1) * sh + ov)
            regions.append((y0, y1))

        # Zonas de transición puras (solapamiento entre franja i e i+1)
        for i in range(g - 1):
            # La zona compartida es el overlap que "sale" de franja i hacia i+1
            # y el que "entra" de franja i+1 hacia i — se toman juntos
            y0 = max(0, (i + 1) * sh - ov)
            y1 = min(H, (i + 1) * sh + ov)
            regions.append((y0, y1))

        return regions   # 5 regiones: [franja0, franja1, franja2, trans01, trans12]

    # ------------------------------------------------------------------
    def __call__(self, img):
        """
        img : Tensor (C, H, W), ya normalizado.
        """
        if random.uniform(0, 1) >= self.probability:
            return img

        C, H, W = img.size()

        # Seleccionar región objetivo uniformemente entre las 5 opciones
        y0_reg, y1_reg = random.choice(self._regions)
        region_h = y1_reg - y0_reg
        region_area = region_h * W

        # Severidad: fracción del área de la región objetivo
        severity = random.uniform(self.severity_low, self.severity_high)
        target_area = severity * region_area

        # Generar máscara rectangular dentro de la región
        # Aspect ratio libre pero acotado para evitar tiras degeneradas
        for _ in range(50):
            aspect = random.uniform(0.3, 3.0)
            mask_h = int(round(math.sqrt(target_area * aspect)))
            mask_w = int(round(math.sqrt(target_area / aspect)))

            if mask_h < region_h and mask_w < W:
                x0 = random.randint(y0_reg, y0_reg + region_h - mask_h)
                y0 = random.randint(0, W - mask_w)

                # Ruido aleatorio centrado en la media del canal (mismo tipo que RE)
                if C == 3:
                    img[0, x0:x0+mask_h, y0:y0+mask_w] = self.mean[0]
                    img[1, x0:x0+mask_h, y0:y0+mask_w] = self.mean[1]
                    img[2, x0:x0+mask_h, y0:y0+mask_w] = self.mean[2]
                else:
                    img[0, x0:x0+mask_h, y0:y0+mask_w] = self.mean[0]
                return img

        return img   # fallback: no erasing si no encuentra máscara válida
