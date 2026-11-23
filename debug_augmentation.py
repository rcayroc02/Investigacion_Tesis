"""
Sanity-check visual para StripeDirectedErasing.

Uso:
    python debug_augmentation.py --img_dir <ruta_con_imagenes_jpg>

Genera 'aug_sanity_check.png' con 3 columnas:
    col 0 – imagen original
    col 1 – Config 2 (RE intensificado)
    col 2 – Config 3 (StripeDirectedErasing)

Permite verificar visualmente que las oclusiones de la Config 3
caen sobre las franjas esperadas antes de lanzar ningún entrenamiento.
"""

import sys, os, argparse, random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms as TV

sys.path.insert(0, os.path.dirname(__file__))
from libs.utils.data import transforms as T

# -----------------------------------------------------------------------
# Parámetros fijos (deben coincidir con default.py)
# -----------------------------------------------------------------------
H, W = 384, 128
PIXEL_MEAN = [0.3525, 0.3106, 0.3140]
PIXEL_STD  = [0.2660, 0.2522, 0.2505]
GRANULARITIES = (2, 3)
OVERLAP_RATIO = 0.2
N_SAMPLES = 8          # imágenes a visualizar

# -----------------------------------------------------------------------
def denorm(t):
    """Tensor (C,H,W) normalizado → numpy (H,W,3) uint8."""
    mean = torch.tensor(PIXEL_MEAN).view(3,1,1)
    std  = torch.tensor(PIXEL_STD ).view(3,1,1)
    img  = t * std + mean
    img  = img.permute(1,2,0).numpy()
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def build_transforms(aug_type):
    normalizer = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    base = [
        T.Resize((H, W), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((H, W)),
        T.ToTensor(),
        normalizer,
    ]
    if aug_type == 'baseline':
        aug = T.RandomErasing(probability=1.0, mean=PIXEL_MEAN)
    elif aug_type == 're_intensified':
        aug = T.RandomErasing(probability=1.0, sl=0.02, sh=0.5, r1=0.3, mean=PIXEL_MEAN)
    else:  # stripe_directed
        aug = T.StripeDirectedErasing(
            probability=1.0,
            img_h=H, img_w=W,
            granularities=GRANULARITIES,
            overlap_ratio=OVERLAP_RATIO,
            severity_low=0.20, severity_high=0.40,
            mean=tuple(PIXEL_MEAN),
        )
    return T.Compose(base + [aug])

def stripe_coords():
    """Devuelve (y0, y1) de las 5 regiones para superponer en la figura."""
    g  = GRANULARITIES[-1]
    sh = H // g
    ov = int(OVERLAP_RATIO * sh)
    regions = []
    for i in range(g):
        regions.append((max(0, i*sh - ov), min(H, (i+1)*sh + ov)))
    for i in range(g - 1):
        regions.append((max(0, (i+1)*sh - ov), min(H, (i+1)*sh + ov)))
    return regions

# -----------------------------------------------------------------------
def main(img_dir):
    paths = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ][:N_SAMPLES]

    if not paths:
        print('No se encontraron imágenes en', img_dir)
        sys.exit(1)

    tf_orig  = T.Compose([T.Resize((H, W), interpolation=3), T.ToTensor()])
    tf_re    = build_transforms('re_intensified')
    tf_strip = build_transforms('stripe_directed')

    regions = stripe_coords()
    region_labels = ['Franja 0', 'Franja 1', 'Franja 2', 'Trans 0-1', 'Trans 1-2']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']

    n = len(paths)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.5))
    if n == 1:
        axes = [axes]

    col_titles = ['Original', 'Config 2: RE intensificado', 'Config 3: Stripe Directed']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9, fontweight='bold')

    for row, path in enumerate(paths):
        img_pil = Image.open(path).convert('RGB')

        orig  = tf_orig(img_pil)
        re_t  = tf_re(img_pil)
        str_t = tf_strip(img_pil)

        for col, tensor in enumerate([orig, re_t, str_t]):
            ax = axes[row][col]
            ax.imshow(denorm(tensor))
            ax.axis('off')

            # Superponer líneas de franja solo en col 2
            if col == 2:
                for (y0, y1), color, lbl in zip(regions, colors, region_labels):
                    rect = mpatches.Rectangle(
                        (0, y0), W, y1 - y0,
                        linewidth=1.5, edgecolor=color,
                        facecolor=color, alpha=0.15
                    )
                    ax.add_patch(rect)
                    ax.axhline(y0, color=color, linewidth=0.8, linestyle='--')
                    ax.axhline(y1, color=color, linewidth=0.8, linestyle='--')

    # Leyenda de franjas
    handles = [
        mpatches.Patch(color=c, label=l, alpha=0.6)
        for c, l in zip(colors, region_labels)
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=7, title='Regiones candidatas (Config 3)')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = 'aug_sanity_check.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print('Guardado:', out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Carpeta con imágenes de prueba (jpg/png)')
    args = parser.parse_args()
    main(args.img_dir)
