import os.path as osp
import sys
from torchvision import transforms as TV
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../..')))

from torch.utils.data import DataLoader
from libs import datasets
from .data import transforms as T
from .data import IterLoader
from .data.sampler import ClassUniformlySampler, RandomMultipleGallerySampler, ClusterProxyBalancedSampler
from .data.preprocessor import Preprocessor, CameraAwarePreprocessor


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    print('root path= {}'.format(root))
    dataset = datasets.create(name, root)
    return dataset


# ---------------------------------------------------------------------------
# Factorías de aumentación  (§4 del documento de mejora)
# ---------------------------------------------------------------------------

def _build_augmentation_baseline(cfg, height, width):
    """
    Config 1 – Solapamiento puro.
    Pipeline original de la tesis con RandomErasing estándar (p=0.5).
    """
    normalizer = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    return T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # moderado
        #T.RandomGrayscale(p=0.1),
        #T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=cfg.INPUT.PIXEL_MEAN),
    ])


def _build_augmentation_re_intensified(cfg, height, width):
    """
    Config 2 – Solapamiento + Random Erasing intensificado.
    Controla la cantidad de oclusión aumentando sl/sh para producir
    un nivel comparable al de la oclusión dirigida (20-40 % del área
    de una franja, que equivale a ~7-13 % del área total de la imagen).
    Se eleva sh de 0.4 → 0.5 y la probabilidad de 0.5 → 0.7 para
    compensar que RE elige toda la imagen como candidato, no solo la franja.
    """
    normalizer = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    return T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # moderado
        #T.RandomGrayscale(p=0.1),
        #T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        normalizer,
        T.RandomErasing(probability=0.7, sl=0.02, sh=0.5, r1=0.3,
                        mean=cfg.INPUT.PIXEL_MEAN),
    ])


def _build_augmentation_stripe_directed(cfg, height, width):
    """
    Config 3 – Solapamiento + Oclusión dirigida por franjas (PROPUESTA).
    Reemplaza RandomErasing por StripeDirectedErasing con la misma semántica
    de relleno (valor de canal ≈ mean) pero localización alineada con las
    franjas del modelo.
    """
    normalizer = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    return T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # moderado
        #T.RandomGrayscale(p=0.1),
        #T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        normalizer,
        T.StripeDirectedErasing(
            probability=0.5,
            img_h=height,
            img_w=width,
            granularities=tuple(cfg.MODEL.GRANULARITIES),
            overlap_ratio=0.2,
            severity_low=0.20,
            severity_high=0.40,
            mean=tuple(cfg.INPUT.PIXEL_MEAN),
        ),
    ])


# Mapa de nombre → factory  (se selecciona via cfg.AUGMENTATION.TYPE)
_AUG_REGISTRY = {
    'baseline':        _build_augmentation_baseline,
    're_intensified':  _build_augmentation_re_intensified,
    'stripe_directed': _build_augmentation_stripe_directed,
}

# ---------------------------------------------------------------------------

def get_train_loader(cfg, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):

    # Seleccionar pipeline de aumentación
    aug_type = getattr(cfg.AUGMENTATION, 'TYPE', 'baseline') \
               if hasattr(cfg, 'AUGMENTATION') else 'baseline'

    if aug_type not in _AUG_REGISTRY:
        raise ValueError(
            'AUGMENTATION.TYPE desconocido: "{}". '
            'Opciones válidas: {}'.format(aug_type, list(_AUG_REGISTRY.keys()))
        )

    print('=> Augmentation type: {}'.format(aug_type))
    train_transformer = _AUG_REGISTRY[aug_type](cfg, height, width)

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)

    # Sampler
    if cfg.SAMPLER.TYPE == 'proxy_balance':
        sampler = ClassUniformlySampler(train_set, class_position=4, k=num_instances)
    elif cfg.SAMPLER.TYPE == 'cluster_balance':
        sampler = ClassUniformlySampler(train_set, class_position=1, k=num_instances)
    elif cfg.SAMPLER.TYPE == 'cam_cluster_balance':
        sampler = RandomMultipleGallerySampler(train_set, class_position=1, num_instances=num_instances)
    elif cfg.SAMPLER.TYPE == 'cam_proxy_balance':
        sampler = RandomMultipleGallerySampler(train_set, class_position=4, num_instances=num_instances)
    elif cfg.SAMPLER.TYPE == 'cluster_proxy_balance':
        sampler = ClusterProxyBalancedSampler(train_set, k=num_instances)
    else:
        raise ValueError('Invalid sampler type name!')

    train_loader = IterLoader(
        DataLoader(
            CameraAwarePreprocessor(train_set, root=dataset.images_dir,
                                    transform=train_transformer),
            batch_size=batch_size, num_workers=workers, sampler=sampler,
            shuffle=False, pin_memory=True, drop_last=True
        ),
        length=iters
    )
    return train_loader


def get_test_loader(cfg, dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader
