# TMGF con parches verticales solapados y oclusión dirigida

Este repositorio desarrolla y evalúa una extensión del método **Transformer-Based Multi-Grained Features (TMGF)** para Re-Identificación de personas no supervisada. El trabajo plantea una estructura de representación basada en **parches verticales solapados** y compara distintas estrategias de oclusión sintética.

Se toma como punto de partida la implementación original de TMGF presentada por Li, Wang y Gong en _Transformer-Based Multi-Grained Features for Unsupervised Person Re-Identification_ (WACV 2023 Workshops) [[artículo original](https://openaccess.thecvf.com/content/WACV2023W/RWS/html/Li_Transformer_Based_Multi-Grained_Features_for_Unsupervised_Person_Re-Identification_WACVW_2023_paper.html)].

## Relación con el trabajo original

El trabajo previo divide la salida espacial del Transformer en ramas de distinta granularidad y aprende características globales y locales sin supervisión manual. Esta propuesta conserva ese marco y extiende su representación local:

- mantiene las dos granularidades originales: divisiones verticales en 2 y 3 regiones;
- utiliza cinco características locales totales;
- conserva las pérdidas, el banco de memoria y el agrupamiento proxy de TMGF;
- modifica las regiones locales para permitir **solapamiento entre franjas adyacentes**;
- introduce una oclusión sintética dirigida a las franjas locales.

La cita bibliográfica del trabajo base se incluye al final de este README.

## Idea central: parches solapados

El mapa de características se divide verticalmente siguiendo las granularidades configuradas en `MODEL.GRANULARITIES`:

```text
Rama 2-split: |------- A -------|------- B -------|
Rama 3-split: |--- C ---|--- D ---|--- E ---|
```

Cada franja ocupa una sección con un margen adicional de solapamiento. Para una altura de mapa `H`, una granularidad `g` y una proporción `r = 0.20`, la altura base de cada franja es `H/g` y el margen compartido es:

```text
overlap = round(r * (H / g))
start_i  = max(0, i * stripe_h - overlap)
end_i    = min(H, (i + 1) * stripe_h + overlap)
```

De esta manera, las regiones vecinas comparten información en sus bordes y reducen la discontinuidad que podría producirse al dividir al cuerpo en partes estrictamente disjuntas.

La implementación propuesta se encuentra en `OverlappedPartPool2d` dentro de [libs/models/vit_encoder.py](libs/models/vit_encoder.py). Actualmente, el bloque tradicional de pooling permanece activo y el bloque solapado está disponible como propuesta dentro del constructor de `TMGF`; debe activarse al sustituir el `AvgPool2d` original por `OverlappedPartPool2d`.

## Estrategias experimentales

Se comparan tres configuraciones. Todas utilizan ViT-S/16+ICS, entrada `384 x 128`, granularidades `[2, 3]`, cinco partes locales y el conjunto OCC-Duke para las evaluaciones principales.

| Configuración | Configuración YACS | Modelo local | Aumentación |
|---|---|---|---|
| 1. Solapamiento puro | `configs/exp_cfg1_baseline.yml` | Propuesta solapada | Random Erasing estándar |
| 2. Solapamiento + RE intensificado | `configs/exp_cfg2_re_intensified.yml` | Propuesta solapada | Random Erasing con mayor probabilidad y área |
| 3. Solapamiento + oclusión dirigida | `configs/exp_cfg3_stripe_directed.yml` | Propuesta solapada | Oclusión alineada con franjas |

### Configuración 1: referencia

Usa Random Erasing estándar con probabilidad `0.5`. Sirve como referencia respecto al Random Erasing original y permite separar el efecto del nuevo pooling del efecto de una aumentación distinta.

### Configuración 2: control de intensidad

Usa Random Erasing intensificado:

- probabilidad `0.7`;
- área borrada relativa `0.02–0.50`;
- proporción de aspecto limitada por `r1 = 0.3`.

Esta variante no dirige la oclusión a una franja concreta. Su objetivo es comprobar si una mejora proviene únicamente de la mayor cantidad de oclusión o de su localización espacial.

### Configuración 3: oclusión dirigida

`StripeDirectedErasing`, implementado en [libs/utils/data/transforms.py](libs/utils/data/transforms.py), selecciona una de cinco regiones candidatas derivadas de las franjas solapadas:

1. franja superior del split de tres partes;
2. franja central del split de tres partes;
3. franja inferior del split de tres partes;
4. zona de transición superior-media;
5. zona de transición media-inferior.

Dentro de la región elegida, el sistema:

- aplica la oclusión con probabilidad `0.5`;
- ocupa aproximadamente el `20–40 %` del área de la región objetivo;
- rellena la máscara con la media estadística de LUPerson;
- mantiene el ancho máximo de la imagen y evita máscaras válidas degeneradas.

Las regiones se construyen con la misma proporción de solapamiento (`0.20`) usada por el pooling propuesto.

## Requisitos

- Python;
- PyTorch;
- torchvision;
- yacs;
- CUDA recomendada para el entrenamiento completo.

Instalación sugerida:

```bash
conda create -n tmgf-overlap python=3.7
conda activate tmgf-overlap
pip install -r requirements.txt
```

El código depende de un peso previo ViT-S/16+ICS. La ruta predeterminada se establece en `MODEL.PRETRAIN_PATH`.

## Preparación de datos

Coloque los conjuntos dentro de `datasets/` o configure `DATASET.ROOT_DIR`:

```text
datasets/
├── Market1501/
├── DukeMTMC-reID/
├── MSMT17/
└── OCC-Duke/
```

Los nombres de dataset reconocidos se cargan desde [libs/datasets](libs/datasets). Para OCC-Duke, utilice `DATASET.NAME OCCDuke`.

## Entrenamiento

Las instrucciones principales están centralizadas en [train.sh](train.sh). Para entrenar las tres variantes en OCC-Duke:

```bash
python train.py \
  --conf configs/exp_cfg1_baseline.yml \
  TASK_NAME cfg1_solapamiento_occd_duke \
  DATASET.ROOT_DIR ./datasets \
  DATASET.NAME OCCDuke \
  MODEL.SIE_CAMERA 8 \
  MEMORY_BANK.POS_K 2

python train.py \
  --conf configs/exp_cfg2_re_intensified.yml \
  TASK_NAME cfg2_re_intensificado_occd_duke \
  DATASET.ROOT_DIR ./datasets \
  DATASET.NAME OCCDuke \
  MODEL.SIE_CAMERA 8 \
  MEMORY_BANK.POS_K 2

python train.py \
  --conf configs/exp_cfg3_stripe_directed.yml \
  TASK_NAME cfg3_oclusion_dirigida_occd_duke \
  DATASET.ROOT_DIR ./datasets \
  DATASET.NAME OCCDuke \
  MEMORY_BANK.POS_K 2
```

Los registros se crean como `YYYYMMDD_TASK_NAME.txt` en `LOG.LOG_DIR`. Los pesos se guardan en `LOG.CHECKPOINT.SAVE_DIR/TASK_NAME`.

Cada archivo de configuración puede modificar valores desde la línea de comandos. Ejemplo:

```bash
python train.py \
  --conf configs/exp_cfg3_stripe_directed.yml \
  TRAIN.EPOCHS 50 \
  TRAIN.ITERS 200 \
  TRAIN.BATCHSIZE 32
```

## Visualización de la oclusión dirigida

Antes de entrenar, puede comprobar visualmente las máscaras generadas:

```bash
python debug_augmentation.py --img_dir /ruta/a/imagenes
```

El programa genera `aug_sanity_check.png` con tres columnas:

- imagen original;
- Configuración 2 con Random Erasing intensificado;
- Configuración 3 con oclusión dirigida.

## Evaluación

```bash
python evaluate.py \
  --conf configs/exp_cfg3_stripe_directed.yml \
  --weight /ruta/al/modelo/weight.pth
```

La evaluación utiliza CMC y Rank-1/mAP según la implementación de [libs/evaluators.py](libs/evaluators.py). El re-ranking puede activarse con:

```bash
python evaluate.py \
  --conf configs/exp_cfg3_stripe_directed.yml \
  --weight /ruta/al/modelo/weight.pth \
  TEST.RE_RANK True
```

## Diseño experimental recomendado

Para atribuir correctamente cada mejora, se recomienda mantener fijos:

- backbone y peso preentrenado;
- resolución de entrada;
- semillas;
- número de épocas e iteraciones;
- dataset y partición;
- pseudoetiquetas y hiperparámetros de memoria;
- estrategia de agrupamiento;
- protocolo de evaluación.

La interpretación esperada es:

1. **Configuración 1** establece la referencia con parches solapados y Random Erasing estándar;
2. **Configuración 2** mide el efecto de intensificar la oclusión;
3. **Configuración 3** determina si alinear la oclusión con las franjas aporta una ganancia adicional sobre el control intensificado.

## Estructura relevante

```text
configs/
├── exp_cfg1_baseline.yml
├── exp_cfg2_re_intensified.yml
├── exp_cfg3_stripe_directed.yml
└── default.py
libs/
├── models/vit_encoder.py       # arquitectura y pooling propuesto
├── utils/data/transforms.py    # oclusión dirigida
├── utils/prepare_data.py       # registro de pipelines de aumentación
├── trainers.py                 # bucle de entrenamiento
└── evaluators.py               # extracción y evaluación
debug_augmentation.py           # validación visual
train.py                        # entrenamiento
evaluate.py                     # evaluación
train.sh                        # comandos de referencia
```

## Métricas y registros

La salida de entrenamiento incluye:

- descripción de la configuración;
- número de clusters detectados;
- cantidad de proxies generados;
- outliers excluidos;
- learning rate;
- resultados periódicos de evaluación;
- pesos guardados a intervalos configurados.

## Resultados

Los resultados se expresan como mAP y CMC en Rank-1, Rank-5 y Rank-10, de acuerdo con las métricas que registra el evaluador. **Baseline** es la referencia y **Contribución** corresponde a la propuesta de este trabajo.

| Dataset | Configuración | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---:|---:|---:|---:|
| Market1501 | Baseline (referencia) | 89.3 | 95.7 | 98.0 | 98.8 |
| Market1501 | Contribución | 89.8 | 95.7 | 98.3 | 99.0 |
| DukeMTMC | Baseline (referencia) | 76.2 | 86.0 | 93.0 | 94.2 |
| DukeMTMC | Contribución | 76.4 | 86.4 | 93.4 | 94.7 |
| MSMT17 | Baseline (referencia) | 57.1 | 83.1 | 89.9 | 91.8 |
| MSMT17 | Contribución | 59.2 | 84.1 | 90.5 | 92.5 |

Resultados adicionales en OCC-Duke:

| Dataset | Configuración | Rank-1 | mAP |
|---|---|---:|---:|
| OCC-Duke | 1. Referencia | 40.5 | 48.5 |
| OCC-Duke | 2. RE intensificado | 47.4 | 55.9 |
| OCC-Duke | 3. Oclusión dirigida | 40.5 | 48.4 |

Las imágenes existentes `assets/pipeline.png` y `assets/perf.png` corresponden a la implementación TMGF original y pueden utilizarse como referencia visual del método base.

## Créditos

La implementación se basa en TMGF y agradece las bases de código de TransReID, TransReID-SSL y O2CAP.

## Citación

Este trabajo extiende el método original. Si utiliza esta implementación o el enfoque descrito, cite ambos trabajos cuando corresponda.

### Trabajo original base

```bibtex
@InProceedings{Li_2023_WACV,
    author    = {Li, Jiachen and Wang, Menglin and Gong, Xiaojin},
    title     = {Transformer Based Multi-Grained Features for Unsupervised Person Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2023},
    pages     = {42-50}
}
```
