

# Entrenador LoRA de Edición In-Contexto para Krea 2

Un entrenador independiente y configurable para enseñar a Krea 2 Raw la edición de imágenes basada en instrucciones con LoRA. Esto no es entrenamiento de texto a imagen: cada imagen de entrada condiciona el modelo a través tanto del codificador visuo-lingüístico como de los tokens de referencia limpios del VAE.

## Arquitectura de edición

La pasada de entrenamiento usa:

```text
[ image-grounded Qwen3-VL instruction
| clean source VAE tokens, RoPE frame 1
| noisy target VAE tokens, RoPE frame 0 ]
```

La imagen de entrada se proporciona dos veces:

1. Qwen3-VL ve la imagen de entrada junto con la instrucción de edición.
2. El DiT de Krea 2 ve el latente limpio de la entrada como tokens de imagen en contexto.

El primer eje de RoPE es un identificador de fotograma, no un cambio en theta de RoPE. El entrenamiento usa flow matching:

```text
x_t = (1 - t) * target + t * noise
velocity_target = noise - target
```

La modulación de tokens de referencia es configurable:

- `shared`: los tokens de referencia y objetivo usan el paso de tiempo de destino muestreado. Esto preserva el comportamiento original del entrenador/nodo.
- `zero`: los tokens de referencia limpios usan modulación `t=0` en cada bloque DiT.
- `blend`: interpola entre ambos para experimentos controlados.

El perfil de 32 GB incluido usa `zero`.

`configs/edit_lora_1008x672_t0_arcface_strong_from500.yaml` es un perfil de preservación de identidad de segunda etapa. Reanuda un punto de control estructuralmente adaptado, inicia un optimizador fresco con una tasa de aprendizaje más baja y aplica un objetivo ArcFace más fuerte. Úsalo solo después de que el perfil base haya generado su punto de control de paso 500 referenciado, o actualiza `resume_from` a otro punto de control validado.

## Estructura del repositorio

```text
krea2_edit_lora/
  configs/                       Preajustes de entrenamiento
  integrations/                  Parche de compatibilidad con ComfyUI
  src/dataset.py                 Conjunto de datos y geometría de imagen
  src/identity.py                Pérdida de identidad ArcFace diferenciable
  src/krea_edit.py               Qwen, VAE, pasada de edición y muestreador
  src/latent_cache.py            Conjunto de datos de entrenamiento en caché
  src/lora.py                    Inyección de LoRA, cuantización y exportación
  tools/build_manifest.py        Conjunto de datos de carpetas a JSONL
  tools/cache_dataset.py         Caché de VAE y Qwen
  tools/cache_identity.py        Alineación facial y caché de identidad
  tools/compress_lora_rank.py    Conversión de rango SVD
  tools/download_models.py       Descarga automática de modelos
  train.py                       Entrenador, puntos de control, muestras y W&B
```

## Conjunto de datos

La estructura de carpetas es:

```text
dataset/
  control/
    1.png
    2.png
  target/
    1.jpg
    1.txt
    2.jpg
    2.txt
```

Los nombres base coincidentes forman un par. El control es la imagen antes de la edición, el objetivo es el resultado deseado y el archivo de texto contiene una instrucción de edición:

```text
Convert the character in the image to a character sheet showing a face close-up, front, side, and back full-body views.
```

No reemplaces la instrucción con una descripción plana de la imagen objetivo. El modelo debe aprender la transformación.

Genera manifiestos de entrenamiento y validación auditables:

```bash
uv run python tools/build_manifest.py \
  --dataset ../dataset \
  --output ../dataset/train.jsonl \
  --validation-count 8
```

Una fila JSONL es:

```json
{"id":"1","control":"control/1.png","target":"target/1.jpg","caption":"Convert the character in the image to a character sheet."}
```

Las múltiples referencias usan una matriz `controls` ordenada. Las referencias reciben IDs de fotograma RoPE 1, 2, etc.:

```json
{"id":"scene-person","controls":["control/scene.png","control/person.png"],"target":"target/result.png","caption":"Place the person beside the window."}
```

## Entorno y modelos

El entrenamiento se probó en Ubuntu 24.04 bajo WSL con CUDA. Desde el repositorio:

```bash
git clone https://github.com/krea-ai/krea-2 vendor/krea-2
uv sync
hf auth login
wandb login
```

Descarga automáticamente todos los modelos requeridos:

```bash
uv run python tools/download_models.py --output-dir models
```

El entrenador necesita:

- `krea/Krea-2-Raw`
- `Qwen/Qwen3-VL-4B-Instruct`
- the Qwen Image VAE
- the official Krea 2 Python implementation in `vendor/krea-2`

Se puede usar directamente un caché existente de Hugging Face. Establece `model.checkpoint`, `model.text_encoder` y `model.vae` en el YAML. La estación de trabajo actual usa:

```text
/home/alissonerdx/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/b2e772263cfa934848fde713159d1553e086778c/raw.safetensors
```

Ejecuta los comandos desde WSL:

```bash
cd /mnt/d/Projects/krea/krea2_edit_lora
```

## Construir los cachés de entrenamiento

El caché principal almacena latentes VAE deterministas de objetivo/referencia y características de Qwen ancladas a imagen, tanto con subtítulo como sin él:

```bash
uv run python tools/cache_dataset.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml
```

Reconstrúyelo después de cambiar el manifiesto, la resolución, la geometría, el VAE, el codificador de texto o el presupuesto de anclaje.

ArcFace es opcional. Cuando está habilitado, construye un segundo caché que contenga el incrustado de identidad de la fuente y la alineación facial del objetivo:

```bash
uv run --no-sync \
  --with "opencv-python-headless>=4.11,<5" \
  python tools/cache_identity.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml
```

La herramienta de caché descarga automáticamente los pesos de OpenCV YuNet y ArcFace. Informa cada par para el cual no se pudo detectar ninguna de las caras. La pérdida de identidad se omite para pares inválidos.

Los pesos de ArcFace incluidos están restringidos para uso de investigación no comercial. Deshabilita `identity_loss.enabled` o proporciona pesos alternativos legalmente compatibles para otros casos de uso. Consulta la [tarjeta del modelo ArcFace](https://huggingface.co/py-feat/arcface_r50).

## Perfil de 32 GB incluido

`configs/edit_lora_1008x672_cached_32gb.yaml` usa:

| Configuración | Valor |
| --- | --- |
| Resolución de salida | `1008 × 672` (3:2, lado más grande por debajo de 1024) |
| Precisión | BF16 |
| Base congelada | Solo pesos INT8 |
| LoRA | rango 64, alfa 64, 256 módulos de atención/MLP |
| Optimizador | AdamW de 8 bits |
| LR | coseno `5e-5` a `1e-5`, calentamiento de 25 pasos |
| Paso de tiempo de referencia | `zero` |
| ArcFace | peso `0.05`, cada 8 pasos, solo cuando `t <= 0.8` |
| Intervalo de punto de control/muestra | 250 pasos |
| Muestreo raw | 52 pasos, CFG 3.5 |

La ruta de identidad reconstruye `x0` desde la predicción de flujo, recorta una ROI facial con relleno en el espacio latente, decodifica solo esa ROI, la alinea de forma diferenciable y calcula la pérdida de identidad por coseno. Esto mantiene la memoria pico de ArcFace cerca de un paso de entrenamiento normal.

Establece `identity_loss.enabled: false` para un experimento puro de flow matching. Los modos de referencia `shared`, `zero` y `blend` permanecen disponibles de forma independiente.

### Inicializar rango 64 desde un adaptador de rango superior

La configuración de la estación de trabajo incluida inicializa desde:

```text
weights/r64-from-r256-step500.safetensors
```

Este archivo fue producido por compresión SVD. Convierte otro adaptador con:

```bash
uv run python tools/compress_lora_rank.py \
  outputs/<run>/checkpoint-0000500/adapter.safetensors \
  weights/r64-init.safetensors \
  --rank 64 --alpha 64
```

Establece `init_adapter: null` para iniciar un nuevo LoRA de rango 64 desde cero. `init_adapter` carga solo los pesos del adaptador y comienza un optimizador fresco en el paso cero.

## Entrenamiento

Ejecuta primero las verificaciones de pre-vuelo:

```bash
uv run python tools/preflight.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml
```

Inicia el entrenamiento:

```bash
uv run accelerate launch train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml
```

Sobrescrituras útiles para prueba rápida (smoke-test):

```bash
uv run python train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --output-dir outputs/smoke \
  --max-steps 1 \
  --identity-every 1 \
  --identity-max-t 1.0 \
  --no-wandb
```

Forzar una vista previa de validación de una sola imagen:

```bash
uv run python train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --output-dir outputs/preview-smoke \
  --max-steps 1 --sample-every 1 --sample-count 1 --no-wandb
```

Los optimizadores soportados son `adamw`, `adamw8bit`, `adanw` y `prodigy`. Los modos de cuantización de base congelada son `none`, `int8`, `float8` y `uint4` experimental.

## Puntos de control, reanudación y W&B

Cada punto de control contiene:

```text
outputs/<run>/checkpoint-0000250/
  adapter.safetensors
  trainer_state.pt
  config.yaml
```

Reanudar pesos, paso, optimizador y estado RNG:

```bash
uv run accelerate launch train.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --resume outputs/<run>/checkpoint-0000250
```

Establece `train.resume_optimizer: false` para cargar los pesos y el paso del punto de control mientras inicias un perfil de optimizador fresco.

W&B recibe:

- pérdida total y pérdida de flujo
- pérdida de ArcFace, pérdida ponderada, similitud coseno y paso de tiempo de identidad muestreado
- tasa de aprendizaje y rendimiento
- VRAM asignada y pico
- conteo de módulos/parámetros de LoRA
- imágenes de validación de control, objetivo y generadas lado a lado

Las muestras locales se escriben en `outputs/<run>/samples/step-XXXXXXX/`.

Mide la similitud ArcFace de fuente/objetivo/generado para una carpeta de muestras:

```bash
uv run --no-sync \
  --with "opencv-python-headless>=4.11,<5" \
  python tools/measure_identity.py \
  --config configs/edit_lora_1008x672_cached_32gb.yaml \
  --samples outputs/<run>/samples/step-0000250
```

## Inferencia con ComfyUI

El adaptador exportado requiere el [paquete de nodos ComfyUI Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit). Usa su parche de modelo de latente de fuente y nodos de codificación anclados a imagen, conectando la misma imagen de entrada a ambas rutas.

Los adaptadores entrenados con `reference_timestep: shared` funcionan con la pasada estándar del paquete. Los adaptadores entrenados con `reference_timestep: zero` requieren ComfyUI actual más el parche de compatibilidad incluido:

```bash
cd ComfyUI/custom_nodes/comfyui-krea2edit
git apply /path/to/krea2_edit_lora/integrations/comfyui-krea2edit-t0.patch
```

Reinicia ComfyUI y selecciona `zero` en el nodo de parche de modelo. Mantén `ref_boost=1.0` para este modo. El parche reordena la secuencia de autoatención a `[text | target | references]`; dado que la atención no es causal y los IDs de RoPE permanecen sin cambios, esto es equivalente al conjunto de tokens de entrenamiento mientras permite la modulación nativa de `t=0` basada en sufijo de ComfyUI.

Usa la misma geometría de referencia `anchor` o `fit` que en el entrenamiento. El perfil incluido usa `anchor`. Las vistas previas de Krea 2 Raw usan 52 pasos y CFG 3.5.

## Créditos

La condición de edición de este entrenador fue reconstruida desde [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) por [@lbouaraba](https://github.com/lbouaraba). Sus nodos son la implementación de referencia del diseño contra el que este repositorio entrena — `[text | references (RoPE frames 1..N) | target (frame 0)]`, el paso de tiempo de referencia compartido y la codificación de instrucciones de Qwen anclada a imagen — y leerlos es lo que hizo posible un entrenador coincidente.

La geometría de referencia sigue el lanzamiento del paquete de nodos por lanzamiento, porque el entrenamiento y la inferencia deben coincidir byte por byte o el adaptador subutilizará la fuente. Dos correcciones de v1.2.4 se reflejan aquí: el eje ajustado se alcanza recortando la fuente para que caiga en la cuadrícula /16 a escala exacta (redimensionar directamente al tamaño redondeado a 16 aplasta el contenido hasta 15 px y duplica la banda en los bordes de referencia), y los desplazamientos de referencia centrados son fraccionarios en lugar de redondeados a entero, ya que las posiciones de RoPE son continuas.

## Seguridad y licencia

Usa solo imágenes para las que tengas autorización para procesarlas. Respeta el consentimiento, la privacidad, la imagen personal, los derechos de autor, la licencia de Krea 2 y las licencias de los modelos de identidad opcionales. Los cachés de modelos, las salidas, los datos de W&B y los pesos `.safetensors` están excluidos de Git.
