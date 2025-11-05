# 🚀 Proyecto SPH-GPU: 

Una implementación de **Hidrodinámica de Partículas Suavizadas (SPH)** acelerada por GPU, escrita en Python utilizando Cupy-Cuda

Este proyecto simula El comportamiento de un fluido incompresible.

---

## 🌟 Características Principales

* **Aceleración por GPU:** Utiliza [**CuPy/CUDA**] para paralelizar los cálculos de SPH (búsqueda de vecinos, cálculo de densidad, fuerzas) logrando un rendimiento significativamente superior al de una CPU.
* **Implementación en Python:** Código limpio y legible que aprovecha el ecosistema científico de Python (NumPy, SciPy).
* **Soporte 2D/3D:** Capaz de ejecutar simulaciones tanto en 2 como en 3 dimensiones.
* **Visualización (Opcional):** Incluye scripts para animar la simulacion usando matplotlib

## 🌠 Visualización de Ejemplo

[Image of an SPH simulation GIF]
> *Una breve descripción de la simulación. Ej: "Simulación 2D de una 'presa rota' (dam break) con 50,000 partículas."*

(Reemplaza la línea de arriba con un GIF o una imagen de tu simulación. Puedes subir la imagen a tu repositorio de GitHub y enlazarla).

## 🛠️ Requisitos e Instalación

Asegúrate de tener un hardware compatible con CUDA y los drivers de NVIDIA actualizados.

### 1. Dependencias Clave

* Python (3.9+)
* `cupy` (para la aceleración GPU)
* `numpy` (para manejo de arrays)
* `matplotlib` (para visualización, si aplica)
* [**Cualquier otra biblioteca, ej: `scipy`, `tqdm`**]

### 2. Instalación

1.  Clona este repositorio:
    ```bash
    git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
    cd TU_REPOSITORIO
    ```

2.  (Recomendado) Crea y activa un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
    *(Asegúrate de tener un archivo `requirements.txt` con las bibliotecas listadas arriba)*.

## ⚡ Cómo Usar el Simulador

Para ejecutar una simulación predeterminada, simplemente corre:

```bash
python main.py --config configs/mi_simulacion.json
