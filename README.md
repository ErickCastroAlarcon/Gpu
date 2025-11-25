# 🚀 Proyecto SPH-GPU: 

Una implementación de **Hidrodinámica de Partículas Suavizadas (SPH)** acelerada por GPU, escrita en Python utilizando Cupy-Cuda.

---

## 🌟 Características Principales

* **Aceleración por GPU:** Utiliza [**CuPy/CUDA**] para paralelizar los cálculos de SPH (búsqueda de vecinos, cálculo de densidad, fuerzas) logrando un rendimiento significativamente superior al de una CPU.
* **Optimización Espacial:** Implementación de una búsqueda de vecinos basada en *Spatial Hashing*, reduciendo la complejidad a $O(N)$.
* **Implementación en Python:** Código limpio y legible que aprovecha el ecosistema de Python (NumPy, Numba).
* **Soporte 2D/3D:** Capaz de ejecutar simulaciones tanto en 2 como en 3 dimensiones.
* **Visualización:** Incluye scripts para animar la simulacion usando vispy

## 🌠 Visualización de Ejemplo
<img src="https://github.com/user-attachments/assets/80c20234-0186-454c-8ba3-b6e8e43ecfad" scale=0.5/>
**Simulación de 80000 particulas SPH**

## Resultados
https://drive.google.com/drive/folders/1DwhmC2sk5G8yHT5xN7Agd_rZS4IxSIeH?usp=drive_link

## 🛠️ Requisitos

Tener un hardware compatible con CUDA y los drivers de NVIDIA actualizados.

### Dependencias Clave

* Python (3.9+)
* `cupy` (para la aceleración GPU)
* `numpy` (para manejo de arrays)
* `vispy` (para visualización)
* numba (para eficiencia)
