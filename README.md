# MLOps @ Mioti

## Instalar librerías
Si estáis utilizando Conda, podéis instalar las librerías con el siguiente comando, siempre que estéis en el directorio en el que se encuentra el archivo:

```bash
conda install --file requirements
```

Recordad que para todos los pasos tenéis que tener activo el entorno conda.

## Experiment Tracking
El primer paso es lanzar MLFlow para poder guardar experimentos en local.

```bash
cd experiment_tracking

mlflow server --port 5000
```

Esto hará que tengáis disponible MLFlow en esta url: [http://127.0.0.1:5000](http://127.0.0.1:5000). Esta misma url es la que tenéis que utilizar en los archivos:

```python
mlflow.set_tracking_uri('http://127.0.0.1:5000')
```

Una vez todo esté listo, sólo tenéis que lanzar el código como `python [nombre-archivo].py`

## API
Podéis usar la configuración de Pycharm para lanzar la API (mejor ver el video de la clase) o desde el terminal.
Lo primero que tenéis que hacer es aseguraros que estáis en el directorio correcto. Desde el directorio principal del repo, podéis hacer `cd api`.
Una vez dentro, tenéis que levantar la API con el siguiente comando:

```bash
uvicorn main:app --reload
```

donde ```main:app``` equivale al `nombre_archivo:nombre_variable_asignada_a_FastAPI()`. Una vez levantada, podéis acceder a ella aquí: [http://127.0.0.1:8000](http://127.0.0.1:8000)
Recordad que la url para la documentación que vimos en clase es la siguiente: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Forking this repo
Como comentamos en clase, quiero que entregueis el challenge de la API (y el challenge de MLFlow si lo queréis hacer) como forks al repo.
Os dejo una guía: https://drupal.gatech.edu/handbook/using-pull-requests-forks