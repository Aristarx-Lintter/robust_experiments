# Robust experiments

Для восстановления окружения экспериментов необходимо загрузить веса и датасет и создать окружение:

1. `make load_dataset`
2. `make load_weights`
3. `make build_pure`
4. `make run_docker_pure_bash`
5. `make jupyter`


Для сбора окружения TensorRT и OpenVINO:
1. `make build_trt`
2. `make run_docker_trt_bash`
3. `make jupyter`

## Артефакты экспериментов

В папке notebooks содержатся результаты запуска моди с автолорированием и ускорение модели на TRT и OpenVINO




