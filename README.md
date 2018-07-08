# Классификатор звуков. Как запустить

1. Cкачайте и распакуйте [архив c google drive](https://drive.google.com/open?id=1_UQRTDw_lQfF6e6QoqojGrG1J5Rzpxee)

2. В папку *data_v_7_stc* скопируйте файл *classifier.py*

3. Запустите *classifier.py* командой
```bash
    python3 classifier.py -m -l -p
```

Для работы скрипта необходимо предварительно установить библиотеки
```bash
    pip3 install numpy pandas sckit-learn scipy
```

4. Стандартный файл *result.txt* заменится на новый, в котором будут результаты классификатора на звуки из папки *test*

* Дополнительно о ключах:

-m: выделить основные параметры аудиофайлов из папки *audio* (используя данные из файла *meta/meta.txt*) и записать их в файл *data.txt*

-l: обучить модель, используя параметры из файла *data.txt* и правильные ответы из файла *meta/meta.txt*, и записать её в файл *model.pkl*

-p: использовать модель из файла *model.pkl* для классификации аудиофайлов из папки *test* и записать результат в файл *result.txt*
