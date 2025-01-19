# Notebooks

## Немного замечаний: 
Почти во всех ноутбуках встречаются одни и те же названия функций и/или классов, но несмотря на это они могут незначительно отличаться. На текущий момент я не выношу их в отдельный модуль, чтобы ~~моя работа казалась больше и круче~~ избежать путаницы при разборе кода в ноутбуках.
Все ноутбуки воспроизводимы!
Все обучения проходят в рамках датасета [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

- [00.FaceRecgognition_baseline](./00.FaceRecgognition_baseline.ipynb) - Обучение нейросети для распознавания лиц (базовое решение)
- [01.FaceRecognition_ArcFace](./01.FaceRecgognition_ArcFace.ipynb) - Обучение на распознавание лиц с новой реализованной функцией потерь **ArcFace loss**  (Additive Angular Margin Loss)
- [02.FaceDetection](./02.FaceDetection.ipynb) - Знакомство с инструментами детекции и выравнивания лиц
- [03.IdentificationRate_metric] - Реализация новой метрики **Identification rate**, проверка базового решения на ее основе
- [04.FaceRecognition_Triplet] - Обучение на распознавание лиц с новой реализованной функцией потерь **Triplet loss**
- [05.FaceRecognition_mix] - Обучение на смеси функций **ArcFace loss** & **Triplet loss**
- [06.FaceRecognition_full] - Выбор лучшего решения, прогон всего пайплайна (эмбеддинги для лиц прямо с фотографии)