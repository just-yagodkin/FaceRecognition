# Notebooks

## Немного замечаний: 
Почти во всех ноутбуках встречаются одни и те же названия функций и/или классов, но несмотря на это они могут незначительно отличаться. На текущий момент я не выношу их в отдельный модуль, чтобы ~~моя работа казалась больше и круче~~ избежать путаницы при разборе кода в ноутбуках.
Все ноутбуки воспроизводимы!
Все обучения проходят в рамках датасета [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

- [00.FaceRecognition_baseline](./00.FaceRecognition_baseline.ipynb) - Обучение нейросети для распознавания лиц (базовое решение)
- [01.FaceRecognition_ArcFace](./01.FaceRecognition_ArcFace.ipynb) - Обучение на распознавание лиц с новой реализованной функцией потерь **ArcFace loss**  (Additive Angular Margin Loss)
- [02.FaceDetection](./02.FaceDetection.ipynb) - Знакомство с инструментами детекции и выравнивания лиц
- [03.IdentificationRate_metric](./03.IdentificationRate_metric.ipynb) - Реализация новой метрики **Identification rate**, проверка базового решения на ее основе
- [04.0.FaceRecognition_Triplet](./04.0.FaceRecognition_Triplet.ipynb) и [04.1.FaceRecognition_Triplet](./04.1.FaceRecognition_Triplet.ipynb) - ~~Обучение на распознавание лиц с новой реализованной функцией потерь **Triplet loss**~~ Жалкие попытки обучиться на триплетах... Результата никакого, но я старался, и, как мне кажется, логика там верна. Ноутбуки отличаются лишь разными функциями метрики (dist)
- [05.Demo](./05.Demo.ipynb) - Демонстрация работы "системы" (эмбеддинги для лиц прямо с фотографии)