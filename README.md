# 1. ОСНОВНАЯ ИНФОРМАЦИЯ: CAR VS DOGS = ДАТАСЕТ, СОДЕРЖАЩИЙ ИЗОБРАЖЕНИЯ АВТОМОБИЛЕЙ И МОТОЦИКЛОВ.
Подготовка датасета для нейросети, осуществляющей распознавание образов автомобилей и мотоциклов. Хакатон проводится в рамках зимней сессии 1-го курса в магистратуре "Инженерия машинного обучения" Уральского Федерального Университета. Работу выполнили:
- Колганов Максим Валерьевич, группа: РИМ-120963
- Башарина Анна Александровна, группа: РИМ-120963
- Шмелев Дмитрий Сергеевич, группа: РИМ-120963
- Шалагин Дмитрий Алексеевич, группа: РИМ-120963

---
# 2. МЕТРИКИ.
Различия между фоту определяются тремя метриками (общие для автомобилей и мотоциклов):
* `maker` - производитель
* `class` - типа кузова
* `angle` - ракурс

Однако их наполнение, будет различаться: например, есть производители, которые выпускают только мотоциклы (например, *Harley-Davidson*) или только автомобили (например, *Lamborghini*). Так же есть типы кузова, свойственные только автомобилям (например, *Hatchback*) или только мотоциклам (например, *Superbike*).

Состав метрик для `CARS`:
``` python
metrics_car = {
    'maker': ['Hyunday','Toyota','BMW','Mercedes','Lamborghini','Tesla','Honda'],
    'class': ['Saloon', 'Hatchback','Estate','Coupe','Convertible','Crossover'],
    'angle': ['Front','Back','Left','Right']
}
```

Состав метрик для `MOTO`:
``` python
metrics_moto = {
    'maker': ['Harley Davidson','Yamaha','Honda','Ducati','Kawasaki','BMW','Suzuki'],
    'class': ['Superbike', 'Classic','Motard','Dragster','Minibike','Sportbike',],
    'angle': ['Front','Back','Left','Right']
}
```
---
# 3. СОСТАВ ДАТАСЕТА.
Итого: мы имеем 4 `df`

*   `test_car_df`
*   `test_moto_df`
*   `train_car_df`
*   `train_car_df`

-> каждый из которых соответствует следующему набору данных
*   `test-car` | ТЕСТОВЫЕ ДАННЫЕ ДЛЯ МАШИН
*   `test-moto` | ТЕСТОВЫЕ ДАННЫЕ ДЛЯ МОТО
*   `train-car` | ТРЕНИРОВОЧНЫЕ ДАННЫЕ ДЛЯ МАШИН
*   `train-moto` | ТРЕНИРОВОЧНЫЕ ДАННЫЕ ДЛЯ МОТО

У каждого `df` определены слудующие столбцы:
`ID` - уникальный `df` в рамках ВСЕГО датасета, который формируется средствами `UUID`.
Оставшиеся стобцы соответствуеют выбранным метрикам:

*   `maker`
*   `class`
*   `angle`

*=> 4 дата-фрейма по 3 метрики в каждом => 12 графиков для каждой метрики => мы сможем на графиках отслеживать равномерность заполнения данных. Если, допустим, в датасете будет всего 3 фото для производителя `BMW` и 120 для `Ferrari`, то это будет отображено на гистограмме.*

## 3.1. Отслеживание состава датасета по количеству `maker` (производителей):

[![TEST-CAR-QUANTITY-MAKER.md.png](https://d.radikal.host/2023/01/20/TEST-CAR-QUANTITY-MAKER.md.png)](https://radikal.host/i/JB3KZx)
[![TEST-MOTO-QUANTITY-MAKER.md.png](https://d.radikal.host/2023/01/20/TEST-MOTO-QUANTITY-MAKER.md.png)](https://radikal.host/i/JB3VhI)
[![TRAIN-CAR-QUANTITY-MAKER.md.png](https://b.radikal.host/2023/01/20/TRAIN-CAR-QUANTITY-MAKER.md.png)](https://radikal.host/i/JB3igh)
[![TRAIN-MOTO-QUANTITY-MAKER.md.png](https://b.radikal.host/2023/01/20/TRAIN-MOTO-QUANTITY-MAKER.md.png)](https://radikal.host/i/JB3rpC)

## 3.2. Отслеживание состава датасета по количеству `class` (тип транспортного средства):
[![TEST-CAR-QUANTITY-CLASS.md.png](https://b.radikal.host/2023/01/20/TEST-CAR-QUANTITY-CLASS.md.png)](https://radikal.host/i/JBCgFu)
[![TEST-MOTO-QUANTITY-CLASS.md.png](https://b.radikal.host/2023/01/20/TEST-MOTO-QUANTITY-CLASS.md.png)](https://radikal.host/i/JBCd8D)
[![TRAIN-CAR-QUANTITY-CLASS.md.png](https://d.radikal.host/2023/01/20/TRAIN-CAR-QUANTITY-CLASS.md.png)](https://radikal.host/i/JBCltr)
[![TRAIN-MOTO-QUANTITY-CLASS.md.png](https://d.radikal.host/2023/01/20/TRAIN-MOTO-QUANTITY-CLASS.md.png)](https://radikal.host/i/JBCoPQ)

## 3.3. Отслеживание состава датасета по количеству `angle` (ракурс):

[![TEST-CAR-QUANTITY-ANGLE.md.png](https://d.radikal.host/2023/01/20/TEST-CAR-QUANTITY-ANGLE.md.png)](https://radikal.host/i/JBCuxz)
[![TEST-MOTO-QUANTITY-ANGLE.md.png](https://d.radikal.host/2023/01/20/TEST-MOTO-QUANTITY-ANGLE.md.png)](https://radikal.host/i/JBCM9K)
[![TRAIN-CAR-QUANTITY-ANGLE.md.png](https://b.radikal.host/2023/01/20/TRAIN-CAR-QUANTITY-ANGLE.md.png)](https://radikal.host/i/JBC9id)
[![TRAIN-MOTO-QUANTITY-ANGLE.md.png](https://b.radikal.host/2023/01/20/TRAIN-MOTO-QUANTITY-ANGLE.md.png)](https://radikal.host/i/JBCLl8)

# 4. ХАРАКТЕРИСТИКИ ДАТАСЕТА.
* Количество фото:
    * `test`:
        * *car*: 511
        * *moto* 522
    * `train`:
        * *car*: 508
        * *moto*: 512
    * `test` + `train` = 2053
    
* Минимальный размер фото: 500 x 500
* Формат фото: ***.jpg****
