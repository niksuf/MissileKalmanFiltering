# Фильтр Калмана для трех баллистических ракет

## Описание проекта
Приложение моделирует работу фильтра Калмана для трех реактивных баллистических ракет. В основе алгоритма лежат основные уравнения фильтра Калмана для расчета оценок
параметров движения реактивных снарядов.

Фильтр Калмана – это рекуррентный байесовский алгоритм оптимальной линейной фильтрации векторного случайного процесса, применение которого базируется на нескольких 
допущениях.

Приложение настроено на три баллистических ракеты, начальные условия которых можно изменить в коде приложения. В будущем планируется сделать удобный графический 
интерфейс.

В результате работы приложения, выводятся следующие графики и значения:
* Дальность по X;
* Дальность по Y;
* Скорость по X;
* Скорость по Y;
* Разница значений дальности по X;
* Разница значений дальности по Y;
* Разница значений скорости по X;
* Разница значений скорости по Y;
* Дальность в конечной точке (значение в консоль);
* Скорость в конечной точке (значение в консоль).

Все графики и значения приведены в трех экземплярах (для каждой ракеты отдельно).

## Инструкция по запуску
1. Скачать репозиторий к себе на компьютер при помощи команды:
```
git clone https://github.com/niksuf/MissileKalmanFiltering
```
2. Запустить приложение командой:
```
python main.py
```
