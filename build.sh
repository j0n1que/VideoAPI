#!/bin/bash

# Создаем директорию для сборки, если она не существует
mkdir -p build
cd build

# Указываем полный путь к исходной директории с CMakeLists.txt
cmake -G "Visual Studio 17 2022" ..

# Собираем проект в конфигурации Release
cmake --build . --config Release

# Копируем результат сборки
cp Release/libinfer.dll ../
