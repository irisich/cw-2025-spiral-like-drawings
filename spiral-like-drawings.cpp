#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Функция для генерации изображения спирали из входного изображения
void GenerateSpiralImage(const std::string& input_path,
                         const std::string& output_path,
                         cv::Size canvas_size = cv::Size(800, 800),
                         double max_thick = 8.8,
                         double min_thick = 0.8,
                         int spacing = 10) {
    // Загрузка изображения с диска
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        throw std::runtime_error("Не удалось загрузить изображение: " + input_path);
    }

    // Преобразование изображения в оттенки серого
    cv::Mat gray;
    if (img.channels() == 1) {
        gray = img.clone();  // изображение уже в оттенках серого
    } else {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);  // конвертация в оттенки серого
    }

    // Определение максимального радиуса спирали
    int max_radius = std::min(canvas_size.width, canvas_size.height) / 2;
    // Вычисление количества витков спирали
    int turns = max_radius / spacing - 1;

    // Создание пустого холста для рисования спирали
    cv::Mat canvas = cv::Mat::ones(canvas_size.height, canvas_size.width, CV_8UC1) * 255;

    // Масштабирование изображения под размер спирали
    int spiral_size = 2 * (turns + 1) * spacing;
    cv::Mat gray_scaled;
    cv::resize(gray, gray_scaled, cv::Size(spiral_size, spiral_size));

    // Определение центра холста
    int center_x = canvas_size.width / 2;
    int center_y = canvas_size.height / 2;

    // Начальная позиция для рисования спирали
    double x = 0.0, y = 0.0;
    int last_x = center_x, last_y = center_y;

    // Основной цикл для рисования спирали
    for (int n = 1; n <= turns; ++n) {
        int s = spacing;
        // Определение точек текущего витка
        std::vector<cv::Point2d> points = {
            cv::Point2d(n * s, -n * s),
            cv::Point2d(n * s, n * s),
            cv::Point2d(-n * s, n * s),
            cv::Point2d(-n * s, -(n + 1) * s)
        };

        // Обработка каждого сегмента витка
        for (const auto& point : points) {
            double dx = point.x;
            double dy = point.y;

            // Вычисление длины сегмента
            double seg_length = std::sqrt((dx - x) * (dx - x) + (dy - y) * (dy - y));
            int steps = std::max(1, static_cast<int>(seg_length));

            // Интерполяция вдоль сегмента
            for (int i = 0; i <= steps; ++i) {
                double t = static_cast<double>(i) / steps;
                double px = x + (dx - x) * t;
                double py = y + (dy - y) * t;

                // Преобразование координат для рисования на холсте
                int canvas_x = static_cast<int>(center_x + px);
                int canvas_y = static_cast<int>(center_y + py);

                // Пропуск точек, выходящих за границы холста
                if (canvas_x < 0 || canvas_x >= canvas_size.width ||
                    canvas_y < 0 || canvas_y >= canvas_size.height) {
                    continue;
                }

                // Получение яркости пикселя из изображения
                int img_x = static_cast<int>((px + (turns + 1) * s) *
                    (spiral_size / (2.0 * (turns + 1) * s)));
                int img_y = static_cast<int>((py + (turns + 1) * s) *
                    (spiral_size / (2.0 * (turns + 1) * s)));

                // Коррекция координат, выходящих за границы изображения
                img_x = std::max(0, std::min(gray_scaled.cols - 1, img_x));
                img_y = std::max(0, std::min(gray_scaled.rows - 1, img_y));

                // Получение яркости и расчет толщины линии
                uchar brightness = gray_scaled.at<uchar>(img_y, img_x);
                double thickness = min_thick + (1.0 - brightness / 255.0) * (max_thick - min_thick);
                thickness = std::max(1.0, std::min(20.0, thickness));

                // Рисование линии на холсте
                if (i == 0) {
                    last_x = canvas_x;
                    last_y = canvas_y;
                }
                if (last_x != canvas_x || last_y != canvas_y) {
                    cv::line(canvas,
                             cv::Point(last_x, last_y),
                             cv::Point(canvas_x, canvas_y),
                             cv::Scalar(0, 0, 0),
                             static_cast<int>(thickness),
                             cv::LINE_AA);
                }
                last_x = canvas_x;
                last_y = canvas_y;
            }
            x = dx;
            y = dy;
        }
    }
    // Сохранение результата на диск
    cv::imwrite(output_path, canvas);
}

// Функция для генерации изображения треугольной спирали
void GenerateTriangularSpiralImage(const std::string& input_path,
                                   const std::string& output_path,
                                   cv::Size canvas_size = cv::Size(800, 800),
                                   double max_thick = 8.8,
                                   double min_thick = 0.8,
                                   int spacing = 10,
                                   double image_scale_factor = 1.5) {
    // Загрузка изображения с диска
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        throw std::runtime_error("Не удалось загрузить изображение: " + input_path);
    }

    // Преобразование изображения в оттенки серого
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Определение максимального радиуса для треугольников
    int max_radius = std::min(canvas_size.width, canvas_size.height) / 2;
    // Вычисление максимального количества слоев треугольников
    int max_layers = max_radius / spacing;

    // Создание пустого холста для рисования
    cv::Mat canvas = cv::Mat::ones(canvas_size.height, canvas_size.width, CV_8UC1) * 255;

    // Масштабирование изображения под размер треугольника
    double triangle_height = max_radius * std::sqrt(3) / 2; // высота равностороннего треугольника
    int effective_size = static_cast<int>(triangle_height * 2 * image_scale_factor);
    cv::Mat gray_scaled;
    cv::resize(gray, gray_scaled, cv::Size(effective_size, effective_size));

    // Определение центра холста
    int center_x = canvas_size.width / 2;
    int center_y = canvas_size.height / 2;

    // Генерация точек для концентрических треугольников
    std::vector<double> point_list_x;
    std::vector<double> point_list_y;

    // Углы вершин треугольника: 90° (вверх), 210° (вниз-влево), 330° (вниз-вправо)
    std::vector<double> triangle_angles = {90, 210, 330};

    // Строим концентрические треугольники
    for (int layer = 1; layer <= max_layers; ++layer) {
        double radius = layer * spacing;
        // Вычисляем координаты вершин треугольника
        std::vector<cv::Point2d> vertices;
        for (double angle : triangle_angles) {
            double x = radius * std::cos(angle * M_PI / 180.0);
            double y = radius * std::sin(angle * M_PI / 180.0);
            vertices.push_back(cv::Point2d(x, y));
        }
        // Для каждой стороны треугольника
        for (int side = 0; side < 3; ++side) {
            cv::Point2d start = vertices[side];
            cv::Point2d end = vertices[(side + 1) % 3];
            // Количество точек на стороне пропорционально длине стороны
            double side_length = std::sqrt(std::pow(end.x - start.x, 2) + std::pow(end.y - start.y, 2));
            int points_on_side = std::max(1, static_cast<int>(side_length / 5));
            // Добавляем точки вдоль стороны
            for (int p = 0; p < points_on_side; ++p) {
                double t = static_cast<double>(p) / points_on_side;
                double x = start.x + (end.x - start.x) * t;
                double y = start.y + (end.y - start.y) * t;
                point_list_x.push_back(x);
                point_list_y.push_back(y);
            }
        }
    }

    // Рисование треугольников на холсте
    for (size_t i = 0; i < point_list_x.size() - 1; ++i) {
        // Текущая и следующая точки
        double x1 = point_list_x[i], y1 = point_list_y[i];
        double x2 = point_list_x[i + 1], y2 = point_list_y[i + 1];
        // Преобразование координат для холста
        int canvas_x1 = static_cast<int>(center_x + x1);
        int canvas_y1 = static_cast<int>(center_y - y1);
        int canvas_x2 = static_cast<int>(center_x + x2);
        int canvas_y2 = static_cast<int>(center_y - y2);
        // Проверка границ холста
        if (canvas_x1 >= 0 && canvas_x1 < canvas_size.width && canvas_y1 >= 0 && canvas_y1 < canvas_size.height &&
            canvas_x2 >= 0 && canvas_x2 < canvas_size.width && canvas_y2 >= 0 && canvas_y2 < canvas_size.height) {
            // Вычисляем расстояние между точками
            double distance = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            // Если точки слишком далеко друг от друга, интерполируем
            if (distance > 2.0) {
                int steps = static_cast<int>(distance / 2.0);
                for (int step = 0; step <= steps; ++step) {
                    double t = static_cast<double>(step) / steps;
                    // Интерполированные координаты
                    double interp_x = x1 + (x2 - x1) * t;
                    double interp_y = y1 + (y2 - y1) * t;
                    // Нормализация координат относительно размера треугольника
                    double normalized_x = (interp_x + max_radius) / (2.0 * max_radius);
                    double normalized_y = (interp_y + max_radius) / (2.0 * max_radius);
                    // Масштабирование для получения координат в изображении
                    int img_x = static_cast<int>(normalized_x * gray_scaled.cols);
                    int img_y = static_cast<int>((1.0 - normalized_y) * gray_scaled.rows);
                    // Коррекция выхода за границы
                    img_x = std::max(0, std::min(gray_scaled.cols - 1, img_x));
                    img_y = std::max(0, std::min(gray_scaled.rows - 1, img_y));
                    // Получение яркости и расчет толщины
                    uchar brightness = gray_scaled.at<uchar>(img_y, img_x);
                    double thickness = min_thick + (1.0 - brightness / 255.0) * (max_thick - min_thick);
                    int thickness_int = std::max(1, std::min(20, static_cast<int>(thickness)));
                    // Рисование точки
                    int step_canvas_x = static_cast<int>(center_x + interp_x);
                    int step_canvas_y = static_cast<int>(center_y - interp_y);
                    if (step_canvas_x >= 0 && step_canvas_x < canvas_size.width &&
                        step_canvas_y >= 0 && step_canvas_y < canvas_size.height) {
                        cv::circle(canvas,
                                   cv::Point(step_canvas_x, step_canvas_y),
                                   thickness_int / 2,
                                   cv::Scalar(0),
                                   -1);
                    }
                }
            } else {
                // Для близких точек
                double mid_x = (x1 + x2) / 2.0;
                double mid_y = (y1 + y2) / 2.0;
                // Нормализация координат
                double normalized_x = (mid_x + max_radius) / (2.0 * max_radius);
                double normalized_y = (mid_y + max_radius) / (2.0 * max_radius);
                int img_x = static_cast<int>(normalized_x * gray_scaled.cols);
                int img_y = static_cast<int>((1.0 - normalized_y) * gray_scaled.rows);
                // Коррекция выхода за границы
                img_x = std::max(0, std::min(gray_scaled.cols - 1, img_x));
                img_y = std::max(0, std::min(gray_scaled.rows - 1, img_y));
                // Получение яркости и расчет толщины
                uchar brightness = gray_scaled.at<uchar>(img_y, img_x);
                double thickness = min_thick + (1.0 - brightness / 255.0) * (max_thick - min_thick);
                int thickness_int = std::max(1, std::min(20, static_cast<int>(thickness)));
                cv::line(canvas,
                         cv::Point(canvas_x1, canvas_y1),
                         cv::Point(canvas_x2, canvas_y2),
                         cv::Scalar(0),
                         thickness_int,
                         cv::LINE_AA);
            }
        }
    }
    // Сохранение результата на диск
    cv::imwrite(output_path, canvas);
}

// Функция для генерации изображения шестиугольной спирали
void GenerateHexagonalSpiralImage(const std::string& input_path,
                                  const std::string& output_path,
                                  cv::Size canvas_size = cv::Size(800, 800),
                                  double max_thick = 8.8,
                                  double min_thick = 0.8,
                                  int spacing = 10) {
    // Загрузка изображения с диска
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        throw std::runtime_error("Не удалось загрузить изображение: " + input_path);
    }

    // Преобразование изображения в оттенки серого
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Определение максимального радиуса для шестиугольников
    int max_radius = std::min(canvas_size.width, canvas_size.height) / 2;
    // Вычисление максимального количества слоев шестиугольника
    int max_layers = max_radius / spacing;
    // Каждый слой имеет 6 сторон
    int num_segments = max_layers * 6;

    // Создание пустого холста для рисования
    cv::Mat canvas = cv::Mat::ones(canvas_size.height, canvas_size.width, CV_8UC1) * 255;

    // Масштабирование изображения под размер спирали
    int spiral_size = max_radius * 2;
    cv::Mat gray_scaled;
    cv::resize(gray, gray_scaled, cv::Size(spiral_size, spiral_size));

    // Определение центра холста
    int center_x = canvas_size.width / 2;
    int center_y = canvas_size.height / 2;

    // Генерация точек спирали для концентрических шестиугольников
    std::vector<double> point_list_x = {0.0}; // начальная точка (центр)
    std::vector<double> point_list_y = {0.0};

    // Углы для шестиугольника: 0°, 60°, 120°, 180°, 240°, 300°
    std::vector<double> angles = {0, 60, 120, 180, 240, 300};

    // Строим концентрические шестиугольники
    for (int layer = 1; layer <= max_layers; ++layer) {
        double radius = layer * spacing;
        // Для каждого слоя строим 6 сторон шестиугольника
        for (int side = 0; side < 6; ++side) {
            // Количество точек на стороне пропорционально радиусу
            int points_on_side = std::max(1, layer);
            double start_angle = angles[side] * M_PI / 180.0;
            double end_angle = angles[(side + 1) % 6] * M_PI / 180.0;
            // Начальная и конечная точки стороны
            double start_x = radius * std::cos(start_angle);
            double start_y = radius * std::sin(start_angle);
            double end_x = radius * std::cos(end_angle);
            double end_y = radius * std::sin(end_angle);
            // Добавляем точки вдоль стороны
            for (int p = 0; p < points_on_side; ++p) {
                double t = static_cast<double>(p) / points_on_side;
                double x = start_x + (end_x - start_x) * t;
                double y = start_y + (end_y - start_y) * t;
                point_list_x.push_back(x);
                point_list_y.push_back(y);
            }
        }
    }

    // Рисование спирали на холсте
    for (size_t i = 0; i < point_list_x.size() - 1; ++i) {
        // Текущая и следующая точки
        double x1 = point_list_x[i], y1 = point_list_y[i];
        double x2 = point_list_x[i + 1], y2 = point_list_y[i + 1];
        // Преобразование координат для холста
        int canvas_x1 = static_cast<int>(center_x + x1);
        int canvas_y1 = static_cast<int>(center_y - y1); // инвертируем Y для корректного отображения
        int canvas_x2 = static_cast<int>(center_x + x2);
        int canvas_y2 = static_cast<int>(center_y - y2);
        // Проверка границ холста
        if (canvas_x1 >= 0 && canvas_x1 < canvas_size.width && canvas_y1 >= 0 && canvas_y1 < canvas_size.height &&
            canvas_x2 >= 0 && canvas_x2 < canvas_size.width && canvas_y2 >= 0 && canvas_y2 < canvas_size.height) {
            // Интерполяция между точками для получения яркости
            double seg_length = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            int steps = std::max(1, static_cast<int>(seg_length / 2));
            for (int step = 0; step < steps; ++step) {
                double t = (steps > 0) ? static_cast<double>(step) / steps : 0.0;
                // Интерполированные координаты
                double interp_x = x1 + (x2 - x1) * t;
                double interp_y = y1 + (y2 - y1) * t;
                // Получение яркости из изображения
                int img_x = static_cast<int>((interp_x + spiral_size / 2.0) * (gray_scaled.cols / static_cast<double>(spiral_size)));
                int img_y = static_cast<int>((-interp_y + spiral_size / 2.0) * (gray_scaled.rows / static_cast<double>(spiral_size)));
                // Коррекция выхода за границы
                img_x = std::max(0, std::min(gray_scaled.cols - 1, img_x));
                img_y = std::max(0, std::min(gray_scaled.rows - 1, img_y));
                // Получение яркости и расчет толщины
                uchar brightness = gray_scaled.at<uchar>(img_y, img_x);
                double thickness = min_thick + (1.0 - brightness / 255.0) * (max_thick - min_thick);
                int thickness_int = std::max(1, std::min(20, static_cast<int>(thickness)));
                // Рисование сегмента линии
                int step_canvas_x1 = static_cast<int>(center_x + interp_x);
                int step_canvas_y1 = static_cast<int>(center_y - interp_y);
                int step_canvas_x2, step_canvas_y2;
                if (step < steps - 1) {
                    double next_t = static_cast<double>(step + 1) / steps;
                    double next_interp_x = x1 + (x2 - x1) * next_t;
                    double next_interp_y = y1 + (y2 - y1) * next_t;
                    step_canvas_x2 = static_cast<int>(center_x + next_interp_x);
                    step_canvas_y2 = static_cast<int>(center_y - next_interp_y);
                } else {
                    step_canvas_x2 = canvas_x2;
                    step_canvas_y2 = canvas_y2;
                }
                // Проверка границ для интерполированных точек
                if (step_canvas_x1 >= 0 && step_canvas_x1 < canvas_size.width &&
                    step_canvas_y1 >= 0 && step_canvas_y1 < canvas_size.height &&
                    step_canvas_x2 >= 0 && step_canvas_x2 < canvas_size.width &&
                    step_canvas_y2 >= 0 && step_canvas_y2 < canvas_size.height) {
                    cv::line(canvas,
                             cv::Point(step_canvas_x1, step_canvas_y1),
                             cv::Point(step_canvas_x2, step_canvas_y2),
                             cv::Scalar(0, 0, 0),
                             thickness_int,
                             cv::LINE_AA);
                }
            }
        }
    }
    // Сохранение результата на диск
    cv::imwrite(output_path, canvas);
}


int main(int argc, char* argv[]) {
    try {
        // Проверка минимального количества аргументов
        if (argc < 4) {
            std::cerr << "Использование: " << argv[0] << " <флаг> <входной_путь> <выходной_путь> [размер_холста] [макс_толщина] [мин_толщина] [интервал]" << std::endl;
            return -1;
        }

        std::string flag = argv[1];
        std::string input_path = argv[2];
        std::string output_path = argv[3];

        // Значения по умолчанию
        cv::Size canvas_size(800, 800);
        double max_thick = 8.8;
        double min_thick = 0.8;
        int spacing = 10;
        double image_scale_factor = 1.5;

        // Обработка дополнительных аргументов
        if (argc >= 5) {
            int size = std::stoi(argv[4]);
            canvas_size = cv::Size(size, size);
        }
        if (argc >= 6) {
            max_thick = std::stod(argv[5]);
        }
        if (argc >= 7) {
            min_thick = std::stod(argv[6]);
        }
        if (argc >= 8) {
            spacing = std::stoi(argv[7]);
        }

        // Вызов соответствующей функции в зависимости от флага
        if (flag == "--square") {
            GenerateSpiralImage(input_path, output_path, canvas_size, max_thick, min_thick, spacing);
        } else if (flag == "--triangle") {
            GenerateTriangularSpiralImage(input_path, output_path, canvas_size, max_thick, min_thick, spacing, image_scale_factor);
        } else if (flag == "--hexagon") {
            GenerateHexagonalSpiralImage(input_path, output_path, canvas_size, max_thick, min_thick, spacing);
        } else {
            std::cerr << "Неверный флаг. Используйте --square, --triangle или --hexagon." << std::endl;
            return -1;
        }

        std::cout << "Спираль успешно создана!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}