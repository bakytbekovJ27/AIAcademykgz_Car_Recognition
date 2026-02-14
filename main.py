import sys
import json
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QScrollArea, QSplitter, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QFont
from PIL import Image
import numpy as np

from utils.core import detect_and_crop_cars, predict_car_brand, draw_box_and_label


class ProcessingThread(QThread):
    """Отдельный поток для обработки изображения"""
    finished = pyqtSignal(object, list, list, list)
    error = pyqtSignal(str)
    
    def __init__(self, image_path, id_to_class):
        super().__init__()
        self.image_path = image_path
        self.id_to_class = id_to_class
        
    def run(self):
        try:
            # Загрузка изображения
            input_image = Image.open(self.image_path).convert("RGB")
            
            # Детекция автомобилей
            cars_bbox, cars_image = detect_and_crop_cars(input_image)
            
            if not cars_image:
                self.error.emit("На изображении не обнаружено автомобилей")
                return
            
            # Классификация
            predicted_brands, predicted_probs = predict_car_brand(cars_image, self.id_to_class)
            
            # Отрисовка результатов
            result_image = draw_box_and_label(input_image, cars_bbox, predicted_brands, predicted_probs)
            result_image = Image.fromarray(result_image)
            
            self.finished.emit(result_image, cars_bbox, predicted_brands, predicted_probs)
            
        except Exception as e:
            self.error.emit(f"Ошибка при обработке: {str(e)}")


class MacOSStyleWidget(QWidget):
    """Базовый виджет в стиле macOS"""
    def __init__(self):
        super().__init__()
        self.setup_style()
        
    def setup_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
            }
        """)


class CarRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.result_image = None
        self.processing_thread = None
        
        # Загрузка маппинга классов
        self.load_class_mapping()
        
        # Настройка UI
        self.init_ui()
        
    def load_class_mapping(self):
        """Загрузка маппинга классов из JSON"""
        try:
            with open('./utils/class_id.json', 'r') as f:
                mapping = json.load(f)
                self.id_to_class = mapping['id_to_class']
        except FileNotFoundError:
            QMessageBox.critical(self, "Ошибка", "Файл class_id.json не найден!")
            sys.exit(1)
            
    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("Car Brand Recognition")
        self.setGeometry(100, 100, 1200, 800)
        
        # Применение стиля macOS
        self.setup_macos_style()
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Заголовок
        title_label = QLabel("🚗 Car Brand Recognition")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: 600;
            color: #ffffff;
            padding: 20px;
        """)
        main_layout.addWidget(title_label)
        
        # Кнопки управления
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.load_btn = self.create_button("Загрузить изображение", "#007AFF")
        self.load_btn.clicked.connect(self.load_image)
        
        self.process_btn = self.create_button("Распознать автомобили", "#34C759")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        
        self.save_btn = self.create_button("Сохранить результат", "#FF9500")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(button_layout)
        
        # Splitter для изображений
        splitter = QSplitter(Qt.Horizontal)
        
        # Исходное изображение
        self.original_scroll = self.create_image_container("Исходное изображение")
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: #2a2a2a; border-radius: 8px;")
        self.original_scroll.widget().layout().addWidget(self.original_label)
        
        # Обработанное изображение
        self.result_scroll = self.create_image_container("Результат распознавания")
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background-color: #2a2a2a; border-radius: 8px;")
        self.result_scroll.widget().layout().addWidget(self.result_label)
        
        splitter.addWidget(self.original_scroll)
        splitter.addWidget(self.result_scroll)
        splitter.setSizes([600, 600])
        
        main_layout.addWidget(splitter, 1)
        
        # Информационная панель
        self.info_label = QLabel("Загрузите изображение для начала работы")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            font-size: 14px;
            color: #a0a0a0;
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 8px;
        """)
        main_layout.addWidget(self.info_label)
        
    def setup_macos_style(self):
        """Настройка стиля в стиле macOS"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
            }
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
            QSplitter::handle {
                background-color: #3a3a3a;
                width: 1px;
            }
        """)
        
    def create_button(self, text, color):
        """Создание кнопки в стиле macOS"""
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {self.adjust_color(color, 1.1)};
            }}
            QPushButton:pressed {{
                background-color: {self.adjust_color(color, 0.9)};
            }}
            QPushButton:disabled {{
                background-color: #3a3a3a;
                color: #666666;
            }}
        """)
        btn.setCursor(Qt.PointingHandCursor)
        return btn
        
    def adjust_color(self, hex_color, factor):
        """Корректировка яркости цвета"""
        color = QColor(hex_color)
        h, s, v, a = color.getHsv()
        v = min(255, int(v * factor))
        color.setHsv(h, s, v, a)
        return color.name()
        
    def create_image_container(self, title):
        """Создание контейнера для изображения"""
        container = QScrollArea()
        container.setWidgetResizable(True)
        container.setStyleSheet("""
            QScrollArea {
                background-color: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 10px;
            }
        """)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: 600;
            color: #ffffff;
            padding: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        container.setWidget(content)
        return container
        
    def load_image(self):
        """Загрузка изображения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.current_image_path = file_path
            
            # Отображение исходного изображения
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.original_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.original_label.setPixmap(scaled_pixmap)
            
            # Активация кнопки обработки
            self.process_btn.setEnabled(True)
            self.info_label.setText(f"Изображение загружено: {os.path.basename(file_path)}")
            
            # Очистка результата
            self.result_label.clear()
            self.result_image = None
            self.save_btn.setEnabled(False)
            
    def process_image(self):
        """Обработка изображения"""
        if not self.current_image_path:
            return
            
        # Проверка наличия моделей
        if not (os.path.exists('models/model_ft.pt') and os.path.exists('models/model_tl.pt')):
            QMessageBox.warning(
                self,
                "Модели не найдены",
                "Файлы моделей не найдены в папке 'models/'.\n"
                "Пожалуйста, обучите модели или поместите их в папку 'models/'."
            )
            return
        
        # Отключение кнопок
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.info_label.setText("Обработка изображения... Пожалуйста, подождите.")
        
        # Запуск обработки в отдельном потоке
        self.processing_thread = ProcessingThread(self.current_image_path, self.id_to_class)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()
        
    def on_processing_finished(self, result_image, cars_bbox, predicted_brands, predicted_probs):
        """Обработка завершена успешно"""
        self.result_image = result_image
        
        # Конвертация PIL Image в QPixmap
        result_array = np.array(result_image)
        height, width, channel = result_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(result_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = pixmap.scaled(
            self.result_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.result_label.setPixmap(scaled_pixmap)
        
        # Формирование информации о результатах
        info_text = f"Обнаружено автомобилей: {len(cars_bbox)}\n"
        for i, (brand, prob) in enumerate(zip(predicted_brands, predicted_probs), 1):
            info_text += f"{i}. {brand} ({prob*100:.1f}%)\n"
        
        self.info_label.setText(info_text)
        
        # Активация кнопок
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
    def on_processing_error(self, error_message):
        """Обработка ошибки"""
        QMessageBox.critical(self, "Ошибка", error_message)
        self.info_label.setText(error_message)
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
    def save_result(self):
        """Сохранение результата"""
        if not self.result_image:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            "detected_" + os.path.basename(self.current_image_path),
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            self.result_image.save(file_path)
            self.info_label.setText(f"Результат сохранен: {os.path.basename(file_path)}")
            

def main():
    app = QApplication(sys.argv)
    
    # Установка шрифта для всего приложения
    font = QFont("-apple-system")
    font.setPointSize(13)
    app.setFont(font)
    
    window = CarRecognitionApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()