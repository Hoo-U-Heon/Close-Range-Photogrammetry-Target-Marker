import cv2
import numpy as np
import os
import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QLineEdit, QStatusBar, QFileDialog,
    QMessageBox, QSlider, QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsTextItem, QFrame,
    QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QTransform, QIntValidator, QPainter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    get_histogram_threshold, roi_binary_mask, calculate_centroid,
    write_to_txt, init_txt, get_unique_save_path
)


class ZoomableGraphicsView(QGraphicsView):
    zoom_signal = Signal(float, QPointF)
    mouse_pos_signal = Signal(int, int)
    roi_finished_signal = Signal(QPointF, QPointF)
    mask_add_signal = Signal(QPointF, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.NoDrag)
        self._zoom = 1.0
        self._pan = False
        self._pan_start = QPointF()
        self._roi_start = None
        self._is_drawing_roi = False
        self._roi_item = None
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom *= 1.15
            else:
                self._zoom /= 1.15
            self._zoom = max(0.05, min(20.0, self._zoom))
            scene_pos = self.mapToScene(event.position().toPoint())
            self.zoom_signal.emit(self._zoom, scene_pos)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            self._is_drawing_roi = True
            self._roi_start = self.mapToScene(event.pos())
            if self._roi_item:
                self.scene().removeItem(self._roi_item)
            self._roi_item = QGraphicsRectItem(QRectF(self._roi_start, self._roi_start))
            self._roi_item.setPen(QPen(QColor(0, 255, 255), 3))
            self._roi_item.setBrush(QColor(0, 255, 255, 30))
            self.scene().addItem(self._roi_item)
        elif event.button() == Qt.RightButton:
            pos = self.mapToScene(event.pos())
            shift_held = event.modifiers() & Qt.ShiftModifier
            self.mask_add_signal.emit(pos, not shift_held)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        elif self._is_drawing_roi and self._roi_item:
            current = self.mapToScene(event.pos())
            rect = QRectF(self._roi_start, current).normalized()
            self._roi_item.setRect(rect)
            self.mouse_pos_signal.emit(int(rect.center().x()), int(rect.center().y()))
        else:
            pos = self.mapToScene(event.pos())
            self.mouse_pos_signal.emit(int(pos.x()), int(pos.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton and self._is_drawing_roi:
            self._is_drawing_roi = False
            if self._roi_start:
                current = self.mapToScene(event.pos())
                rect = QRectF(self._roi_start, current).normalized()
                if rect.width() > 5 and rect.height() > 5:
                    self.roi_finished_signal.emit(rect.topLeft(), rect.bottomRight())
                else:
                    if self._roi_item:
                        self.scene().removeItem(self._roi_item)
                        self._roi_item = None
        super().mouseReleaseEvent(event)

    def set_zoom(self, zoom_factor, pivot=None):
        self._zoom = zoom_factor
        if pivot is None:
            pivot = self.sceneRect().center()
        self.setTransform(QTransform().scale(zoom_factor, zoom_factor))
        self.centerOn(pivot)

    def get_zoom(self):
        return self._zoom


class ControlPointLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("控制点标注工具")
        
        screen_geometry = QApplication.instance().primaryScreen().geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        min_width = 1024
        min_height = 600
        
        window_width = max(window_width, min_width)
        window_height = max(window_height, min_height)
        
        self.resize(window_width, window_height)
        self.setMinimumSize(min_width, min_height)
        
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.img_path = ""
        self.img_original = None
        self.img_gray = None
        self.h, self.w = 0, 0
        self.output_dir = ""
        
        self.labeled_points = []
        self.current_mask = None
        self.mask_history = []
        self.max_history = 50
        
        self.save_path_txt = ""
        self.save_path_img = ""

        self.zoom_level = 0.5
        self.scene = None
        self.pixmap_item = None
        self.mask_pixmap_item = None
        self.graphics_view = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setMinimumWidth(280)
        self.scroll_area.setMaximumWidth(360)
        
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setSpacing(15)
        self.control_layout.setContentsMargins(20, 20, 20, 20)
        
        self.scroll_area.setWidget(self.control_panel)
        
        main_layout.addWidget(self.scroll_area, 1)
        main_layout.addWidget(self._create_graphics_area(), 4)

        title_label = QLabel("控制点标注工具")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #64C8FF;")
        self.control_layout.addWidget(title_label)

        input_group = QFrame()
        input_group.setStyleSheet("background-color: #1D1D1D; border-radius: 8px; padding: 15px;")
        input_layout = QVBoxLayout(input_group)
        
        input_layout.addWidget(QLabel("📁 输入图像"))
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("请选择输入图像")
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setStyleSheet("background-color: #2D2D2D; border: 1px solid #3D3D3D; padding: 6px;")
        input_layout.addWidget(self.image_path_edit)
        
        self.select_image_btn = QPushButton("选择图像")
        self.select_image_btn.clicked.connect(self.select_input_image)
        self.select_image_btn.setStyleSheet("background-color: #3D5A80; color: white; padding: 6px 12px; border-radius: 4px;")
        input_layout.addWidget(self.select_image_btn)

        input_layout.addWidget(QLabel("📂 输出目录"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("请选择输出目录")
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_edit.setStyleSheet("background-color: #2D2D2D; border: 1px solid #3D3D3D; padding: 6px;")
        input_layout.addWidget(self.output_dir_edit)
        
        self.select_output_btn = QPushButton("选择目录")
        self.select_output_btn.clicked.connect(self.select_output_directory)
        self.select_output_btn.setStyleSheet("background-color: #3D5A80; color: white; padding: 6px 12px; border-radius: 4px;")
        input_layout.addWidget(self.select_output_btn)
        self.control_layout.addWidget(input_group)

        status_group = QFrame()
        status_group.setStyleSheet("background-color: #1D1D1D; border-radius: 8px; padding: 15px;")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("状态: 等待输入")
        self.status_label.setStyleSheet("color: #64FF64;")
        status_layout.addWidget(self.status_label)
        
        self.image_info_label = QLabel("图片信息: 未加载")
        status_layout.addWidget(self.image_info_label)
        
        self.mouse_label = QLabel("鼠标: (0, 0)")
        status_layout.addWidget(self.mouse_label)
        self.control_layout.addWidget(status_group)

        zoom_group = QFrame()
        zoom_group.setStyleSheet("background-color: #1D1D1D; border-radius: 8px; padding: 15px;")
        zoom_layout = QVBoxLayout(zoom_group)
        
        zoom_layout.addWidget(QLabel("🔍 缩放控制"))
        slider_layout = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(5)
        self.zoom_slider.setMaximum(500)
        self.zoom_slider.setValue(50)
        self.zoom_slider.setFixedWidth(200)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        slider_layout.addWidget(self.zoom_slider)
        self.zoom_value_label = QLabel("50%")
        slider_layout.addWidget(self.zoom_value_label)
        zoom_layout.addLayout(slider_layout)
        self.control_layout.addWidget(zoom_group)

        help_group = QFrame()
        help_group.setStyleSheet("background-color: #1D1D1D; border-radius: 8px; padding: 15px;")
        help_layout = QVBoxLayout(help_group)
        
        help_layout.addWidget(QLabel("📖 操作说明"))
        help_text = QLabel(
            "• 左键拖拽: 标注矩形框(ROI)\n"
            "• 右键点击: 添加掩膜\n"
            "• Shift+右键: 移除掩膜\n"
            "• Ctrl+滚轮: 缩放图像\n"
            "• 中键拖拽: 平移图像"
        )
        help_text.setStyleSheet("color: #AAAAAA; font-size: 12px;")
        help_layout.addWidget(help_text)
        self.control_layout.addWidget(help_group)

        point_group = QFrame()
        point_group.setStyleSheet("background-color: #1D1D1D; border-radius: 8px; padding: 15px;")
        point_layout = QVBoxLayout(point_group)
        
        point_layout.addWidget(QLabel("📍 控制点编号"))
        self.point_id_input = QLineEdit("1")
        self.point_id_input.setFixedWidth(100)
        self.point_id_input.setValidator(QIntValidator(1, 99999))
        self.point_id_input.setStyleSheet("background-color: #2D2D2D; border: 1px solid #3D3D3D; padding: 6px;")
        point_layout.addWidget(self.point_id_input)
        
        self.confirm_mask_btn = QPushButton("确认掩膜")
        self.confirm_mask_btn.clicked.connect(self.confirm_mask)
        self.confirm_mask_btn.setEnabled(False)
        self.confirm_mask_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 16px; border-radius: 4px;")
        point_layout.addWidget(self.confirm_mask_btn)
        
        self.confirm_point_btn = QPushButton("确认控制点")
        self.confirm_point_btn.clicked.connect(self.confirm_control_point)
        self.confirm_point_btn.setEnabled(False)
        self.confirm_point_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px 16px; border-radius: 4px;")
        point_layout.addWidget(self.confirm_point_btn)
        self.control_layout.addWidget(point_group)

        points_group = QFrame()
        points_group.setStyleSheet("background-color: #1D1D1D; border-radius: 8px; padding: 15px;")
        points_layout = QVBoxLayout(points_group)
        
        points_layout.addWidget(QLabel("已标注控制点"))
        self.points_list = QListWidget()
        self.points_list.setMaximumHeight(120)
        points_layout.addWidget(self.points_list)
        self.control_layout.addWidget(points_group)

        action_group = QFrame()
        action_group.setStyleSheet("background-color: #1D1D1D; border-radius: 8px; padding: 15px;")
        action_layout = QVBoxLayout(action_group)
        
        self.clear_mask_btn = QPushButton("清除掩膜")
        self.clear_mask_btn.clicked.connect(self.clear_mask)
        self.clear_mask_btn.setEnabled(False)
        self.clear_mask_btn.setStyleSheet("background-color: #757575; color: white; padding: 6px 12px; border-radius: 4px;")
        action_layout.addWidget(self.clear_mask_btn)
        
        self.undo_btn = QPushButton("撤销")
        self.undo_btn.clicked.connect(self.undo_mask)
        self.undo_btn.setEnabled(False)
        self.undo_btn.setStyleSheet("background-color: #757575; color: white; padding: 6px 12px; border-radius: 4px;")
        action_layout.addWidget(self.undo_btn)
        
        self.save_exit_btn = QPushButton("保存并退出")
        self.save_exit_btn.clicked.connect(self.save_and_exit)
        self.save_exit_btn.setEnabled(False)
        self.save_exit_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px 16px; border-radius: 4px;")
        action_layout.addWidget(self.save_exit_btn)
        
        self.quit_btn = QPushButton("放弃退出")
        self.quit_btn.clicked.connect(self.quit_labeling)
        self.quit_btn.setStyleSheet("background-color: #C86464; color: white; padding: 8px 16px; border-radius: 4px;")
        action_layout.addWidget(self.quit_btn)
        self.control_layout.addWidget(action_group)
        self.control_layout.addStretch()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("准备就绪 - 请选择输入图像")

        self.apply_styles()

    def _create_graphics_area(self):
        graphics_widget = QWidget()
        graphics_layout = QVBoxLayout(graphics_widget)
        graphics_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QColor(40, 40, 40))
        
        placeholder_label = QGraphicsTextItem("请先选择输入图像")
        placeholder_label.setDefaultTextColor(QColor(100, 100, 100))
        placeholder_label.setPos(100, 100)
        self.scene.addItem(placeholder_label)

        self.graphics_view = ZoomableGraphicsView()
        self.graphics_view.setScene(self.scene)
        self.graphics_view.zoom_signal.connect(self._on_zoom_changed)
        self.graphics_view.mouse_pos_signal.connect(self._on_mouse_moved)
        self.graphics_view.roi_finished_signal.connect(self._on_roi_finished)
        self.graphics_view.mask_add_signal.connect(self._on_mask_edit)
        
        self.graphics_view.setSizePolicy(
            self.graphics_view.sizePolicy().horizontalPolicy(),
            QSizePolicy.Expanding
        )
        
        graphics_layout.addWidget(self.graphics_view)
        return graphics_widget

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2D2D2D; }
            QWidget { color: #E0E0E0; background-color: #2D2D2D; }
            QPushButton { border: none; }
            QPushButton:hover { opacity: 0.9; }
            QPushButton:pressed { opacity: 0.8; }
            QPushButton:disabled { background-color: #444444 !important; }
            QListWidget { background-color: #2D2D2D; border: 1px solid #3D3D3D; border-radius: 4px; }
            QSlider::groove:horizontal { border: 1px solid #3D3D3D; height: 8px; background: #2D2D2D; border-radius: 4px; }
            QSlider::handle:horizontal { background: #64C8FF; width: 16px; margin: -4px 0; border-radius: 8px; }
            QLabel { color: #E0E0E0; }
            QStatusBar { background-color: #1D1D1D; color: #AAAAAA; }
        """)

    def select_input_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入图像",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if file_path:
            self.img_path = file_path
            self.image_path_edit.setText(os.path.basename(file_path))
            
            self.img_original = cv2.imread(file_path)
            if self.img_original is None:
                QMessageBox.warning(self, "错误", "无法读取图像文件")
                return
            
            self.img_gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
            self.h, self.w = self.img_gray.shape[:2]
            
            self.current_mask = np.zeros((self.h, self.w), dtype=np.uint8)
            self.mask_history = []
            
            self._load_image_to_scene()
            self.image_info_label.setText(f"图片信息: {self.w} x {self.h}")
            self.status_label.setText("状态: 图像已加载")
            self.status_bar.showMessage(f"已加载图像: {os.path.basename(file_path)}")
            
            self.update_buttons_state()

    def select_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录", "")
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_edit.setText(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            
            if self.img_path:
                self.save_path_txt = get_unique_save_path(self.img_path, dir_path, "_control_points", ".txt")
                self.save_path_img = get_unique_save_path(self.img_path, dir_path, "_labeled", ".jpg")
                init_txt(self.save_path_txt)
            
            self.status_label.setText("状态: 输出目录已设置")
            self.status_bar.showMessage(f"输出目录: {dir_path}")
            self.update_buttons_state()

    def _load_image_to_scene(self):
        self.scene.clear()
        self.scene.setBackgroundBrush(QColor(40, 40, 40))
        
        rgb_img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * self.w
        q_image = QImage(rgb_img.data, self.w, self.h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image))
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(0, 0, self.w, self.h)
        
        self.mask_pixmap_item = QGraphicsPixmapItem()
        self.mask_pixmap_item.setZValue(1)
        self.mask_pixmap_item.setOpacity(0.5)
        self.scene.addItem(self.mask_pixmap_item)
        
        self._fit_to_window()

    def _fit_to_window(self):
        view_size = self.graphics_view.viewport().size()
        scene_rect = self.scene.sceneRect()
        scale_w = (view_size.width() - 20) / scene_rect.width()
        scale_h = (view_size.height() - 20) / scene_rect.height()
        self.zoom_level = min(scale_w, scale_h)
        self.graphics_view.set_zoom(self.zoom_level)
        self.zoom_slider.setValue(int(self.zoom_level * 100))
        self.zoom_value_label.setText(f"{int(self.zoom_level * 100)}%")

    def _on_zoom_changed(self, zoom, pivot):
        self.zoom_level = zoom
        self.graphics_view.set_zoom(zoom, pivot)
        self.zoom_slider.setValue(int(zoom * 100))
        self.zoom_value_label.setText(f"{int(zoom * 100)}%")

    def _on_zoom_slider_changed(self, value):
        self.zoom_level = value / 100.0
        self.graphics_view.set_zoom(self.zoom_level)
        self.zoom_value_label.setText(f"{value}%")

    def _on_mouse_moved(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.mouse_label.setText(f"鼠标: ({x}, {y})")
        else:
            self.mouse_label.setText(f"鼠标: ({x}, {y}) [超出]")

    def _on_roi_finished(self, start, end):
        if not self.img_original is None:
            x1 = int(min(start.x(), end.x()))
            y1 = int(min(start.y(), end.y()))
            x2 = int(max(start.x(), end.x()))
            y2 = int(max(start.y(), end.y()))

            if x2 - x1 > 5 and y2 - y1 > 5:
                roi_rect = (x1, y1, x2, y2)
                threshold = get_histogram_threshold(self.img_gray, roi_rect)
                self.save_state()
                self.current_mask = roi_binary_mask(self.img_gray, roi_rect, threshold)
                component_area = np.sum(self.current_mask > 0)
                self.status_label.setText(f"状态: ROI已生成")
                self.status_bar.showMessage(f"ROI完成 - 阈值:{threshold:.0f} - 面积:{component_area}像素")
                self._update_mask_layer()
                self.update_buttons_state()

    def _on_mask_edit(self, pos, add):
        if not self.img_original is None:
            x = int(pos.x())
            y = int(pos.y())
            if 0 <= x < self.w and 0 <= y < self.h:
                self.save_state()
                radius = 3
                y1 = max(0, y - radius)
                y2 = min(self.h, y + radius)
                x1 = max(0, x - radius)
                x2 = min(self.w, x + radius)
                if add:
                    self.current_mask[y1:y2, x1:x2] = 255
                    self.status_label.setText("状态: 添加掩膜")
                else:
                    self.current_mask[y1:y2, x1:x2] = 0
                    self.status_label.setText("状态: 移除掩膜")
                self._update_mask_layer()
                self.update_buttons_state()

    def _update_mask_layer(self):
        if self.mask_pixmap_item and np.any(self.current_mask):
            mask_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            mask_img[self.current_mask > 0] = [0, 255, 0]
            mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            q_image = QImage(mask_rgb.data, self.w, self.h, 3 * self.w, QImage.Format_RGB888)
            self.mask_pixmap_item.setPixmap(QPixmap.fromImage(q_image))
            self.mask_pixmap_item.setVisible(True)
        elif self.mask_pixmap_item:
            self.mask_pixmap_item.setVisible(False)

    def save_state(self):
        if len(self.mask_history) >= self.max_history:
            self.mask_history.pop(0)
        self.mask_history.append(self.current_mask.copy())

    def update_buttons_state(self):
        has_image = self.img_original is not None
        has_output = self.output_dir != ""
        has_mask = self.current_mask is not None and np.any(self.current_mask)
        has_history = len(self.mask_history) > 0
        has_points = len(self.labeled_points) > 0
        
        self.confirm_mask_btn.setEnabled(has_image and has_mask)
        self.confirm_point_btn.setEnabled(has_image and has_output and has_mask)
        self.clear_mask_btn.setEnabled(has_image and has_mask)
        self.undo_btn.setEnabled(has_image and has_history)
        self.save_exit_btn.setEnabled(has_image and has_output)

    def confirm_mask(self):
        if np.any(self.current_mask):
            component_area = np.sum(self.current_mask > 0)
            self.status_label.setText(f"状态: 掩膜已确认")
            self.status_bar.showMessage(f"掩膜已确认 - 面积:{component_area}像素")
            QMessageBox.information(self, "提示", "掩膜已确认，可以继续添加控制点")

    def confirm_control_point(self):
        try:
            point_id = int(self.point_id_input.text())
        except:
            QMessageBox.warning(self, "警告", "请输入有效的控制点编号")
            return

        if not np.any(self.current_mask):
            QMessageBox.warning(self, "警告", "掩膜为空，请先标注ROI")
            return

        centroid = calculate_centroid(self.current_mask)
        if centroid is None:
            QMessageBox.warning(self, "警告", "无法计算重心")
            return

        cx, cy = centroid
        self.labeled_points.append((point_id, cx, cy))
        write_to_txt(self.save_path_txt, point_id, cx, cy)

        self.points_list.addItem(f"#{point_id} ({cx:.1f}, {cy:.1f})")

        self.status_label.setText(f"状态: 已标注 #{point_id}")
        self.status_bar.showMessage(f"已标注控制点 #{point_id}，重心: ({cx:.1f}, {cy:.1f})")

        self.current_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.mask_history.clear()
        self._update_mask_layer()
        self.point_id_input.setText(str(point_id + 1))
        self.update_buttons_state()

    def clear_mask(self):
        self.save_state()
        self.current_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self._update_mask_layer()
        self.status_label.setText("状态: 掩膜已清除")
        self.status_bar.showMessage("掩膜已清除")
        self.update_buttons_state()

    def undo_mask(self):
        if self.mask_history:
            self.current_mask = self.mask_history.pop()
            self._update_mask_layer()
            self.status_label.setText("状态: 已撤销")
            self.status_bar.showMessage("撤销上一次操作")
            self.update_buttons_state()

    def save_and_exit(self):
        if self.img_original is None:
            QMessageBox.warning(self, "警告", "没有可保存的内容")
            return

        final_img = self.img_original.copy()
        if np.any(self.current_mask):
            mask_color = np.zeros_like(final_img)
            mask_color[self.current_mask > 0] = [0, 255, 0]
            final_img = cv2.addWeighted(final_img, 0.7, mask_color, 0.3, 0)
        
        for (pid, cx, cy) in self.labeled_points:
            cv2.putText(final_img, str(pid), (int(cx) + 5, int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imwrite(self.save_path_img, final_img)

        QMessageBox.information(
            self, "保存完成",
            f"已标注 {len(self.labeled_points)} 个控制点\n\n"
            f"坐标文件: {self.save_path_txt}\n"
            f"标注图片: {self.save_path_img}"
        )
        self.close()

    def quit_labeling(self):
        reply = QMessageBox.question(
            self, "确认退出",
            "确定要放弃当前工作并退出吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.close()


def main():
    app = QApplication(sys.argv)
    window = ControlPointLabeler()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
