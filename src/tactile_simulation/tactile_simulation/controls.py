import sys

import rclpy
from rclpy.node import Node
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QApplication, QComboBox, QLabel, QPushButton,
                            QSpinBox, QVBoxLayout, QWidget)
from std_srvs.srv import Trigger
from tactile_interfaces.srv import SpawnObjects

from .objects import KINDS


class Controls(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle('Grasp objects')
        self.setMinimumWidth(300)
        self.add_client = node.create_client(SpawnObjects, 'scene/spawn_objects')
        self.clear_client = node.create_client(Trigger, 'scene/clear_objects')
        self.reset_client = node.create_client(Trigger, 'scene/reset_arm')
        self.layout_client = node.create_client(Trigger, 'scene/reset_layout')
        self.pending = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Object type'))
        self.kind = QComboBox()
        self.kind.addItems(('mixed', *KINDS))
        layout.addWidget(self.kind)
        layout.addWidget(QLabel('Quantity (48 scene slots total)'))
        self.count = QSpinBox()
        self.count.setRange(1, 48)
        self.count.setValue(8)
        layout.addWidget(self.count)
        layout.addWidget(QLabel('Layout seed'))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        layout.addWidget(self.seed)
        self.buttons = []
        for label, action in [('Add objects', self.add), ('Clear objects', self.clear),
                              ('New mixed scene', self.new_scene),
                              ('Restore original layout', self.restore_layout), ('Reset arm pose', self.reset)]:
            button = QPushButton(label)
            button.clicked.connect(action)
            layout.addWidget(button)
            self.buttons.append(button)
        self.status = QLabel('Connecting to playground…')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(QLabel('Arm sliders command the simulator.\nContacts and gravity are enabled.'))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(30)

    def send(self, client, request, after=None):
        if not client.service_is_ready():
            self.status.setText('Playground service is not available')
            return
        self.pending = (client.call_async(request), after)
        self.status.setText('Updating scene…')

    def add(self):
        request = SpawnObjects.Request()
        request.kind, request.count, request.seed = self.kind.currentText(), self.count.value(), self.seed.value()
        self.send(self.add_client, request)

    def clear(self):
        self.send(self.clear_client, Trigger.Request())

    def new_scene(self):
        self.kind.setCurrentText('mixed')
        self.seed.setValue((self.seed.value() + 1) % 2147483647)
        self.send(self.clear_client, Trigger.Request(), self.add)

    def reset(self):
        self.send(self.reset_client, Trigger.Request())

    def restore_layout(self):
        self.send(self.layout_client, Trigger.Request())

    def poll(self):
        rclpy.spin_once(self.node, timeout_sec=0)
        if self.pending and self.pending[0].done():
            future, after = self.pending
            self.pending = None
            try:
                result = future.result()
                self.status.setText(result.message)
                if result.success and after:
                    after()
            except Exception as exc:
                self.status.setText(str(exc))
        ready = self.pending is None and self.add_client.service_is_ready()
        for button in self.buttons:
            button.setEnabled(ready)
        if ready and self.status.text() == 'Connecting to playground…':
            self.status.setText('Ready')


def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    node = Node('grasp_object_controls')
    window = Controls(node)
    window.show()
    try:
        app.exec_()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
