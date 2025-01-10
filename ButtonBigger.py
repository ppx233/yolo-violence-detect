from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize


class HoverButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super(HoverButton, self).__init__(*args, **kwargs)
        self.default_size = QSize(151, 51)  # ��ʼ��С

    def enterEvent(self, event):
        self.setFixedSize(self.default_size.width() * 1.2, self.default_size.height() * 1.2)  # �Ŵ�
        super(HoverButton, self).enterEvent(event)

    def leaveEvent(self, event):
        self.setFixedSize(self.default_size)  # �ָ�ԭʼ��С
        super(HoverButton, self).leaveEvent(event)
