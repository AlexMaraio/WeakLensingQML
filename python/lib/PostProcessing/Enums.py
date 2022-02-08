from enum import Enum


class ClType(Enum):
    QML = 'QML'
    PCl = 'PCl'


class CovType(Enum):
    analytic = 'analytic'
    numeric = 'numeric'
