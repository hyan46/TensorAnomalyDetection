"""
Tensor-based anomaly detection and image process monitoring.

Modules:
- tensor_monitoring: UPCA, Tucker (MPCA), CP (TROD) control charts (Yan et al., IEEE TASE 2015).
- tensordetect: TensorHotellingT2 (T² and residual-based detection).
- tensor_anomaly_detection: Sparse + low-rank decomposition for anomaly localization.
- three_way_decomposition: Background + anomaly tensor decomposition.
"""

from tensor_monitoring import (
    BaseTensorControlChart,
    UPCAControlChart,
    TuckerControlChart,
    CPControlChart,
)
from tensordetect import TensorHotellingT2
from tensor_anomaly_detection import TensorAnomalyDetection

__all__ = [
    "BaseTensorControlChart",
    "UPCAControlChart",
    "TuckerControlChart",
    "CPControlChart",
    "TensorHotellingT2",
    "TensorAnomalyDetection",
]
