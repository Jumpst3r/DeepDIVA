from .apply_model import ApplyModel
from .bidimensional import Bidimensional
from .image_classification import ImageClassification
from .triplet import Triplet
from .image_auto_encoding import ImageAutoEncoding
from .semantic_segmentation import SemanticSegmentation
from .semantic_segmentation_hisdb import SemanticSegmentationHisdb
from .semantic_segmentation_hisdb_singleclass import SemanticSegmentationHisdbSingleclass

__all__ = ['ImageClassification', 'Bidimensional', 'Triplet', 'ApplyModel',
           'ImageAutoEncoding', 'SemanticSegmentation', 'SemanticSegmentationHisdb',
           'SemanticSegmentationHisdbSingleclass']