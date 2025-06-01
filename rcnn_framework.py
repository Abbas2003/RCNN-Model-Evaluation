import torchvision
import torch

class RCNNComparator:
    def __init__(self, num_classes=21, device=None):
        """
        Initializes lightweight RCNN models for comparison.
        Args:
            num_classes (int): Number of classes for detection.
            device (str or torch.device): Device to load models on.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Lightweight Faster R-CNN (MobileNet backbone)
        faster_rcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        faster_rcnn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        faster_rcnn = faster_rcnn.to(self.device)

        # Lightweight Mask R-CNN (ResNet-50 backbone, as MobileNet is not available in torchvision)
        mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        in_features_mask = mask_rcnn.roi_heads.box_predictor.cls_score.in_features
        mask_rcnn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features_mask, num_classes
        )
        in_features_mask = mask_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        mask_rcnn.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        mask_rcnn = mask_rcnn.to(self.device)

        # Lightweight RetinaNet (ResNet-50 backbone, as MobileNet is not available in torchvision)
        retinanet = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
        in_features_retina = retinanet.backbone.out_channels
        num_anchors = retinanet.head.classification_head.num_anchors
        retinanet.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
            in_features_retina, num_anchors, num_classes
        )
        retinanet = retinanet.to(self.device)

        self.models = {
            'Faster R-CNN': faster_rcnn,
            'Mask R-CNN': mask_rcnn,
            'RetinaNet': retinanet
        }