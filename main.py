import running_inference as ri




dataset = 'test'
model = 'rf-detr_model_top-view.pth'
boxes, masks_resized, scores, classes, annotated_frame = ri.detect_model(model,dataset)
