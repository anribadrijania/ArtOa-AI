class Segment:
    def __init__(self, model):
        self.model = model

    def predict(self, image):
        result = self.model(image)[0]
        masks = result.masks.data.cpu().numpy() if result.masks else None
        return masks

