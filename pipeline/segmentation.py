class Segment:
    """
    A class for performing image segmentation using a provided model.
    """
    def __init__(self, model):
        """
        Initialize the Segment class with a segmentation model.

        :param model: The pre-trained segmentation model to use for prediction.
        """
        self.model = model

    def predict(self, image):
        """
        Perform segmentation on the given image.

        :param image: The input image to segment.
        :return: A NumPy array of segmentation masks if available, otherwise None.
        """
        result = self.model(image, conf=0.1)[0]
        masks = result.masks.data.cpu().numpy() if result.masks else None
        return masks

