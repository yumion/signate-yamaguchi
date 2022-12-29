import cv2


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """
        cls.model = None

        return True


    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: Data of the sample you want to make inference from (str)

        Returns:
            list: Inference for the given input.

        """
        prediction = []
        cap = cv2.VideoCapture(input)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if ret:
                prediction.append({'frame_id':frame_id, 'line': 1, 'sign': 0, 'light': 0})
                frame_id += 1
            else:
                break

        return prediction
