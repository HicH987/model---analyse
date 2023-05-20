import tensorflow as tf
import numpy as np

from helpers.normlization import norm_X

class DeepFitClassifier:
    def __init__(self, model_path, labels):
        """
        Load the model and allocate resources \n
        Get input and output tensors.
        """
        self.labels = labels
        if model_path.endswith(".tflite"):
            self.model_type = "tflite"
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        elif model_path.endswith(".h5"):
            self.model_type = "h5"
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

    def predict(self, X) -> tuple:
        """
        X -> (1, 36) shaped numpy array
        First 18 are the x coordinates of the 18 keypoints \n
        next 18 are the y coordinates of the 18 keypoints \n
        Returns: \n
        Winning Label
        """
        # Setup input
        X_sample = np.array(X).reshape(1, -1)

        X_sample_norm = norm_X(X_sample)

        if self.model_type == "tflite":
            input_data = np.array(X_sample_norm[0], np.float32).reshape(1, 36)
            # Invoke the model on the input data
            
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()
            # Get the result
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        elif self.model_type == "h5":
            input_data = np.array(X_sample_norm[0], np.float32).reshape(1, 36)
            # Invoke the model on the input data
            output_data = self.model.predict(input_data)

        np_output_data = np.array(output_data)
        champ_idx = np.argmax(np_output_data)
        self.results = dict(zip(self.labels, output_data[0]))

        prediction_proba = float("{:.2f}".format(np_output_data[0][champ_idx]))
        return self.labels[champ_idx], prediction_proba

    def get_results(self) -> dict:
        """
        Get a dictionary containing the probability of each label for the last predicted input
        """
        return self.results
