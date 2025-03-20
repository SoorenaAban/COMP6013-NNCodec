import matplotlib.pyplot as plt
import numpy as np

class PerformanceDisplay:
    def __init__(self, logs):
        """
        Initialize with a list of log objects.
        :param logs: List of log objects.
        """
        self.logs = logs

    def _add_trend_line(self, x, y):
        """
        Compute and return the trend line values using linear regression.
        :param x: x-axis values (e.g., log entry order).
        :param y: y-axis values.
        :return: y values of the trend line.
        """
        slope, intercept = np.polyfit(x, y, 1)
        trend = slope * np.array(x) + intercept
        return trend

    def plot_encoded_symbol_probability(self, save_path=None):
        """
        Generate a scatter plot with a trend line for EncodedSymbolProbability logs.
        X-axis: Log entry order.
        Y-axis: 'prob' attribute (probability).
        
        :param save_path: (Optional) File path to save the graph.
        """
        probabilities = [log.prob for log in self.logs if hasattr(log, 'prob')]
        if not probabilities:
            print("No EncodedSymbolProbability logs found.")
            return
        
        x = list(range(1, len(probabilities) + 1))
        y = probabilities
        trend = self._add_trend_line(x, y)
        
        plt.figure()
        plt.scatter(x, y, label="Data points")
        plt.plot(x, trend, label="Trend line")
        plt.title("Encoded Symbol Probability")
        plt.xlabel("Log Entry Order")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_prediction_model_training_log(self, save_path=None):
        """
        Generate a scatter plot with a trend line for PredictionModelTrainingLog logs.
        X-axis: Log entry order.
        Y-axis: 'loss' attribute (training loss value).
        
        :param save_path: (Optional) File path to save the graph.
        """
        losses = [log.loss for log in self.logs if hasattr(log, 'loss')]
        if not losses:
            print("No PredictionModelTrainingLog logs found.")
            return
        
        x = list(range(1, len(losses) + 1))
        y = losses
        trend = self._add_trend_line(x, y)
        
        plt.figure()
        plt.scatter(x, y, label="Data points")
        plt.plot(x, trend, label="Trend line")
        plt.title("Prediction Model Training Loss")
        plt.xlabel("Log Entry Order")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_coding_log(self, save_path=None):
        """
        Generate a scatter plot with a trend line for CodingLog logs.
        X-axis: Log entry order.
        Y-axis: The ratio computed as (symbol_size / encoded_size).
        
        :param save_path: (Optional) File path to save the graph.
        """
        ratios = []
        for log in self.logs:
            if hasattr(log, 'symbol_size') and hasattr(log, 'encoded_size'):
                ratio = log.symbol_size / log.encoded_size if log.encoded_size != 0 else 0
                ratios.append(ratio)
        if not ratios:
            print("No CodingLog logs found.")
            return
        
        x = list(range(1, len(ratios) + 1))
        y = ratios
        trend = self._add_trend_line(x, y)
        
        plt.figure()
        plt.scatter(x, y, label="Data points")
        plt.plot(x, trend, label="Trend line")
        plt.title("Coding Log Ratio (symbol_size / encoded_size)")
        plt.xlabel("Log Entry Order")
        plt.ylabel("Ratio")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()
