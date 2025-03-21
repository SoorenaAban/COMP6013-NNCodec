import matplotlib.pyplot as plt
import numpy as np

class PerformanceDisplay:
    def __init__(self, logs, 
                 fig_size=(10, 6), dpi=100, font_size=12, 
                 dot_size=5, dot_alpha=0.3,
                 dot_color='blue',
                 trend_line_color='red', trend_line_linewidth=2,
                 moving_avg_window=10):
        """
        Initialize with a list of log objects and visual settings.
        
        Parameters:
          logs: list of log objects.
          fig_size: tuple, figure size in inches (width, height).
          dpi: int, dots per inch for the figure.
          font_size: int, base font size.
          dot_size: int, size of the scatter plot dots.
          dot_alpha: float, transparency for dots (0=transparent, 1=opaque).
          dot_color: color for scatter dots.
          trend_line_color: color for the moving average trend line.
          trend_line_linewidth: line width for the trend line.
          moving_avg_window: int, window size for computing the moving average.
        """
        self.logs = logs
        self.fig_size = fig_size
        self.dpi = dpi
        self.font_size = font_size
        self.dot_size = dot_size
        self.dot_alpha = dot_alpha
        self.dot_color = dot_color
        self.trend_line_color = trend_line_color
        self.trend_line_linewidth = trend_line_linewidth
        self.moving_avg_window = moving_avg_window

    def _moving_average(self, data):
        """
        Compute the moving average of the data using a window specified by self.moving_avg_window.
        Uses mode 'same' so that the output has the same length as the input.
        
        Parameters:
            data: list or numpy array of numerical values.
        
        Returns:
            numpy array of moving average values.
        """
        window = self.moving_avg_window
        if window < 1:
            raise ValueError("moving_avg_window must be at least 1")
        return np.convolve(data, np.ones(window) / window, mode='same')
    
    def _plot_graph(self, y_values, title, xlabel, ylabel, save_path=None):
        """
        Unified plotting function that generates a scatter plot with a moving average trend line.
        
        Parameters:
          y_values: list of numerical values for y-axis.
          title: title of the graph.
          xlabel: label for x-axis.
          ylabel: label for y-axis.
          save_path: (Optional) file path to save the graph.
        """
        if not y_values:
            print(f"No data available for {title}.")
            return
        
        # x-axis is simply the log entry order
        x = np.arange(1, len(y_values) + 1)
        y = np.array(y_values)
        trend = self._moving_average(y)
        
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # Plot the data points using a scatter plot with small, transparent dots.
        plt.scatter(x, y, s=self.dot_size, alpha=self.dot_alpha, color=self.dot_color, label="Data points")
        # Plot the moving average trend line.
        plt.plot(x, trend, color=self.trend_line_color, linewidth=self.trend_line_linewidth, label="Moving Average Trend")
        
        plt.title(title, fontsize=self.font_size + 2)
        plt.xlabel(xlabel, fontsize=self.font_size)
        plt.ylabel(ylabel, fontsize=self.font_size)
        plt.grid(True)
        plt.legend(fontsize=self.font_size)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_encoded_symbol_probability(self, save_path=None):
        """
        Extracts the probability values from EncodedSymbolProbability logs and plots them.
        Y-axis: log.prob (probability).
        """
        values = [log.prob for log in self.logs if hasattr(log, 'prob')]
        self._plot_graph(values, "Encoded Symbol Probability", "Log Entry Order", "Probability", save_path)

    def plot_prediction_model_training_log(self, save_path=None):
        """
        Extracts the loss values from PredictionModelTrainingLog logs and plots them.
        Y-axis: log.loss (training loss).
        """
        values = [log.loss for log in self.logs if hasattr(log, 'loss')]
        self._plot_graph(values, "Prediction Model Training Loss", "Log Entry Order", "Loss", save_path)

    def plot_coding_log(self, save_path=None):
        """
        Computes the ratio (symbol_size / encoded_size) from CodingLog logs and plots them.
        Y-axis: ratio computed as (symbol_size / encoded_size).
        """
        values = []
        for log in self.logs:
            if hasattr(log, 'symbol_size') and hasattr(log, 'encoded_size'):
                # Prevent division by zero.
                ratio = log.symbol_size / log.encoded_size if log.encoded_size != 0 else 0
                values.append(ratio)
        self._plot_graph(values, "Coding Log Ratio (symbol_size / encoded_size)", "Log Entry Order", "Ratio", save_path)
