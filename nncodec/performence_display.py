import matplotlib.pyplot as plt
import numpy as np

class PerformanceDisplay:
    def __init__(self, logs, 
                 fig_size=(10, 6), dpi=100, font_size=12, 
                 dot_size=5, dot_alpha=0.3,
                 dot_color='blue',
                 trend_line_color='red', trend_line_linewidth=2,
                 moving_avg_window=10):
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
        window = self.moving_avg_window
        if window < 1:
            raise ValueError("moving_avg_window must be at least 1")
        return np.convolve(data, np.ones(window) / window, mode='same')
    
    def _plot_graph(self, y_values, title, xlabel, ylabel, show_graph = False, save_path=None):
        if not y_values:
            print(f"No data available for {title}.")
            return
        
        x = np.arange(1, len(y_values) + 1)
        y = np.array(y_values)
        trend = self._moving_average(y)
        
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        plt.scatter(x, y, s=self.dot_size, alpha=self.dot_alpha, color=self.dot_color, label="Data points")
        plt.plot(x, trend, color=self.trend_line_color, linewidth=self.trend_line_linewidth, label="Moving Average Trend")
        
        plt.title(title, fontsize=self.font_size + 2)
        plt.xlabel(xlabel, fontsize=self.font_size)
        plt.ylabel(ylabel, fontsize=self.font_size)
        plt.grid(True)
        plt.legend(fontsize=self.font_size)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        if show_graph:
            plt.show()

    def generate_encoded_symbol_probability_plot(self, show_graphs = False, save_path=None):
        values = [log.prob for log in self.logs if hasattr(log, 'prob')]
        self._plot_graph(values, "Encoded Symbol Probability", "Log Entry Order", "Probability", show_graphs, save_path)

    def generate_prediction_model_training_log_plot(self, show_graphs = False, save_path=None):
        values = [log.loss for log in self.logs if hasattr(log, 'loss')]
        self._plot_graph(values, "Prediction Model Training Loss", "Log Entry Order", "Loss", show_graphs, save_path)

    # def plot_coding_log(self, save_path=None):
    #     """
    #     Computes the ratio (symbol_size / encoded_size) from CodingLog logs and plots them.
    #     Y-axis: ratio computed as (symbol_size / encoded_size).
    #     """
    #     values = []
    #     for log in self.logs:
    #         if hasattr(log, 'symbol_size') and hasattr(log, 'encoded_size'):
    #             # Prevent division by zero.
    #             ratio = log.symbol_size / log.encoded_size if log.encoded_size != 0 else 0
    #             values.append(ratio)
    #     self._plot_graph(values, "Coding Log Ratio (symbol_size / encoded_size)", "Log Entry Order", "Ratio", save_path)
