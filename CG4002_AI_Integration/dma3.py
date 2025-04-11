from pynq import Overlay, allocate
import numpy as np
from statistics import median, variance, mean, stdev
from scipy.stats import iqr, skew, kurtosis
from math import sqrt
import json
import time

INPUT_SIZE = 72
NUM_COLS = 12
BITSTREAM_PATH = '/home/xilinx/Integration/design_1.bit' 
TEST_DATA_PATH = '/home/xilinx/Integration/Dataset/test_data.json'
FEATURE_MIN_PATH = ''
FEATURE_MAX_PATH = ''

class action_predictor():
    def __init__(self):
        print("Loading FPGA Overlay...")
        self.ol = Overlay(BITSTREAM_PATH)
        print("Overlay Loaded.")
        self.dma = self.ol.axi_dma_0
        self.neural_net = self.ol.predict_action_0
        self.neural_net.write(0x00, 0x81) # start and auto restart

        self.dma_send = self.dma.sendchannel
        self.dma_recv = self.dma.recvchannel

        self.input_buffer = allocate(shape=(INPUT_SIZE,), dtype='int32')
        self.output_buffer = allocate(shape=(1,), dtype='int32')

        self.feature_min = np.load(FEATURE_MIN_PATH)
        self.feature_max = np.load(FEATURE_MAX_PATH)

        self.hacky_start_nn()

    def hacky_start_nn(self):
        print("Starting hacky_start_nn...")
        self.predict_action([0]*INPUT_SIZE)
        print("hacky_start_nn completed.")
    
    def normalize(self, data):
        """ Normalize feature vector """
        epsilon = 1e-8  # Small value to prevent division by zero
        return (data - self.feature_min) / (self.feature_max - self.feature_min + epsilon)
    
    def process_data(self, raw_data):
        '''
        takes in 30 * 12 raw data from the sensors and process the data, extracting the following
        features from the data to get an INPUT_SIZE of 96.
        '''

        features = []
        for col in range(NUM_COLS):
            col_data = [row[col] for row in raw_data]
            features.append(median(col_data))
            features.append(variance(col_data))
            features.append(max(col_data))
            features.append(min(col_data))
            features.append(mean(col_data))
            features.append(stdev(col_data))
            features.append(kurtosis(col_data))
            features.append(skew(col_data))
        features = np.array(features) 
        
        normalized_features = self.normalize(features)

        normalized_features = (normalized_features * (2 ** 16)).astype(np.int32)
        
        return normalized_features

    def predict_action(self, input_data):
        start_dma = time.time()
        print(f"🔍 Input Data Sent to DMA: {input_data[:5]} ...")  # Print first few values

        for i in range(INPUT_SIZE):
            self.input_buffer[i] = input_data[i]

        print("Sending data to DMA...")
        print(f"DMA Input Buffer Before Sending: {self.input_buffer[:5]}")
        self.dma_send.transfer(self.input_buffer)
        self.dma_recv.transfer(self.output_buffer)

        # Check for hanging DMA waits
        start_wait = time.time()
        while not self.dma_recv.idle:
            if time.time() - start_wait > 5:  # Timeout after 5s
                print("⚠️ DMA receive is stuck! Aborting...")
                return -1
            time.sleep(0.1)

        self.dma_send.wait()
        self.dma_recv.wait()
        print(f"DMA Transfer Time: {time.time() - start_dma:.4f} seconds")

        action = self.output_buffer[0]
        print(f"Predicted action: {action}")
        return action


    def get_prediction(self, raw_data):
        features = self.process_data(raw_data)
        return self.predict_action(features)
    
    def test_model(self, sample_size = 5):
        with open(TEST_DATA_PATH) as test_data_json:
            labelled_test_data = json.load(test_data_json)

        # Extract features and labels
        X_test = [d['features'] for d in labelled_test_data]
        y_test = [d['label'] for d in labelled_test_data]
        
        wrong_predictions = []
        correct_predictions = 0
        start_time = time.time()

        for i, x in enumerate(X_test):
            predicted_action = self.predict_action(x)
            if predicted_action == y_test[i]:
                correct_predictions += 1
            else:
                wrong_predictions.append((i, predicted_action, y_test[i]))
        elapsed_time = time.time() - start_time
        accuracy = (correct_predictions / len(y_test)) * 100

        print(f"\nPrediction completed in {elapsed_time:.2f} seconds.")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Number of wrong predictions: {len(wrong_predictions)}")

        # Display wrong predictions
        if wrong_predictions:
            print("\nWrong Predictions:")
            for idx, pred, actual in wrong_predictions:
                print(f"Sample {idx}: Predicted = {pred}, Actual = {actual}")

if __name__ == '__main__':
    classifier = action_predictor()
    classifier.test_model()
    