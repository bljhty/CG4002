This is the source code for NUS CG4002 LaserTag AR+ capstone project. 
The AI_model.py is used to train the data collected. the weights and biases is than being saved as a c++ header file 
which will than be used in the HLS systhesis of the AI model to be integrated into the FPGA.
Once the HLS code is done, it will be used to generate an IP block which is being used in VIVADO to do up a block design
A bitstream will then be gnerated and uploaded into the FPGA

dma3.py serves as the main driver for the AI model in the FPGA
