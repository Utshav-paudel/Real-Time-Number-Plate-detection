# Real Time Automatic-Number-Plate-Recognition-YOLOv8



## Data

The sample video to test can be downloaded [here](https://drive.google.com/file/d/1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta/view?usp=sharing).

## Model

A Yolov8 pre-trained model (YOLOv8n) was used to detect vehicles.

A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4). 
- The model is available [here](https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing).

## Dependencies

The sort module needs to be downloaded from [this repository](https://github.com/abewley/sort).

## Project Setup

Follow this steps to get your real time number plate detection ready

* Make an environment with python=3.8 using the following command 
``` bash
conda create --prefix ./env python==3.8 -y
```
* Activate the environment
``` bash
conda activate ./env
``` 

* Install the project dependencies using the following command 
```bash
pip install -r requirements.txt
```
* Run main.py with the sample video file to generate the test.csv file or provide ip of your camera
``` python
python main.py
```
* Run the add_missing_data.py file for interpolation of values to match up for the missing frames and smooth output.
```python
python add_missing_data.py
```

* Finally run the visualize.py passing in the interpolated csv files and hence obtaining a smooth output for license plate detection.
```python
python visualize.py
```

# Demo of Real Time object detection

Problem to solve  : Implement process with gpu
