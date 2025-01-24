from typing import List
from pydantic import BaseModel


# Define the input format for each individual data row (ModelInput) with default values
class ModelInput(BaseModel):
    date_time: str = "24 08 2018 21:20"
    wind_speed: float = 8.218296051
    theoretical_power: float = 1657.373187
    wind_direction: float = 78.12586975


# Define the input format for a batch of inputs (BatchInput) with default values
class BatchInput(BaseModel):
    inputs: List[ModelInput] = [
        ModelInput(
            date_time="24 08 2018 21:20",
            wind_speed=8.218296051,
            theoretical_power=1657.373187,
            wind_direction=78.12586975
        ),
        ModelInput(
            date_time="20 05 2018 11:10",
            wind_speed=4.995032787,
            theoretical_power=334.7802577,
            wind_direction=17.07011986
        ),
        ModelInput(
            date_time="04 04 2018 19:00",
            wind_speed=2.212670088,
            theoretical_power=0.0,
            wind_direction=127.6598969
        ),
    ]


# Define the input format for each individual training data row (TrainInput) with default values
class TrainInput(BaseModel):
    date_time: str = "24 08 2018 21:20"
    active_power: float = 1339.537964
    wind_speed: float = 8.218296051
    theoretical_power: float = 1657.373187
    wind_direction: float = 78.12586975


# Define the input format for a batch of training inputs (BatchTrainInput) with default values
class BatchTrainInput(BaseModel):
    inputs: List[TrainInput] = [
        TrainInput(
            date_time="24 08 2018 21:20",
            active_power=1339.537964,
            wind_speed=8.218296051,
            theoretical_power=1657.373187,
            wind_direction=78.12586975
        ),
        TrainInput(
            date_time="20 05 2018 11:10",
            active_power=277.7774048,
            wind_speed=4.995032787,
            theoretical_power=334.7802577,
            wind_direction=17.07011986
        ),
        TrainInput(
            date_time="04 04 2018 19:00",
            active_power=0.0,
            wind_speed=2.212670088,
            theoretical_power=0.0,
            wind_direction=127.6598969
        ),
    ]
