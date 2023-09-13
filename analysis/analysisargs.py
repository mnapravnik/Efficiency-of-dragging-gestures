import typing
import json
from typing import Any
import pandas as pd


ProjectionsType = typing.Literal['Cartesian', 'Polar']
FuncIdsType = typing.Literal[0, 1, 2, 3, 4, 5]
DevicesType = typing.Literal['Mouse', 'Graphic tablet']
TestModesType = typing.Literal[0, 1]
IodsType = typing.Literal[
    'length',
    'length:width',
    'kappa',
    'length:(width*r^(1:3))',
    'length:(width+1:r)',
    'length:(width+1:r+width*1:r)',
    'tmp',
]


class AnalysisArgs:
    iodModelName: IodsType = 'tmp'
    centralTendency: typing.Literal['mean', 'median'] = 'mean'
    plotTextSize = 18

    PROJECTIONS: typing.List[ProjectionsType] = ['Cartesian', 'Polar']
    FUNC_IDS: typing.List[FuncIdsType] = [0, 1, 2, 3, 4, 5]
    DEVICES: typing.List[DevicesType] = ['Mouse', 'Graphic tablet']
    TEST_MODES: typing.List[TestModesType] = [0, 1]

    npoints: int = 5000
    """
    How many points (coordinates) to generate for intergal calculate (ID)
    and for error calculation etc.
    """

    useCroatian: bool=False

    save_figures: bool=False
    """If True, will save any plots made during analysis."""

class Logs:
    df: pd.DataFrame
    """This dataframe contains all of the drawing time logs, all data collected during experiment.
    
    Available columns:
    - "Participant name"
    - "Participant age"
    - "Participant handedness"
    - "Device"
    - "Test mode"
    - "Logging timestamp"
    - "Function ID"
    - "Function difficulty"
    - "Function projection"
    - "Drawing time"
    - "Error approx"
    - "Expert Mouse User"
    - "Expert Graphic Tablet User"
    """
    test0data: pd.DataFrame
    """A copy of data containing logs only from the first part of the experiment."""
    test1data: pd.DataFrame
    """A copy of data containing logs only from the second part of the experiment."""

    def __init__(self) -> None:
        self.df = pd.read_csv('timelogs.csv')
        self.test0data = self.df[self.df['Test mode'] == 0]
        self.test1data = self.df[self.df['Test mode'] == 1]
