from analysisargs import AnalysisArgs
from analysisargs import Logs
from analysisargs import ProjectionsType
from analysisargs import DevicesType
from analysisargs import FuncIdsType
from analysisargs import TestModesType
import numpy as np
import typing


class DrawingTimesHandler:
    _a: AnalysisArgs
    """Hyperparameters of the analysis"""
    _l: Logs
    """All of the generated data & logs from the experiment"""

    def __init__(self, args: AnalysisArgs, logs: Logs = None) -> None:
        self._a = args
        self._l = logs if logs is not None else Logs()


    def getAllDrawingTimesForFunc(self, projection: ProjectionsType, funcId: FuncIdsType, device: DevicesType, experimentMode: TestModesType):
        # filter out by projection, Cartesian or Polar
        drawingTimes = self._l.df[self._l.df['Function projection'] == projection]
        # filter out by function ID
        drawingTimes = drawingTimes[drawingTimes['Function ID'] == funcId]
        # filter out by test (experiment mode)
        drawingTimes = drawingTimes[drawingTimes['Test mode'] == experimentMode]
        # filter out by device
        drawingTimes = drawingTimes[drawingTimes['Device'] == device]
        
        # for each user, find his/her average for this function
        participants = list(set(drawingTimes["Participant name"]))
        # find average of each participant for this function
        retval = []
        for participant in participants:
            dataForParticipant = drawingTimes[drawingTimes["Participant name"] == participant]
            # we always return the mean here, regardless of central tendency
            # because here we're calculating for each participant, each curve, the mean between repetitions
            avg = np.mean(dataForParticipant["Drawing time"].values)

            retval.append(avg)
            
        # return this to return ALL drawing times, without calculating mean for each participant
        return retval
    
    def getAvgForFunc(self, projection: ProjectionsType, funcId: FuncIdsType, device: DevicesType, experimentMode: TestModesType = 0):
        drawingTimes = self.getAllDrawingTimesForFunc(projection, funcId, device, experimentMode)
        
        if self._a.centralTendency == 'mean':
            avg = np.mean(drawingTimes)
        elif self._a.centralTendency == 'median':
            avg = np.median(drawingTimes)
        else:
            raise NotImplementedError(f'Central tendency of type {self._a.centralTendency} not supported!')        
        return avg #* 1000 # transform to milliseconds

