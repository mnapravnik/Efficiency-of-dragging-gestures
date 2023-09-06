from analysisargs import AnalysisArgs
from analysisargs import Logs
from analysisargs import ProjectionsType
from analysisargs import DevicesType
from analysisargs import FuncIdsType
from analysisargs import TestModesType
import numpy as np
import display_properties
import typing
from curve_functions import FunctionProvider, get_difficulty_and_task_from_func_id
from curve_functions import get_test_index_from_test_mode_and_projection
import os
import glob
import matplotlib.pyplot as plt


class DrawingHandler:
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


    def get_mean_drawtime_for_participant(self, participantname:str, device:DevicesType, projections:typing.List[ProjectionsType]):
        """Will return the mean drawing time across all curves (trajectories),
        drawn using the specified device and in specified projection(s).

        Args:
            participantname (str): Name of the participant.
            device (DevicesType): Device used to draw the curve (trajcetory).
            projections (typing.List[ProjectionsType]): Across which projections should the mean drawing
                times be calculated. Can contain both Cartesian and Polar projection, just the Polar
                or just the Cartesian projection.

        Returns:
            tuple: the first element is the drawing time mean, the second element is
                the standard deviation.
        """
        # gather all logs from which to calculate mean drawing time
        participant_logs = self._l.df.query(
            f'`Participant name` == "{participantname}" and ' +
            f'`Device` == "{device}" and ' +
            f'`Function projection` in {projections}'
        )

        mean = participant_logs['Drawing time'].mean()
        std = participant_logs['Drawing time'].std()
        return (mean, std)

    def get_error(
        self,
        participantname: str,
        device: DevicesType,
        projection: ProjectionsType,
        test_mode: TestModesType,
        func_id: FuncIdsType,
        npoints:int = 5000
    ):
        """Get the error a user drew during experiments.

        Args:
            participantname (str): Name of the user for who to fetch
                errors.
            device (DevicesType): which device did the user use? Should the error be calculated
                for drawing made using Mouse or Graphic tablet?
            projection (ProjectionsType): Is the curve, for which to calculate the error,
                drawn in polar or cartesian coordinates?
            test_mode (TestModesType): was this drawing done in the first or second part of the experiment?
            func_id (FuncIdsType): ID of the curve/trajectory/function.
            npoints (int, optional): How many points to generate on the *real* curve
                to use for finding nearest point (distance from drawn point to curve's nearest point).
                Defaults to 5000.

        Returns:
            An array where each item containes calculated error for i-th drawing.
            If a user drew the same curve twice, then this will have two elements.
            Each element is a dictionary with the following attributes:
            - 'drawnx' - an array of X coordinates drawn
            - 'drawny' - an array of Y coordinates drawn. 'drawnx' and 'drawny' are coordinate pairs
            - 'realx' - an array of X coordinates of points which are nearest to the i-th drawn point
            - 'realy' - an array of Y coordinates, 'realx' and 'realy' make a coordinate point which
                lies on the real function trajectory
            - 'dist' - distance between the real and the drawn point
        """
        # When calculating errors, we generate N points distributed equally across X axis
        # then we go along the drawing curve for each of the N points, and calculate which
        # point on the real curve is nearest
        # and what is the distance to it.

        # first get the real Y values which should have been drawn
        xrange_start = display_properties.X_RANGE['start']
        xrange_end = display_properties.X_RANGE['end']
        fp = FunctionProvider()
        realxpoints = np.linspace(start=xrange_start, stop=xrange_end, num=npoints)
        difficulty, task = get_difficulty_and_task_from_func_id(func_id=func_id)
        test_index = get_test_index_from_test_mode_and_projection(test_mode=test_mode, projection=projection)

        realypoints = fp.provide_function_y(
            difficulty=difficulty,
            task=task,
            test_index=test_index,
            x=realxpoints
        )

        # now get the drawn coordinates
        drawnpoints = get_drawn_coordinates(
            participantname=participantname,
            device=device,
            projection=projection,
            test_mode=test_mode,
            func_id=func_id)
        
        # test to see the drawn vs real coordinates
        if projection == 'Cartesian':
            # cartesian projection
            fig, ax = plt.subplots()
        else:
            # polar projection
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(drawnpoints[0]['x'], drawnpoints[0]['y'], label='Drawn 1st')
        ax.plot(drawnpoints[1]['x'], drawnpoints[1]['y'], label='Drawn 2nd')
        ax.plot(realxpoints, realypoints, label='Real func')
        ax.legend()
        plt.show(); plt.close(fig)

        # once we've established the proper coordinates were fetched, now we go onto
        # calculating the actual error
        # for each drawn point, search for the nearest real point
        retval = []
        for i, rpt in enumerate(drawnpoints):
            # rpt = number of repeats
            # i.e. we're iterating over each drawing repeat
            errors = dict({'realx': [], 'realy': [], 'drawnx': [], 'drawny': [], 'dist': []})
            for x_a, y_a in zip(rpt['x'], rpt['y']):
                # this will be an array of distance from drawn point A to
                # ALL real curve points
                dists = get_dist_between_points(x_a, y_a, realxpoints, realypoints, projection=projection)
                # chose whichever point was the nearest, and calculate
                # the error as distance to that point
                nearest_point_idx = np.argmin(dists)
                errors['realx'].append(realxpoints[nearest_point_idx])
                errors['realy'].append(realypoints[nearest_point_idx])
                errors['drawnx'].append(x_a)
                errors['drawny'].append(y_a)
                errors['dist'].append(np.min(dists))
            print(f'Total error in {i+1}-th repetition:', np.sum(errors['dist']), ', average:', np.mean(errors['dist']))
            retval.append(errors)
        return retval

        
def get_dist_between_points(x_a, y_a, x_b,  y_b, projection: ProjectionsType):
    if projection == 'Cartesian':
        c = (x_a - x_b)**2 + (y_a - y_b)**2
    else:
        # polar projection
        phi_a, r_a = x_a, y_a
        phi_b, r_b = x_b, y_b
        c = r_a**2 + r_b**2 - 2*r_a*r_b*np.cos(phi_b-phi_a)
    dist = np.sqrt(c)
    return dist 

def get_drawn_coordinates(
    participantname: str,
    device: DevicesType,
    projection: ProjectionsType,
    test_mode: TestModesType,
    func_id: FuncIdsType,
):
    """
    
    This will return an array of arrays.
    Each row represent a spearate curve drawing, a single drawing of the entire curve.
    Each item in these rows a dict that has X and Y coordinates as arrays.
    The number of drawn points in each row is not equal, i.e. the
    returned value is not a matrix. This is because on their first try,
    the participant might have drawn 4000 points,
    but on their second try they might have drawn 1000 points etc.
    
    """

    test_index = get_test_index_from_test_mode_and_projection(test_mode=test_mode, projection=projection)

    # then get the logs for this curve
    # folder where all logs for this participant and this device are
    foldername = os.path.join(f"../Results_backup{test_mode}", participantname, device)
    # there will be two logs with this wildcard name
    # because they were done in different orders and repeated twice
    logspath = os.path.join(foldername, f'id-{func_id}_proj-{test_index}_order-*.txt')
    
    retval = []
    for logsfilename in glob.glob(logspath):
        file = open(logsfilename)
        
        coords = {'x': [], 'y': []}

        # over here we can parse the logged coordinates
        for coord in file.readlines():
            drawnx, drawny = coord.split()
            coords['x'].append(float(drawnx))
            coords['y'].append(float(drawny))
        file.close()
        retval.append(coords)
    return retval