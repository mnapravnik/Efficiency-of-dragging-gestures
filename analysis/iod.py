from analysisargs import AnalysisArgs
from statutils import calculate_riemann_integral
import json
from curve_functions import FunctionProvider, is_cartesian
import display_properties as dp
import numpy as np
import sympy as sp
from analysisargs import IodsType
import os
import typing


class IodHandler:
    _a: AnalysisArgs
    """Holds the hyperparamaters and config how to handle indices of difficulty."""

    _fp: FunctionProvider
    """For handling functions and calculating indices of difficulty from them"""

    _x: sp.Symbol
    """Independent vairable symbol for using sympy."""

    _iod_set: typing.Dict[int, typing.Dict[int, float]]
    """Currently active set of IoDs."""


    def __init__(self, args: AnalysisArgs) -> None:
        self._a = args
        self._fp = FunctionProvider()
        self._x = sp.Symbol('x')
        self._iod_set = None

    def get_iod_set(self):
        """Get a set of IoDs for the specific model.

        Returns:
            A dictionary of IoDs, by their respective test parts and difficulties.
        """
        iod_model_name = self._a.iodModelName

        if self._iod_set is None:
            filename = os.path.join(self.get_iod_filepath(iod_model_name=iod_model_name))

            if os.path.exists(filename) is False:
                print(f'No IoD found under {filename}, calculating it...')
                self.calculate_index_of_difficulty_integral(iod_model_name=iod_model_name)
            
            with open(filename, 'r') as file:
                print('Loading IoD file...')
                set = json.load(file)

            self._iod_set = set

        return self._iod_set

    def getIodForFunc(self, projection, experimentMode, funcId):
        test = 0
        if projection == 'Cartesian' and experimentMode == 1:
            test = 1
        elif projection == 'Polar' and experimentMode == 0:
            test = 2
        elif projection == 'Polar' and experimentMode == 1:
            test = 3
        retval = float(self.get_iod_set()[str(test)][str(funcId)])
        return retval

    def getIodsAsArray(self, projections, experimentModes):
        iodsArr = []
        for experimentMode in experimentModes:
            for projection in projections:            
                for funcId in self._a.FUNC_IDS:
                    iodsArr.append(self.getIodForFunc(projection, experimentMode, funcId))
        return iodsArr

    def getMaxIodForPlot(self):
        return round(max(self.getIodsAsArray(self._a.PROJECTIONS, self._a.TEST_MODES))) * 1.2
    
    def get_iod_equation(
            self,
            difficulty: int,
            task: int,
            test_index: int,
            iod_model_name: IodsType = None,
            lambdified:bool=True
        ):
        """Get equation for a trajectory's index of difficulty.

        difficulty, task and test_index can be calculated based on test_mode (sometimes called
        experiment_mode) and function ID. See the methods available
        in curve_functions.py.

        Args:
            difficulty (int): difficulty index of the trajectory.
            task (int): trajectory task index
            test_index (int): test index of the trajectory
            iod_model_name (IodsType, optional): name of the ID model. If None,
                will be parsed from AnalysisArgs (given in init). Defaults to None.
            lambdified (bool, optional): whether to return the equation
                in its sympy lambdified form. Defaults to True.

        Raises:
            ValueError: if index of difficulty model is not implemented.

        Returns:
            the (optionally lambdified) equation describing the index of difficulty.
        """
        if iod_model_name is None:
            iod_model_name = self._a.iodModelName

        kappa = self._fp.get_function_curvature(difficulty, task, test_index)
        length = self._fp.get_function_length(difficulty, task, test_index)
        # alpha: width of the line.
        alpha = 1
        if(is_cartesian(test_index=test_index)):
            length = length * dp.CARTESIAN_UNIT_LENGTH_IN_INCH
            alpha *= dp.CARTESIAN_UNIT_LENGTH_IN_INCH
        else:
            length = length * dp.POLAR_UNIT_LENGTH_IN_INCH
            alpha *= dp.POLAR_UNIT_LENGTH_IN_INCH
        
        iod = None
        radius = 1 / (0.0000001 + np.abs(kappa))
        if iod_model_name == 'length':
            iod = length
        elif iod_model_name == 'kappa':
            iod = kappa
        elif iod_model_name == 'length:width':
            iod = length / alpha
        elif iod_model_name == 'length:(width*r^(1:3))':
            # this below is taken from the paper "Modeling user performance on Curved Constrained Paths"
            iod = length / (alpha * (radius+1) ** (1/3))
        elif iod_model_name == 'length:(width+1:r)':
            iod = length / (alpha + 1/(radius+1))
        elif iod_model_name == 'length:(width+1:r+width*1:r)':
            iod = length / (alpha + 1/(radius+1) + alpha * 1 / (radius+1))
        else:
            raise ValueError(f'Index of difficulty by the name of "{iod_model_name}" not implemented!')

        retval = iod
        if lambdified is True:
            lambified_iod = sp.lambdify(self._x, iod, "numpy")
            retval = lambified_iod
        
        return retval

    
    def calculate_index_of_difficulty_integral(self, iod_model_name: IodsType = None):
        """Calculate integral of index of difficulty.

        This will calculate integral approximations for all of the trajectories available
        and will store them in a json file, ready for future use.

        Args:
            iod_model_name (IodsType, optional): name of the ID model. If None,
                will be parsed from AnalysisArgs (given in init). Defaults to None.
        """
        if iod_model_name == None:
            iod_model_name = self._a.iodModelName

        x0 = dp.X_RANGE['start']
        x1 = dp.X_RANGE['end']
        npoints = self._a.npoints

        integrals_approx = {}
        for test in range(len(self._fp.function_array)):
            integrals_approx[test] = {}
            for difficulty in range(len(self._fp.function_array[test])):
                tasks_num = len(self._fp.function_array[test][difficulty])
                for task in range(tasks_num):
                    function_id = tasks_num * difficulty + task
                    print("########## Test mode:", test, "; Fuction ID", function_id, " Difficlty:", difficulty, "; Task:", task)
                    index_of_difficulty = self.get_iod_equation(
                        difficulty=difficulty,
                        task=task,
                        test_index=test,
                        iod_model_name=iod_model_name,
                        lambdified=True
                    )
                    index_of_difficulty = calculate_riemann_integral(index_of_difficulty, x0, x1, numpoints=npoints)
                    print(index_of_difficulty)
                    integral_approx = index_of_difficulty

                    print("Integral: ", integral_approx, "\n")
                    integrals_approx[test][str(function_id)] = str(integral_approx)
        print(integrals_approx)

        os.makedirs('iods', exist_ok=True)
        file = open(self.get_iod_filepath(iod_model_name=iod_model_name), "w")
        json.dump(integrals_approx, fp=file, sort_keys=True, indent=4)
        file.close()
    
    def get_iod_filepath(self, iod_model_name: IodsType = None):
        """
        Path were the ID model `iod_model_name` should be stored.
        This will return a string WITH the `.json` extension included.
        """
        if iod_model_name == None:
            iod_model_name = self._a.iodModelName
        
        return os.path.join('iods', f'index_of_difficulty-{iod_model_name}.json')