from analysisargs import AnalysisArgs
# TODO Continue refactoring, to get IoDs and load them properly

# this part above was the only thing cut from the original ipynb

class IodHandler:
    _a: AnalysisArgs
    """Holds the hyperparamaters and config how to handle indices of difficulty."""

    def __init__(self, args: AnalysisArgs) -> None:
        self._a = args

    def get_iod_set(self):
        """Get a set of IoDs for the specific model.

        Returns:
            A dictionary of IoDs, by their respective test parts and difficulties.
        """
        return self._a.funcIoDs[self._a.iodModelName]

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