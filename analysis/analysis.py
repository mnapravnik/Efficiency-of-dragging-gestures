from analysisargs import AnalysisArgs
from analysisargs import Logs
from analysisargs import ProjectionsType
from analysisargs import DevicesType
from analysisargs import FuncIdsType
from analysisargs import TestModesType
import numpy as np
import typing
from iod import IodHandler
from drawingtimes import DrawingHandler
from scipy.stats import shapiro as normal_shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os, shutil
import sympy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
from statutils import checkLinearAssumption, autocorrelationAssumption, getBasePlotTitle, printRegressionModelMetrics
from statutils import getRegressionCoefficients, normalErrorsAssumption, homoscedasticityAssumption, getMseAndRmspe

class Analysis:
    _a: AnalysisArgs
    _iod: IodHandler
    dt: DrawingHandler

    MAX_AVG_DRAW_TIME: float
    MIN_AVG_DRAW_TIME: float


    def __init__(self, args: AnalysisArgs, logs: Logs = None) -> None:
        self._a = args
        self._iod = IodHandler(args)
        self.dt = DrawingHandler(args=args, logs=logs)

        # for plot xy ranges
        self.MAX_AVG_DRAW_TIME = round(max(self.getAvgsByFilter(self._a.PROJECTIONS, self._a.FUNC_IDS, self._a.DEVICES, self._a.TEST_MODES)[0]))
        self.MIN_AVG_DRAW_TIME = round(min(self.getAvgsByFilter(self._a.PROJECTIONS, self._a.FUNC_IDS, self._a.DEVICES, self._a.TEST_MODES)[0]))
        self.MAX_AVG_DRAW_TIME += 0.2 * self.MAX_AVG_DRAW_TIME
        self.MIN_AVG_DRAW_TIME -= 0.2 * self.MIN_AVG_DRAW_TIME

    def getAvgsByFilter(self, projections:typing.List[ProjectionsType], funcIds: typing.List[FuncIdsType], devices: typing.List[DevicesType], experimentModes: typing.List[TestModesType]):
        times = []
        iods = []
        for experimentMode in experimentModes:
            for device in devices:
                for projection in projections:
                    for funcId in funcIds:
                        avg = self.dt.getAvgForFunc(projection, funcId, device, experimentMode)
                        times.append(avg)
                        iods.append(self._iod.getIodForFunc(projection, experimentMode, funcId))
                        # use this if you want to get ALL drawing times paired with iods
                        # avg = getAllDrawingTimesForFunc(projection, funcId, device, experimentMode, data)
                        # times.extend(avg)
                        # iods.extend([getIodForFunc(projection, experimentMode, funcId)] * len(avg))                
        return times, iods
    
    def testRegressionAssumptions(self, reg, x, y, title):
        # plot the linearity assumption separately
        figure, axes = plt.subplots(1)
        figure.set_size_inches(6, 6)
        
        checkLinearAssumption(reg, x, y, axes, xy_minmax=(self.MIN_AVG_DRAW_TIME, self.MAX_AVG_DRAW_TIME))
        # reimplement saving figures
        # saveFigure(linearRegressionsFolderPath + "_assumptions_linearity_" + title.replace(' ', '_').replace('\n', ''))
        plt.show(figure)
        plt.close(figure)
        
        figure, axes = plt.subplots(1, 2)
        figure.set_size_inches(12, 6)
        normalErrorsAssumption(reg, x, y, axes[0], plotTextSize=self._a.plotTextSize)
        # multicollinearityAssumption(reg, x, y, axes[1, 0], feature_names=None)
        autocorrelationAssumption(reg, x, y)
        homoscedasticityAssumption(reg, x, y, axes[1], plotTextSize=self._a.plotTextSize)
        
        st = figure.suptitle(title)
    #     st = figure.suptitle('')
        figure.tight_layout(pad=2)
        # shift subplots down:
        st.set_y(1)
        figure.subplots_adjust(top=0.85)
        
        # saveFigure(linearRegressionsFolderPath + "_assumptions_residuals_" + title.replace(' ', '_').replace('\n', ''))
        plt.show(figure)
        plt.close(figure)


    # get x and y data for linear regression
    def getDataForRegression(self, projections, experimentModes, device):
        y, iods = np.array(self.getAvgsByFilter(projections, self._a.FUNC_IDS, [device], experimentModes))
        x = [[iod] for iod in iods]
        return x, y

    # training and retrieving the model 
    def getRegressionModel(self, projections, experimentModes, device, axes):
        print('\n\n##### TRAINING #####')

        x, y = self.getDataForRegression(projections, experimentModes, device)
        reg = LinearRegression().fit(x, y)
        title = getBasePlotTitle(projections, device, experimentModes, args=self._a)
        self.plotDataAndReg(reg, x, y, title, axes)
        printRegressionModelMetrics(reg, x, y)
        self.testRegressionAssumptions(reg, x, y, title)

        return reg

    def plotDataAndReg(self, reg, x, y, title, axes):
        MAX_IOD = self._iod.getMaxIodForPlot()

        predictX = np.linspace(0, MAX_IOD, 10)
        predictY = reg.predict([[x] for x in predictX])
        
        axes.plot(predictX, predictY, color="black")
        axes.scatter(x, y, color="turquoise", edgecolors="navy",)
        
        axes.set_ylabel("Drawing time [s]", fontsize=self._a.plotTextSize)
        axes.set_xlabel("Index of difficulty", fontsize=self._a.plotTextSize)
        
        axes.tick_params(axis='both', labelsize=self._a.plotTextSize)
        
        # if useCroatian is True:
        #     axes.set_ylabel("Vrijeme crtanja [s]", fontsize=self._a.plotTextSize)
        #     axes.set_xlabel("Indeks težine [bit]", fontsize=self._a.plotTextSize)        
        
        axes.set_ylim([0, self.MAX_AVG_DRAW_TIME])
        axes.set_xlim([0, MAX_IOD])
        axes.grid(True)
        print(title)
        axes.set_title(title, fontsize=self._a.plotTextSize * 1.1)
        # axes.axis('scaled')
        
        a, b = getRegressionCoefficients(reg)
        axesText = 'MT = %.3f + %.3f ID\nR^2 = %.3f' % (b, a, reg.score(x, y))
        axes.text(MAX_IOD * 0.05, self.MAX_AVG_DRAW_TIME * 0.95,
                axesText,
                size=self._a.plotTextSize * 1.1,
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=1)
        )
        

    def validateRegressionModel(self, reg, projections, experimentModes, device, axes):
        print('\n\n##### VALIDATION #####')
        
        x, y = self.getDataForRegression(projections, experimentModes, device)
        title = getBasePlotTitle(projections, device, experimentModes, args=self._a)
        
        self.plotDataAndReg(reg, x, y, title, axes)
        self.testRegressionAssumptions(reg, x, y, title)
        printRegressionModelMetrics(reg, x, y)
        
        
    def trainRegressionModelThenValidate(self, projections, device):
        (width, height) = 12, 5

        figure, axes = plt.subplots(1)
        figure.set_size_inches(width, height)
        reg = self.getRegressionModel(projections, [0], device, axes)
        title = getBasePlotTitle(projections, device, [0], args=self._a)
    #     axes.set_title('') # remove the default title
        axes.set_title(title)
        # save the training reg
        self.save_figure(fig=figure, filename=title.replace(' ', '_').replace('\n', ''))
        plt.show(figure)
        plt.close(figure)
        self.save_results(reg_model=reg, devices=device, projections=projections, test_modes=[0])

        figure, axes = plt.subplots(1)
        figure.set_size_inches(width, height)
        self.validateRegressionModel(reg, projections, [1], device, axes)
        title = getBasePlotTitle(projections, device, [1], args=self._a)
    #     axes.set_title('') # remove the default title
        axes.set_title(title)
        # save the validation reg
        self.save_figure(fig=figure, filename=title.replace(' ', '_').replace('\n', ''))
        plt.show(figure)
        plt.close(figure)
        self.save_results(reg_model=reg, devices=device, projections=projections, test_modes=[1])

    def get_workdir_folderpath(self):
        folderpath = os.path.join('_analysis_workdir', self._a.iodModelName)
        os.makedirs(folderpath, exist_ok=True)
        return folderpath
    
    def get_results_filepath(self):
        workdir = self.get_workdir_folderpath()
        resultspath = os.path.join(workdir, 'results.csv')
        return resultspath

    def save_figure(self, fig, filename: str):
        if (self._a.save_figures is True):
            folderpath = self.get_workdir_folderpath()
            fullfigname = os.path.join(folderpath, filename)
            fig.savefig(fullfigname)

    def save_results(
            self,
            reg_model: LinearRegression,
            devices: typing.List[DevicesType],
            projections: typing.List[ProjectionsType],
            test_modes: typing.List[TestModesType]
        ):
        x, y_true = self.getDataForRegression(projections=projections, experimentModes=test_modes, device=devices)
        a_coef, b_coef = getRegressionCoefficients(reg_model)
        mse, rmspe = getMseAndRmspe(reg=reg_model, x=x, y=y_true)
        r2 = reg_model.score(x, y_true)
        resultspath = self.get_results_filepath()
        if os.path.exists(resultspath):
            df = pd.read_csv(resultspath, index_col='entry_id')
        else:
            df = pd.DataFrame(columns=['modelname', 'r2', 'mse', 'rmspe', 'a_coef', 'b_coef', 'device', 'projection', 'test_mode', 'central_tendency', 'entry_id'])
            df.set_index('entry_id', inplace=True)

        # init an empty row
        entry_id = len(df)
        df.loc[entry_id] = 0
        # then populate this row with data
        df.loc[entry_id, 'modelname'] = self._a.iodModelName
        df.loc[entry_id, 'central_tendency'] = self._a.centralTendency
        df.loc[entry_id, 'r2'] = r2
        df.loc[entry_id, 'mse'] = mse
        df.loc[entry_id, 'rmspe'] = rmspe
        df.loc[entry_id, 'device'] = str(devices)
        df.loc[entry_id, 'projection'] = str(projections)
        df.loc[entry_id, 'a_coef'] = a_coef
        df.loc[entry_id, 'b_coef'] = b_coef
        df.loc[entry_id, 'test_mode'] = str(test_modes)

        df.drop_duplicates(inplace=True, ignore_index=True, subset=['projection', 'device', 'test_mode', 'central_tendency'], keep='last')
        df.to_csv(resultspath, index_label='entry_id')

        
