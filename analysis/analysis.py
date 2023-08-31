from analysisargs import AnalysisArgs
from analysisargs import Logs
from analysisargs import ProjectionsType
from analysisargs import DevicesType
from analysisargs import FuncIdsType
from analysisargs import TestModesType
import numpy as np
import typing
from iod import IodHandler
from drawingtimes import DrawingTimesHandler
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
from statutils import getRegressionCoefficients, normalErrorsAssumption, homoscedasticityAssumption

class Analysis:
    _a: AnalysisArgs
    _iod: IodHandler
    dt: DrawingTimesHandler

    MAX_AVG_DRAW_TIME: float
    MIN_AVG_DRAW_TIME: float


    def __init__(self, args: AnalysisArgs, logs: Logs = None) -> None:
        self._a = args
        self._iod = IodHandler(args)
        self.dt = DrawingTimesHandler(args=args, logs=logs)

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
        checkLinearAssumption(reg, x, y, axes)
        # reimplement saving figures
        # saveFigure(linearRegressionsFolderPath + "_assumptions_linearity_" + title.replace(' ', '_').replace('\n', ''))
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
        plt.close(figure)


    # get x and y data for linear regression
    def getDataForRegression(self, projections, experimentModes, device):
        y, iods = np.array(self.getAvgsByFilter(projections, self._a.FUNC_IDS, [device], experimentModes))
        x = [[iod] for iod in iods]
        return x, y

    # training and retrieving the model 
    def getRegressionModel(self, projections, experimentModes, device, axes):
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
        x, y = self.getDataForRegression(projections, experimentModes, device)
        title = getBasePlotTitle(projections, device, experimentModes, args=self._a)
        
        self.testRegressionAssumptions(reg, x, y, title)
        self.plotDataAndReg(reg, x, y, title, axes)
        printRegressionModelMetrics(reg, x, y)
        
        
    def trainRegressionModelThenValidate(self, projections, device):
        print("\n\nSKLEARN")
        MAX_IOD = self._iod.getMaxIodForPlot()
        (width, height) = 12, 5

        figure, axes = plt.subplots(1)
        figure.set_size_inches(width, height)
        
        reg = self.getRegressionModel(projections, [0], device, axes)
        title = getBasePlotTitle(projections, device, [0], args=self._a)
    #     axes.set_title('') # remove the default title
        axes.set_title(title) # remove the default title
        # saveFigure(linearRegressionsFolderPath + title.replace(' ', '_').replace('\n', ''))
        plt.show(figure)
        
        figure, axes = plt.subplots(1)
        figure.set_size_inches(width, height)
    
        self.validateRegressionModel(reg, projections, [1], device, axes)
        title = getBasePlotTitle(projections, device, [1], args=self._a)
    #     axes.set_title('') # remove the default title
        axes.set_title(title) # remove the default title

        # figure.tight_layout(pad=2)
        # saveFigure(linearRegressionsFolderPath + title.replace(' ', '_').replace('\n', ''))
        
        plt.show(figure)
        
        # print("\n\nSTATSMODELS")
        # x, y = getDataForRegression(projections, [0], device)
        # constx = sm.add_constant(x)
        # reg = sm.OLS(y, constx).fit()
        # print(reg.summary())
        
