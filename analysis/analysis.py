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


class Analysis:
    _a: AnalysisArgs
    _iod: IodHandler
    _dt: DrawingTimesHandler

    MAX_AVG_DRAW_TIME: float
    MIN_AVG_DRAW_TIME: float


    def __init__(self, args: AnalysisArgs) -> None:
        self._a = args
        self._iod = IodHandler(args)
        self._dt = DrawingTimesHandler(args=args)

        # for plot xy ranges
        self.MAX_AVG_DRAW_TIME = round(max(self.getAvgsByFilter(self._a.PROJECTIONS, self._a.FUNC_IDS, self._a.DEVICES, self._a.TEST_MODES)[0]))
        self.MIN_AVG_DRAW_TIME = round(min(self.getAvgsByFilter(self._a.PROJECTIONS, self._a.FUNC_IDS, self._a.DEVICES, self._a.TEST_MODES)[0]))
        self.MAX_AVG_DRAW_TIME += 0.2 * self.MAX_AVG_DRAW_TIME
        self.MIN_AVG_DRAW_TIME -= 0.2 * self.MIN_AVG_DRAW_TIME


    def isDataNormallyDistributed(self, data, p_value_thresh=0.05):
        # Anderson-Darling test for normal distribution unknown mean and variance.
        # Or shapiro-wilk
        p_value = normal_shapiro(data)[1]
        print('p-value from the test - below 0.05 generally means non-normal:', p_value)
        
        # Reporting the normality of the residuals
        if p_value < p_value_thresh:
            print('Data is not normally distributed')
        else:
            print('Data is normally distributed')
        return p_value

    def getAvgsByFilter(self, projections:typing.List[ProjectionsType], funcIds: typing.List[FuncIdsType], devices: typing.List[DevicesType], experimentModes: typing.List[TestModesType]):
        times = []
        iods = []
        for experimentMode in experimentModes:
            for device in devices:
                for projection in projections:
                    for funcId in funcIds:
                        avg = self._dt.getAvgForFunc(projection, funcId, device, experimentMode)
                        times.append(avg)
                        iods.append(self._iod.getIodForFunc(projection, experimentMode, funcId))
                        # use this if you want to get ALL drawing times paired with iods
                        # avg = getAllDrawingTimesForFunc(projection, funcId, device, experimentMode, data)
                        # times.extend(avg)
                        # iods.extend([getIodForFunc(projection, experimentMode, funcId)] * len(avg))                
        return times, iods
    
    def calculateResiduals(self, model, x, y):
        """
        Creates predictions on the features with the model and calculates residuals
        """
        predictions = model.predict(x)
        residuals = abs(y) - abs(predictions)
        return residuals, predictions


    def checkLinearAssumption(self, model, x, y, axes):
        """
        Linearity: Assumes that there is a linear relationship between the predictors and
                the response variable. If not, either a quadratic term or another
                algorithm should be used.
        """
        print('Assumption 1: Linear Relationship between the Target and the Feature', '\n')
            
        print('Checking with a scatter plot of actual vs. predicted.',
            'Predictions should follow the diagonal line.')
        
        # Calculating residuals for the plot
        residuals, predictions = self.calculateResiduals(model, x, y)
                
        # Plotting the diagonal line
        lineCoords = np.arange(self.MIN_AVG_DRAW_TIME, self.MAX_AVG_DRAW_TIME)
        axes.plot(lineCoords, lineCoords, color="black")
        
        for actual_y, predicted_y in zip(y, predictions):
            axes.plot([predicted_y, predicted_y], [predicted_y, actual_y], color="darkgrey")
            pass
        
        # Plotting the actual vs predicted values
        axes.scatter(x=predictions, y=y, color="coral", edgecolors="grey")
        axes.axis("equal")
        axes.set_title("Actual vs Predicted")
        axes.set_ylabel("Actual")
        axes.set_xlabel("Predicted")
        axes.tick_params(axis='both', labelsize=self._a.plotTextSize)
            
        axes.grid(True)
        
        mspe, rmspe = self.getMseAndRmspe(model, x, y)
        axesText = 'MSE = %.3f\nRMSPE = %.3f%%' % (mspe, rmspe) 
        axes.text(self.MIN_AVG_DRAW_TIME * 1.05, self.MAX_AVG_DRAW_TIME * 0.95,
                axesText,
                size=self._a.plotTextSize,
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=1)
        )
        axes.set_ylim([self.MIN_AVG_DRAW_TIME, self.MAX_AVG_DRAW_TIME])
        axes.set_xlim([self.MIN_AVG_DRAW_TIME, self.MAX_AVG_DRAW_TIME])
        return axes

    def normalErrorsAssumption(self, model, x, y, axes, p_value_thresh=0.05):
        """
        Normality: Assumes that the error terms are normally distributed. If they are not,
        nonlinear transformations of variables may solve this.
                
        This assumption being violated primarily causes issues with the confidence intervals
        """
        print('Assumption 2: The error terms are normally distributed', '\n')
        
        # Calculating residuals for the Anderson-Darling test
        residuals, predictions = self.calculateResiduals(model, x, y)
        
        print('Using the Anderson-Darling test for normal distribution')

        # Performing the test on the residuals
        # p_value = normal_ad(residuals)[1]
        p_value = self.isDataNormallyDistributed(residuals)
        print('p-value from the test - below 0.05 generally means non-normal:', p_value)
        
        # Reporting the normality of the residuals
        if p_value < p_value_thresh:
            print('Residuals are not normally distributed')
        else:
            print('Residuals are normally distributed')
        
        # Plotting the residuals distribution
        axes.set_title('Distribution of Residuals', fontsize=self._a.plotTextSize)
        axes.set_ylabel('Broj', fontsize=self._a.plotTextSize)
        axes.tick_params(axis='both', labelsize=self._a.plotTextSize)
        sns.histplot(residuals, ax=axes)
        
        print()
        if p_value > p_value_thresh:
            print('Assumption satisfied')
        else:
            print('Assumption not satisfied')
            print()
            print('Confidence intervals will likely be affected')
            print('Try performing nonlinear transformations on variables')

            
    def multicollinearityAssumption(model, x, y, axes, feature_names=None):
        """
        Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                        correlation among the predictors, then either remove prepdictors with high
                        Variance Inflation Factor (VIF) values or perform dimensionality reduction
                            
                        This assumption being violated causes issues with interpretability of the 
                        coefficients and the standard errors of the coefficients.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        print('Assumption 3: Little to no multicollinearity among predictors')
            
        # Plotting the heatmap
        sns.heatmap(pd.DataFrame(x, columns=feature_names).corr(), annot=True, ax=axes)
        axes.set_title('Correlation of Variables')
            
        print('Variance Inflation Factors (VIF)')
        print('> 10: An indication that multicollinearity may be present')
        print('> 100: Certain multicollinearity among the variables')
        print('-------------------------------------')
        
        # Gathering the VIF for each variable
        # 30.8.2023: idk what 'features' is
        VIF = [variance_inflation_factor(features, i) for i in range(x.shape[1])]
        for idx, vif in enumerate(VIF):
            print('{0}: {1}'.format(feature_names[idx], vif))
            
        # Gathering and printing total cases of possible or definite multicollinearity
        possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
        definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
        print()
        print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
        print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
        print()

        if definite_multicollinearity == 0:
            if possible_multicollinearity == 0:
                print('Assumption satisfied')
            else:
                print('Assumption possibly satisfied')
                print()
                print('Coefficient interpretability may be problematic')
                print('Consider removing variables with a high Variance Inflation Factor (VIF)')

        else:
            print('Assumption not satisfied')
            print()
            print('Coefficient interpretability will be problematic')
            print('Consider removing variables with a high Variance Inflation Factor (VIF)')

            
    def autocorrelationAssumption(self, model, x, y):
        """
        Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                        autocorrelation, then there is a pattern that is not explained due to
                        the current value being dependent on the previous value.
                        This may be resolved by adding a lag variable of either the dependent
                        variable or some of the predictors.
        """
        from statsmodels.stats.stattools import durbin_watson
        print('Assumption 4: No Autocorrelation', '\n')
        
        # Calculating residuals for the Durbin Watson-tests
        residuals, predictions = self.calculateResiduals(model, x, y)

        print('\nPerforming Durbin-Watson Test')
        print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
        print('0 to 2< is positive autocorrelation')
        print('>2 to 4 is negative autocorrelation')
        print('-------------------------------------')
        durbinWatson = durbin_watson(residuals)
        print('Durbin-Watson:', durbinWatson)
        if durbinWatson < 1.5:
            print('Signs of positive autocorrelation', '\n')
            print('Assumption not satisfied')
        elif durbinWatson > 2.5:
            print('Signs of negative autocorrelation', '\n')
            print('Assumption not satisfied')
        else:
            print('Little to no autocorrelation', '\n')
            print('Assumption satisfied')

    def homoscedasticityAssumption(self, model, x, y, axes):
        """
        Homoscedasticity: Assumes that the errors exhibit constant variance
        """
        print('Assumption 5: Homoscedasticity of Error Terms', '\n')
        
        print('Residuals should have relative constant variance')
            
        # Calculating residuals for the plot
        residuals, predictions = self.calculateResiduals(model, x, y)

        # Plotting the residuals
        indices = np.arange(0, len(residuals))
        axes.scatter(x=indices, y=residuals, alpha=0.5)
        axes.plot(np.repeat(0, np.max(indices) + 1), color='darkorange', linestyle='--')
        axes.set_title('Residuals', fontsize=self._a.plotTextSize)
        axes.tick_params(axis='both', labelsize=self._a.plotTextSize)
        axes.set_ylim([-5, 5])
    
    def testRegressionAssumptions(self, reg, x, y, title):
        # plot the linearity assumption separately
        figure, axes = plt.subplots(1)
        figure.set_size_inches(6, 6)
        self.checkLinearAssumption(reg, x, y, axes)
        # reimplement saving figures
        # saveFigure(linearRegressionsFolderPath + "_assumptions_linearity_" + title.replace(' ', '_').replace('\n', ''))
        plt.close(figure)
        
        figure, axes = plt.subplots(1, 2)
        figure.set_size_inches(12, 6)
        self.normalErrorsAssumption(reg, x, y, axes[0])
        # multicollinearityAssumption(reg, x, y, axes[1, 0])
        self.autocorrelationAssumption(reg, x, y)
        self.homoscedasticityAssumption(reg, x, y, axes[1])
        
        st = figure.suptitle(title)
    #     st = figure.suptitle('')
        figure.tight_layout(pad=2)
        # shift subplots down:
        st.set_y(1)
        figure.subplots_adjust(top=0.85)
        
        # saveFigure(linearRegressionsFolderPath + "_assumptions_residuals_" + title.replace(' ', '_').replace('\n', ''))
        plt.close(figure)


    def getBasePlotTitle(self, projections, device, experimentModes):
        # if self._a.useCroatian is True:
        #     return "Linearna regresija za %s sustav, %s, \nIndeks težine = %s, faza = %s" \
        #                    %(translate(projections), translate(device), iodModelName, experimentModes[0] + 1)
            
            
        return "Linear regression for %s, %s, mode=%s,\nCentral Tendency=%s,Index Of Difficulty=%s" \
                    %(projections, device, experimentModes, self._a.centralTendency, self._a.iodModelName)

    def getRegressionCoefficients(self, reg):
        x1 = 0
        x2 = 1
        y1, y2 = reg.predict(np.array([[x1], [x2]]))
        # Coefficients: y = ax + b
        b = y1
        a = (y2 - y1) / (x2 - x1)
        return a, b

    # get x and y data for linear regression
    def getDataForRegression(self, projections, experimentModes, device):
        y, iods = np.array(self.getAvgsByFilter(projections, self._a.FUNC_IDS, [device], experimentModes))
        x = [[iod] for iod in iods]
        return x, y

    def getMseAndRmspe(self, reg, x, y):
        y_predicted = reg.predict(x)
        mse = metrics.mean_squared_error(y_predicted, y)
        rmspe = (np.sqrt(np.mean(np.square((y_predicted - y) / y_predicted)))) * 100
        return mse, rmspe

    def getFormattedRegressionMetrics(self, reg, x, y):
        a, b = self.getRegressionCoefficients(reg)
        y_predicted = reg.predict(x)
        mse, rmspe = self.getMseAndRmspe(reg, x, y)
        print(rmspe)
        return 'MT = %.3fx + %.3f\nR^2 = %.3f\nMSE = %.3f\nRMSPE = %.3f%%' % (a, b, reg.score(x, y), np.sqrt(mse), rmspe)

    # model metrics :: for evaulating the regression model
    def printRegressionModelMetrics(self, reg, x, y):
        print(self.getFormattedRegressionMetrics(reg, x, y))

    # training and retrieving the model 
    def getRegressionModel(self, projections, experimentModes, device, axes):
        x, y = self.getDataForRegression(projections, experimentModes, device)
        reg = LinearRegression().fit(x, y)
        title = self.getBasePlotTitle(projections, device, experimentModes)
        self.plotDataAndReg(reg, x, y, title, axes)
        self.printRegressionModelMetrics(reg, x, y)
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
        
        a, b = self.getRegressionCoefficients(reg)
        axesText = 'MT = %.3f + %.3f ID\nR^2 = %.3f' % (b, a, reg.score(x, y))
        axes.text(MAX_IOD * 0.05, self.MAX_AVG_DRAW_TIME * 0.95,
                axesText,
                size=self._a.plotTextSize * 1.1,
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=1)
        )
        

    def validateRegressionModel(self, reg, projections, experimentModes, device, axes):  
        x, y = self.getDataForRegression(projections, experimentModes, device)
        title = self.getBasePlotTitle(projections, device, experimentModes)
        
        self.testRegressionAssumptions(reg, x, y, title)
        self.plotDataAndReg(reg, x, y, title, axes)
        self.printRegressionModelMetrics(reg, x, y)
        
        
    def trainRegressionModelThenValidate(self, projections, device):
        print("\n\nSKLEARN")
        MAX_IOD = self._iod.getMaxIodForPlot()
        (width, height) = 12, 5

        figure, axes = plt.subplots(1)
        figure.set_size_inches(width, height)
        
        reg = self.getRegressionModel(projections, [0], device, axes)
        title = self.getBasePlotTitle(projections, device, [0])
    #     axes.set_title('') # remove the default title
        axes.set_title(title) # remove the default title
        # saveFigure(linearRegressionsFolderPath + title.replace(' ', '_').replace('\n', ''))
        plt.show(figure)
        
        figure, axes = plt.subplots(1)
        figure.set_size_inches(width, height)
    
        self.validateRegressionModel(reg, projections, [1], device, axes)
        title = self.getBasePlotTitle(projections, device, [1])
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
        
