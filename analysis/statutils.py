from analysisargs import AnalysisArgs
import numpy as np
import typing
from scipy.stats import shapiro as normal_shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression


def isDataNormallyDistributed(data, p_value_thresh=0.05):
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

def calculateResiduals(model, x, y):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(x)
    residuals = abs(y) - abs(predictions)
    return residuals, predictions

            
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

        
def autocorrelationAssumption(model, x, y):
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
    residuals, predictions = calculateResiduals(model, x, y)

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

def homoscedasticityAssumption(model, x, y, axes, plotTextSize=18):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
    print('Assumption 5: Homoscedasticity of Error Terms', '\n')
    
    print('Residuals should have relative constant variance')
        
    # Calculating residuals for the plot
    residuals, predictions = calculateResiduals(model, x, y)

    # Plotting the residuals
    indices = np.arange(0, len(residuals))
    axes.scatter(x=indices, y=residuals, alpha=0.5)
    axes.plot(np.repeat(0, np.max(indices) + 1), color='darkorange', linestyle='--')
    axes.set_title('Residuals', fontsize=plotTextSize)
    axes.tick_params(axis='both', labelsize=plotTextSize)
    axes.set_ylim([-5, 5])

def normalErrorsAssumption(model, x, y, axes, p_value_thresh=0.05, plotTextSize=18):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.
            
    This assumption being violated primarily causes issues with the confidence intervals
    """
    print('Assumption 2: The error terms are normally distributed', '\n')
    
    # Calculating residuals for the Anderson-Darling test
    residuals, predictions = calculateResiduals(model, x, y)
    
    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    # p_value = normal_ad(residuals)[1]
    p_value = isDataNormallyDistributed(residuals)
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')
    
    # Plotting the residuals distribution
    axes.set_title('Distribution of Residuals', fontsize=plotTextSize)
    axes.set_ylabel('Broj', fontsize=plotTextSize)
    axes.tick_params(axis='both', labelsize=plotTextSize)
    sns.histplot(residuals, ax=axes)
    
    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
    print('Try performing nonlinear transformations on variables')



def getBasePlotTitle(projections, device, experimentModes, args:AnalysisArgs):
    # if self._a.useCroatian is True:
    #     return "Linearna regresija za %s sustav, %s, \nIndeks težine = %s, faza = %s" \
    #                    %(translate(projections), translate(device), iodModelName, experimentModes[0] + 1)
        
        
    return "Linear regression for %s, %s, mode=%s,\nCentral Tendency=%s,Index Of Difficulty=%s" \
                %(projections, device, experimentModes, args.centralTendency, args.iodModelName)

def getRegressionCoefficients(reg):
    x1 = 0
    x2 = 1
    y1, y2 = reg.predict(np.array([[x1], [x2]]))
    # Coefficients: y = ax + b
    b = y1
    a = (y2 - y1) / (x2 - x1)
    return a, b


def getMseAndRmspe(reg, x, y):
    y_predicted = reg.predict(x)
    mse = metrics.mean_squared_error(y_predicted, y)
    rmspe = (np.sqrt(np.mean(np.square((y_predicted - y) / y_predicted)))) * 100
    return mse, rmspe

def getFormattedRegressionMetrics(reg, x, y):
    a, b = getRegressionCoefficients(reg)
    y_predicted = reg.predict(x)
    mse, rmspe = getMseAndRmspe(reg, x, y)
    print(rmspe)
    return 'MT = %.3fx + %.3f\nR^2 = %.3f\nMSE = %.3f\nRMSPE = %.3f%%' % (a, b, reg.score(x, y), np.sqrt(mse), rmspe)

# model metrics :: for evaulating the regression model
def printRegressionModelMetrics( reg, x, y):
    print(getFormattedRegressionMetrics(reg, x, y))

def checkLinearAssumption(model, x, y, axes, plotTextSize=18, xy_minmax=(0, 5)):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
            the response variable. If not, either a quadratic term or another
            algorithm should be used.
    """
    print('Assumption 1: Linear Relationship between the Target and the Feature', '\n')
        
    print('Checking with a scatter plot of actual vs. predicted.',
        'Predictions should follow the diagonal line.')
    
    # Calculating residuals for the plot
    residuals, predictions = calculateResiduals(model, x, y)
            
    # Plotting the diagonal line
    MIN_AVG_DRAW_TIME, MAX_AVG_DRAW_TIME = xy_minmax
    lineCoords = np.arange(MIN_AVG_DRAW_TIME, MAX_AVG_DRAW_TIME)
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
    axes.tick_params(axis='both', labelsize=plotTextSize)
        
    axes.grid(True)
    
    mspe, rmspe = getMseAndRmspe(model, x, y)
    axesText = 'MSE = %.3f\nRMSPE = %.3f%%' % (mspe, rmspe) 
    axes.text(MIN_AVG_DRAW_TIME * 1.05, MAX_AVG_DRAW_TIME * 0.95,
            axesText,
            size=plotTextSize,
            ha='left', va='top',
            bbox=dict(facecolor='white', alpha=1)
    )
    axes.set_ylim([MIN_AVG_DRAW_TIME, MAX_AVG_DRAW_TIME])
    axes.set_xlim([MIN_AVG_DRAW_TIME, MAX_AVG_DRAW_TIME])
    return axes


