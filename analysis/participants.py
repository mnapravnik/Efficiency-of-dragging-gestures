from analysisargs import Logs, AnalysisArgs
import matplotlib.pyplot as plt
import numpy as np


class ParticipantAnalysis:
    _a: AnalysisArgs
    _l: Logs

    def __init__(self, logs: Logs) -> None:
        self._l = logs
        pass

    def get_unique_participant_data(self):
        """Will return a dataframe of unique participant data, i.e.
        with no duplicate entries by participant.
        Note that in this case, the returned data will contain useful values only in
        participant-related fields, (Age, name, handedness..)

        Returns:
            pandas Dataframe with unique values for each participant.
        """
        unique_participants = self._l.df.drop_duplicates(subset=['Participant name'], keep='first')
        return unique_participants

    def get_age_histogram(self, axes: plt.Axes = None):
        """Plot a histogram of participant age.

        Args:
            axes (plt.Axes, optional): If provided, the histogram will be plotted on this Axis.
                If None, a new axis and figure will be created.
                Defaults to None.

        Returns:
            plt.Axes: Axes on which the histogram was plotted.
        """
        # first, there are duplicate entries for participants
        # because each participant went through each curve twice.
        # so drop duplicate entries
        unique_participants = self.get_unique_participant_data()
        ages = unique_participants['Participant age']

        if axes is None:
            fig = plt.figure()
            axes = fig.gca()

        axes.hist(ages.values, color="lightcoral", bins=15, edgecolor="coral")
        axes.set_xlabel('Age')
        axes.set_ylabel('Frequency')
        axes.set_title('Participant age distribution')

        print(f'{"Particpant age mean":<25}: {np.mean(ages.values):7.3f}')
        print(f'{"Particpant age median":<25}: {np.median(ages.values):7.3f}')
        print(f'{"Particpant age stdev":<25}: {np.std(ages.values):7.3f}')

        return axes

    def get_handedness_histogram(self, axes: plt.Axes = None):
        unique_participants = self.get_unique_participant_data()
        handedness = unique_participants['Participant handedness']

        if axes is None:
            fig = plt.figure()
            axes = fig.gca()

        axes.hist(handedness.values, color="lightblue", bins=15, edgecolor="blue")
        axes.set_xlabel('Handedness')
        axes.set_ylabel('Frequency')
        axes.set_title('Participant handedness distribution')
        print(handedness.value_counts())

        return axes
    
    def get_device_experience(self):
        """Run an analysis of previous experience the participants
        had with each device, whether or not they're expert users.

        Returns:
            _type_: _description_
        """
        unique_participants = self.get_unique_participant_data()
        mouse_exp = unique_participants['Expert Mouse User']
        gt_exp = unique_participants['Expert Graphic Tablet User']

        print('** Expert mouse users (0=Not expert, 1=Is Expert):')
        print(mouse_exp.value_counts())
        print('** Expert GT users (0=Not expert, 1=Is Expert):')
        print(gt_exp.value_counts())

        pass
