import numpy as np


class Scenario():
    def __init__(self, chunk_index, scenario_row):
        self.chunk_index = chunk_index
        self.scenario_index = scenario_row["scenario_index"]
        self.scenario_name = scenario_row["scenario_name"]
        self.scenario_path = scenario_row["scenario_path"]
        self.round = scenario_row["round"]
        self.cognitive_task = scenario_row["cognitive_task"]
        self.start_time = scenario_row["start_time"]
        self.end_time = scenario_row["end_time"]
        self.duration_s = scenario_row["duration_s"]
        self.duration_frames = scenario_row["duration_frames"]
        self.participant_name = scenario_row["participant_name"]
        self.intoxicated = scenario_row["intoxicated"]
        self.experiment_version = scenario_row["experiment_version"]
        self.number_all_crashes = scenario_row["number_all_crashes"] if scenario_row["number_all_crashes"] != -1 else np.nan
        self.number_hazard_crashes = scenario_row["number_hazard_crashes"] if scenario_row["number_hazard_crashes"] != -1 else np.nan
        self.anomalies = scenario_row["anomalies"]
            
    def get_resample_times(self, fps):
        if fps == 0:
            return np.array([self.start_time, self.end_time])
        resample_times = np.array([x // fps for x in list(range(int(self.start_time * fps), int(self.end_time * fps), int(1e9)))])
        resample_times -= resample_times[0]
        return resample_times
        
    @staticmethod
    def list_from_table(df):
        scenarios = []
        for scenario_it in range(df.shape[0]):
            scenario_row = df.iloc[[scenario_it]].to_dict(orient='records')[0]
            scenarios.append(Scenario(scenario_it, scenario_row))
        return scenarios
    
    def __str__(self):
        return f"(Participant: {self.participant_name}, Scenario: {self.scenario_name})"
    
class CollatedScenario():
    def __init__(self, list_of_scenarios, config):    
        self.chunk_index = np.array([x.chunk_index for x in list_of_scenarios])
        self.scenario_index = np.array([x.scenario_index for x in list_of_scenarios])
        self.scenario_name = np.array([x.scenario_name for x in list_of_scenarios])
        self.scenario_path = np.array([x.scenario_path for x in list_of_scenarios])
        self.round = np.array([x.round for x in list_of_scenarios])
        self.cognitive_task = np.array([x.cognitive_task for x in list_of_scenarios])
        self.start_time = np.array([x.start_time for x in list_of_scenarios])
        self.end_time = np.array([x.end_time for x in list_of_scenarios])
        self.duration_s = np.array([x.duration_s for x in list_of_scenarios])
        self.duration_frames = np.array([x.duration_frames for x in list_of_scenarios])
        self.participant_name = np.array([x.participant_name for x in list_of_scenarios])
        self.intoxicated = np.array([x.intoxicated for x in list_of_scenarios])
        self.experiment_version = np.array([x.experiment_version for x in list_of_scenarios])
        self.number_all_crashes = np.array([x.number_all_crashes for x in list_of_scenarios])
        self.number_hazard_crashes = np.array([x.number_hazard_crashes for x in list_of_scenarios])
        self.anomalies = [x.anomalies for x in list_of_scenarios]
    
    def __str__(self):
        return f"(CollatedScenario Batch Size {self.chunk_index.shape[0]})"
    