class ModelStatistics(object):
    def __init__(self, model, tunningRun, testAcc, trainAcc, avgTestAcc, avgTrainAcc, bestParams, obs):
        self.model = model
        self.tunningRun = tunningRun
        self.testAcc = testAcc
        self.trainAcc = trainAcc
        self.avgTestAcc = avgTestAcc
        self.avgTrainAcc = avgTrainAcc
        self.bestParams = bestParams
        self.obs = obs

    def to_dict(self):
        return {
            'Model': self.model,
            'TunningRun': self.tunningRun,
            'TestAcc(%)': self.testAcc,
            'TrainAcc(%)': self.trainAcc,
            'AvgTestAcc(%)': self.avgTestAcc,
            'AvgTrainAcc(%)': self.avgTrainAcc,
            'BestParams': self.bestParams,
            'Obs': self.obs
        }