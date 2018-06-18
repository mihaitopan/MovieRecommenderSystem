import pandas as pd
import numpy as np
import scipy.optimize as sc
import random


class CrossCollaborativeFiltering:
    def __init__(self, dataset, trainSetPath, noCross=5, noFeatures=100, noMaxIterations=100000, lambdaCoeff=0.01):
        # necessary constants
        self._dataset = dataset
        self._trainSetPath = trainSetPath
        self._noCross = noCross

        # dataset constants
        self._trainSetName = "crossSet"
        self._testSetRatingsName = "Ratings"
        self._trainSetResultName = "Results"
        self._trainSetExt = ".csv"
        self._trainResultsName = "Recommendations"

        # algorithm constants
        self._noFeatures = noFeatures
        self._noMaxIterations = noMaxIterations
        self._lambdaCoeff = lambdaCoeff


    def _readData(self, noMaxUsers=1000, noMaxMovies=10000):
        Ratings = pd.read_csv(self._dataset, low_memory=False)

        noUniqueMovies = Ratings[["movieId"]].drop_duplicates().size
        assert (noUniqueMovies < noMaxMovies)
        noUniqueUsers = Ratings[["userId"]].drop_duplicates().size
        assert (noUniqueUsers < noMaxUsers)

        values = Ratings[["userId", "movieId", "rating"]].values
        RatingsArray = np.zeros(shape=(noUniqueMovies, noUniqueUsers))
        for i in range(0, values.shape[0]):
            user = np.int64(values[i][0])
            movie = np.int64(values[i][1])
            rating = values[i][2]
            assert (RatingsArray[movie][user] == 0)
            assert (rating >= 0)
            if rating == 0:
                rating = 0.01
            RatingsArray[movie][user] = rating

        return RatingsArray


    def _saveCrossSets(self, T, RatingsArray):
        for idx in range(0, self._noCross):
            trainSet = pd.DataFrame.from_records(T[idx])
            trainSet.to_csv(self._trainSetPath + self._trainSetName + str(idx) + self._trainSetExt)

            testSetArray = RatingsArray.copy()
            testSetArray = testSetArray * T[idx]
            testSet = pd.DataFrame.from_records(testSetArray)
            testSet.to_csv(
                self._trainSetPath + self._trainSetName + str(idx) + self._testSetRatingsName + self._trainSetExt)


    def _saveCrossResults(self, T, idx, RatingsArray):
        testResultsSet = pd.DataFrame.from_records(T)
        testResultsSet.to_csv(
            self._trainSetPath + self._trainSetName + str(idx) + self._trainSetResultName + self._trainSetExt)

        recommendationsSet = pd.DataFrame.from_records(RatingsArray)
        recommendationsSet.to_csv(
            self._trainSetPath + self._trainSetName + str(idx) + self._trainResultsName + self._trainSetExt)


    def _costFunction(self, params, Y, R, noMovies, noUsers, noFeatures, lambdaCoeff):
        """
        :param params: X(movies x features) and Theta(users x features) reshaped
        :param Y: movies x users(matrix of user ratings of movies)
        :param R: movies x users with R(m, u) = 1 if movie m was rated by user u
        :param lambdaCoeff: learning rate
        """

        '''unpack X and theta'''
        X = np.reshape(params[:noMovies * noFeatures], (noMovies, noFeatures), order='F')
        Theta = np.reshape(params[noMovies * noFeatures:], (noUsers, noFeatures), order='F')

        sqrdErrors = (np.matmul(X, Theta.transpose()) - Y) ** 2
        '''cost J'''
        J = (1 / 2) * np.sum(sqrdErrors * R) + \
            (lambdaCoeff / 2) * np.sum(X ** 2) + (lambdaCoeff / 2) * np.sum(Theta ** 2)

        '''movie to users matrix'''
        ratings = (np.matmul(X, Theta.transpose()) - Y) * R
        '''X gradient'''
        Xgrad = np.matmul(ratings, Theta) + lambdaCoeff * X
        '''theta gradient'''
        Thetagrad = np.matmul(ratings.transpose(), X) + lambdaCoeff * Theta

        '''pack X and theta'''
        grad = np.concatenate((Xgrad.reshape(Xgrad.size, order='F'), Thetagrad.reshape(Thetagrad.size, order='F')))
        return J, grad

    def _trainSet(self, RatingsArray, R, T):
        noMovies = len(RatingsArray)
        noUsers = len(RatingsArray[0])

        # generate random X and Theta
        moviesToFeatures = np.random.rand(noMovies, self._noFeatures)
        usersToFeatures = np.random.rand(noUsers, self._noFeatures)

        # normalize ratings
        meanList = []
        ratingsNormalized = []
        for movieRating in RatingsArray:
            newMovieRating = movieRating.copy()
            movieMean = np.sum(newMovieRating[newMovieRating > 0]) / np.count_nonzero(newMovieRating)
            meanList.append(movieMean)
            newMovieRating[movieRating > 0] = movieRating[movieRating > 0] - movieMean
            ratingsNormalized.append(newMovieRating.tolist())
        # RatingsArray
        Y = np.array(ratingsNormalized)

        # pack X and Theta into params
        X = moviesToFeatures.copy()
        Theta = usersToFeatures.copy()
        params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

        # shortcut for the cost function
        def costFunc(params):
            return self._costFunction(params, Y, R, noMovies, noUsers, self._noFeatures, self._lambdaCoeff)

        # minimize cost function
        results = sc.minimize(costFunc,
                              x0=params,
                              options={'disp': True, 'maxiter': self._noMaxIterations},
                              method="L-BFGS-B",
                              jac=True)
        # unfold the results
        finalParams = results["x"]
        X = np.reshape(finalParams[:noMovies * self._noFeatures], (noMovies, self._noFeatures), order='F')
        Theta = np.reshape(finalParams[noMovies * self._noFeatures:], (noUsers, self._noFeatures), order='F')

        # get ratings result
        RatingsResult = np.matmul(X, Theta.transpose())

        # add the mean back to the ratings result
        movieIdx = 0
        resultNormalised = []
        for movieRating in RatingsResult:
            newMovieRating = movieRating.copy()
            newMovieRating[movieRating > 0] = movieRating[movieRating > 0] + meanList[movieIdx]
            resultNormalised.append(newMovieRating.tolist())
            movieIdx += 1
        RatingsResult = np.array(resultNormalised)

        # calibrate the ratings result
        RatingsResult = abs(RatingsResult)
        RatingsResult[RatingsResult > 5] = 5

        # get the ratings for the given test set
        testResults = RatingsResult.copy()
        testResults = testResults * T

        return resultNormalised, testResults


    def train(self):
        # read data
        ratingsArray = self._readData()

        # TestSet - array of cross valid sets
        TestSet = []
        for _ in range(0, self._noCross):
            Te = ratingsArray.copy()
            Te[Te > 0] = 0
            TestSet.append(Te)

        # R - 1 where rating
        R = ratingsArray.copy()
        R[R > 0] = 1
        noRatings = R.sum()

        # get random (movieId, userId) pairs (with rating)
        haveBeenRated = list((np.argwhere(R)).tolist())
        random.shuffle(haveBeenRated)
        noRated = len(haveBeenRated)
        assert (noRatings == noRated)

        # get random cross validation sets
        for e in range(0, self._noCross):
            for i in range(e, noRated, self._noCross):
                row, column = haveBeenRated[i]
                assert(R[row][column] == 1)
                TestSet[e][row][column] = 1
                R[row][column] = 0
        assert(R.sum() == 0)

        # flush to disk test sets
        self._saveCrossSets(TestSet, ratingsArray)

        # for each test, get train sets and run algorithm
        for i in range(0, self._noCross):
            T = TestSet[i]
            R = ratingsArray.copy()
            R[R > 0] = 0
            for j in range(0, self._noCross):
                if j != i:
                    R = R + TestSet[j]

            moviesToUsersResult, testResults = self._trainSet(ratingsArray, R, T)

            # flush to disk the results
            self._saveCrossResults(testResults, i, moviesToUsersResult)


    def test(self):
        crossMAEMean = 0
        crossRMSDMean = 0
        for idx in range(0, self._noCross):
            # read data into arrays
            testRatingsSet = pd.read_csv(self._trainSetPath + self._trainSetName + str(idx)
                                                 + self._testSetRatingsName + self._trainSetExt, low_memory=False)
            testResultsSet = pd.read_csv(self._trainSetPath + self._trainSetName + str(idx) +
                                         self._trainSetResultName + self._trainSetExt, low_memory=False)

            testRatingsSet = testRatingsSet.drop(testRatingsSet.columns[[0]], 1)
            testResultsSet = testResultsSet.drop(testResultsSet.columns[[0]], 1)

            testRatings = testRatingsSet.values
            testResults = testResultsSet.values

            # get RMSD for each test set
            testDiff = abs(testRatings - testResults)
            mae = testDiff[testDiff > 0].mean()
            crossMAEMean += mae
            rmsd = (np.diff(testDiff[testDiff > 0]) ** 2).mean() ** 0.5
            crossRMSDMean += rmsd

        # return RMSD and Accuracy
        crossMAEMean = crossMAEMean / self._noCross
        crossRMSDMean = crossRMSDMean / self._noCross
        crossAccuracy = 1 - crossRMSDMean / 5
        return crossMAEMean, crossRMSDMean, crossAccuracy

