import pandas as pd
import numpy as np
import scipy.optimize as sc
import heapq
from sys import exit


class CollaborativeFiltering:
    def __init__(self, dataset, trainSetPath, noFeatures=100, noMaxIterations=100000, lambdaCoeff=0.01):
        # necessary constants
        self._dataset = dataset
        self._trainSetPath = trainSetPath

        # dataset constants
        self._ratingsSetName = "CollaborativeRatings"
        self._featuresSetName = "CollaborativeFeatures"
        self._trainSetExt = ".csv"

        # algorithm constants
        self._noFeatures = noFeatures
        self._noMaxIterations = noMaxIterations
        self._lambdaCoeff = lambdaCoeff


    def _readData(self, noMaxUsers=10000, noMaxMovies=10000):
        Ratings = pd.read_csv(self._dataset, low_memory=False)

        noUniqueMovies = Ratings[["movieId"]].drop_duplicates().size
        # assert (noUniqueMovies < noMaxMovies)
        if noUniqueMovies > noMaxMovies:
            print("dataset exceeds maximum size (noMaxMovies=10000)")
            exit()
        noUniqueUsers = Ratings[["userId"]].drop_duplicates().size
        # assert (noUniqueUsers < noMaxUsers)
        if noUniqueUsers > noMaxUsers:
            print("dataset exceeds maximum size (noMaxUsers=10000)")
            exit()

        values = Ratings[["userId", "movieId", "rating"]].values
        RatingsArray = np.zeros(shape=(noUniqueMovies, noUniqueUsers))
        for i in range(0, values.shape[0]):
            user = np.int64(values[i][0])
            movie = np.int64(values[i][1])
            rating = values[i][2]
            # assert (RatingsArray[movie][user] == 0)
            # assert (rating >= 0)
            if rating == 0:
                rating = 0.01
            RatingsArray[movie][user] = rating

        return RatingsArray


    def _readFeatures(self):
        FeatureArray = pd.read_csv(self._trainSetPath + self._featuresSetName + self._trainSetExt, low_memory=False)
        FeatureArray = FeatureArray.drop(FeatureArray.columns[[0]], 1)
        FeatureArray = FeatureArray.values
        return FeatureArray


    def _readRatings(self):
        RatingsArray = pd.read_csv(self._trainSetPath + self._ratingsSetName + self._trainSetExt, low_memory=False)
        RatingsArray = RatingsArray.drop(RatingsArray.columns[[0]], 1)
        RatingsArray = RatingsArray.values
        return RatingsArray


    def _saveResults(self, FeatureArray, RatingsArray):
        featuresSet = pd.DataFrame.from_records(FeatureArray)
        featuresSet.to_csv(self._trainSetPath + self._featuresSetName + self._trainSetExt)

        recommendationsSet = pd.DataFrame.from_records(RatingsArray)
        recommendationsSet.to_csv(self._trainSetPath + self._ratingsSetName + self._trainSetExt)


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


    def train(self):
        # read data
        RatingsArray = self._readData()

        noMovies = len(RatingsArray)
        noUsers = len(RatingsArray[0])

        # generate random X and Theta
        moviesToFeatures = np.random.rand(noMovies, self._noFeatures)
        usersToFeatures = np.random.rand(noUsers, self._noFeatures)

        # R - 1 where rating
        R = RatingsArray.copy()
        R[R > 0] = 1
        noRatings = R.sum()

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
                              options={'disp': False, 'maxiter': self._noMaxIterations},
                              method="L-BFGS-B",
                              jac=True)
        # unfold the results
        finalParams = results["x"]
        X = np.reshape(finalParams[:noMovies * self._noFeatures], (noMovies, self._noFeatures), order='F')
        Theta = np.reshape(finalParams[noMovies * self._noFeatures:], (noUsers, self._noFeatures), order='F')

        # get ratings and {movies to features}
        RatingsResult = np.matmul(X, Theta.transpose())
        FeaturesResult = X

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

        self._saveResults(FeaturesResult, RatingsResult)


    def getSimilarMovies(self, movieId, noMovies):
        FeatureArray = self._readFeatures()
        # assert (RatingsArray.shape[0] > movieId)
        if FeatureArray.shape[0] < movieId:
            print("movieId exceeds length of movieIds")
            exit()
        givenFeatureVector = FeatureArray[movieId]
        similarMoviesHeap = []
        for idx in range(0, FeatureArray.shape[0]):
            if movieId == idx:
                continue
            row = FeatureArray[idx]
            compareFeatureVectors = abs(row - givenFeatureVector)
            assert (len(similarMoviesHeap) <= noMovies)
            if len(similarMoviesHeap) < noMovies:
                heapq.heappush(similarMoviesHeap, (-compareFeatureVectors.sum(), compareFeatureVectors, idx))
            else:
                #heapq.heapify(similarMoviesHeap)
                heapq.heappushpop(similarMoviesHeap, (-compareFeatureVectors.sum(), compareFeatureVectors, idx))

        similarMoviesHeapSize = len(similarMoviesHeap)
        similarMovies = [heapq.heappop(similarMoviesHeap) for _ in range(0, similarMoviesHeapSize)]
        similarMovies = list(reversed(similarMovies))
        similarMoviesIds = [elem[2] for elem in similarMovies]
        return similarMoviesIds


    def getImpersonatedUserMovies(self, userId, noMovies):
        RatingsArray = self._readRatings()
        # assert (RatingsArray.shape[1] > userId)
        if RatingsArray.shape[1] < userId:
            print("userId exceeds length of userIds")
            exit()
        userRatings = list(RatingsArray[:,userId])
        for i in range(0, len(userRatings)):
            userRatings[i] = (i, userRatings[i])
        userRatings.sort(key=lambda x: x[1], reverse=True)
        # assert (len(userRatings) > noMovies)
        if len(userRatings) < noMovies:
            print("noMovies exceeds length of userRatings")
            exit()
        return userRatings[:noMovies]
