import pandas as pd
import numpy as np
import random
from sys import exit


class CrossContentFiltering:
    def __init__(self, ratingsFilename, tagsFilename, trainSetPath, noCross=5):
        # necessary constants
        self._ratingsData = ratingsFilename
        self._tagsData = tagsFilename
        self._trainSetPath = trainSetPath
        self._noCross = noCross

        # dataset constants
        self._trainSetName = "crossSet"
        self._testSetRatingsName = "Ratings"
        self._trainSetResultName = "Results"
        self._trainSetExt = ".csv"
        self._trainResultsName = "Recommendations"


    def _readData(self, noMaxUsers=10000, noMaxMovies=10000):
        Ratings = pd.read_csv(self._ratingsData, encoding='latin1', low_memory=False)
        Tags = pd.read_csv(self._tagsData, encoding='latin1', low_memory=False)
        Ratings = Ratings[["movieId", "userId", "rating"]]
        Tags = Tags[["movieId", "userId", "tag"]]

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

        return RatingsArray, Ratings, Tags


    def _saveCrossSets(self, T, idx, RatingsArray):
        trainSet = pd.DataFrame.from_records(T)
        trainSet.to_csv(self._trainSetPath + self._trainSetName + str(idx) + self._trainSetExt)

        testSetArray = RatingsArray.copy()
        testSetArray = testSetArray * T
        testSet = pd.DataFrame.from_records(testSetArray)
        testSet.to_csv(
            self._trainSetPath + self._trainSetName + str(idx) + self._testSetRatingsName + self._trainSetExt)


    def _readCrossResults(self, idx, noMaxUsers=10000, noMaxMovies=10000):
        FoundRatings = pd.read_csv(self._trainSetPath + self._trainSetName + str(idx) + self._trainResultsName
                                     + self._trainSetExt, low_memory=False)

        noUniqueMovies = FoundRatings[["movieId"]].drop_duplicates().size
        # assert (noUniqueMovies < noMaxMovies)
        if noUniqueMovies > noMaxMovies:
            print("dataset exceeds maximum size (noMaxMovies=10000)")
            exit()
        noUniqueUsers = FoundRatings[["userId"]].drop_duplicates().size
        # assert (noUniqueUsers < noMaxUsers)
        if noUniqueUsers > noMaxUsers:
            print("dataset exceeds maximum size (noMaxUsers=10000)")
            exit()

        values = FoundRatings[["userId", "movieId", "rating"]].values
        # margin for movies which don't ha associated tags
        noUniqueMoviesMargin = noUniqueMovies + int(noUniqueMovies / 50)
        noUniqueUsersMargin = noUniqueUsers + int(noUniqueUsers / 50)
        RatingsArray = np.zeros(shape=(noUniqueMoviesMargin, noUniqueUsersMargin))
        for i in range(0, values.shape[0]):
            user = np.int64(values[i][0])
            movie = np.int64(values[i][1])
            rating = values[i][2]
            # assert (RatingsArray[movie][user] == 0)
            RatingsArray[movie][user] = rating

        recommendationsSet = pd.DataFrame.from_records(RatingsArray)
        recommendationsSet.to_csv(
            self._trainSetPath + self._trainSetName + str(idx) + self._trainResultsName + self._trainSetExt)

        return RatingsArray


    def _trainSet(self, Ratings, Tags, idx):
        '''movie plays the role of document in TF IDF'''
        '''TF - the number of times a tag was tagged to a movie'''
        Tags = Tags.groupby(["movieId", "tag"], as_index=False, sort=False).count()
        Tags = Tags.rename(columns={"userId": "TF"})
        Tags = Tags[["movieId", "tag", "TF"]]

        '''DF - the number of different movies a tag was tagged to'''
        distinctTags = Tags[["tag", "movieId"]].drop_duplicates()
        distinctTags = distinctTags.groupby(["tag"], as_index=False, sort=False).count()
        distinctTags = distinctTags.rename(columns={"movieId": "DF"})
        distinctTags = distinctTags[["tag", "DF"]]

        '''inverse document frequency = log(N/nt) = log(N) - log(nt)'''
        noDistinctTags = np.log(len(np.unique(Tags["movieId"])))
        distinctTags["IDF"] = noDistinctTags - np.log(distinctTags["DF"])

        '''compute TF-IDF = TF * IDF for each (movie, tag)'''
        Tags = pd.merge(Tags, distinctTags, on='tag', how='left', sort=False)
        Tags["TF-IDF"] = Tags["TF"] * Tags["IDF"]
        Tags = Tags[["movieId", "tag", "TF-IDF"]]

        '''compute TF-IDF SquaredSum SquareRoot = squareRoot of sum(TF-IDF ^ 2)'''
        TagWeight = Tags[["movieId", "TF-IDF"]].copy()
        TagWeight["TF-IDF Squared"] = TagWeight["TF-IDF"] ** 2
        TagWeight = TagWeight.groupby(["movieId"], as_index=False, sort=False).sum()
        TagWeight["TF-IDF SquaredSum SquareRoot"] = np.sqrt(TagWeight["TF-IDF Squared"])
        TagWeight = TagWeight[["movieId", "TF-IDF SquaredSum SquareRoot"]]

        '''compute TagWeight = TF-IDF / TF-IDF SquaredSum SquareRoot'''
        Tags = pd.merge(Tags, TagWeight, on="movieId", how='left', sort=False)
        Tags["TagWeight"] = Tags["TF-IDF"] / Tags["TF-IDF SquaredSum SquareRoot"]
        Tags = Tags[["movieId", "tag", "TagWeight"]]

        '''compute user tag preferences for each user'''
        usersPreferences = pd.DataFrame()
        userIdsArray = np.unique(Ratings["userId"])
        for userId in userIdsArray:
            '''normalize user ratings'''
            userVector = Ratings.copy()
            userVector = userVector[Ratings["userId"] == userId]
            userRatingMean = userVector["rating"].mean()
            userVector["rating"] = userVector["rating"] - userRatingMean

            '''append preferences of (movie, tag) of user's rated movies'''
            userVector = pd.merge(Tags, userVector, on="movieId", how='inner', sort=False)
            userVector["movieTagPreference"] = userVector["TagWeight"] * userVector["rating"]

            '''compute preferences per tag regardless of the movie'''
            userVector = userVector.groupby(["tag"], as_index=False, sort=False).sum()
            userVector = userVector.rename(columns={"movieTagPreference": "tagPreference"})
            userVector = userVector[["tag", "tagPreference"]]

            '''append result (with user info)'''
            userVector["user"] = userId
            usersPreferences = usersPreferences.append(userVector, ignore_index=True)

        # append ratings to file and reset dataframe (not to get too big)
        FoundRatings = pd.DataFrame([["","userId","movieId","rating"]])
        FoundRatings.to_csv(self._trainSetPath + self._trainSetName + str(idx) +
                            self._trainResultsName + self._trainSetExt, header=False, index=False)

        # append to the results file each user's predictions
        FoundRatings = pd.DataFrame()
        userIdsArray = np.unique(Ratings["userId"])
        for userId in userIdsArray:
            userPreference = usersPreferences[usersPreferences["user"] == userId]

            # append ratings to file and reset dataframe (not to get too big)
            FoundRatings.to_csv(self._trainSetPath + self._trainSetName + str(idx) +
                                self._trainResultsName + self._trainSetExt, mode='a', header=False)
            FoundRatings = pd.DataFrame()

            '''compute predictions of user's ratings for all movies'''
            distinctMovies = np.unique(Tags["movieId"])
            for movie in distinctMovies:
                '''compute the tag importance =  weight * preference'''
                movieVector = Tags[Tags["movieId"] == movie]
                movieTagPreference = pd.merge(movieVector, userPreference, on="tag", how='left', sort=False)
                movieTagPreference["tagPreference"] = movieTagPreference["tagPreference"].fillna(0)
                movieTagPreference["tagImportance"] = movieTagPreference["TagWeight"] \
                                                      * movieTagPreference["tagPreference"]

                '''compute totalWeight & totalPreference 
                (of which product is the denominator in cosine similarity formula)'''
                totalWeight = np.sqrt(np.sum(np.square(movieTagPreference["TagWeight"]), axis=0))
                totalPreference = np.sqrt(np.sum(np.square(userPreference["tagPreference"]), axis=0))

                '''compute the total importance (numerator in cosine similarity formula)'''
                predictedRating = movieTagPreference.groupby(["user", "movieId"])[["tagImportance"]].sum()
                predictedRating = predictedRating.rename(columns={"tagImportance": "cosineSimilarity"}).reset_index()

                '''compute cosine similarity between userVector and movieVector'''
                predictedRating["cosineSimilarity"] = predictedRating["cosineSimilarity"] \
                                                      / (totalWeight * totalPreference)
                predictedRating["cosineSimilarity"] = (1 - np.arccos(predictedRating["cosineSimilarity"]) / np.pi) * 5
                predictedRating["cosineSimilarity"] = 2.5 + (predictedRating["cosineSimilarity"] - 2.5) * 5
                predictedRating = predictedRating.rename(columns={"cosineSimilarity": "Rating"})

                # if predictedRating.empty:
                #     predictedRating = pd.DataFrame([[user, movie, 0]], columns=["user", "movieId", "Rating"])

                '''append result to ratings dataframe'''
                FoundRatings = FoundRatings.append(predictedRating, ignore_index=True)


    def train(self):
        # read data
        ratingsArray = Ratings = Tags = None
        try:
            ratingsArray, Ratings, Tags = self._readData()
        except IOError as _:
            print("Could not read input. Please respect initial input data and directory tree.")
            exit()

        R = ratingsArray.copy()
        R[R > 0] = 1

        # get random indexes in Ratings dataset
        noRatings = Ratings.shape[0]
        assert (noRatings == R.sum())
        randomIndexesList = list(range(0, noRatings))
        random.shuffle(randomIndexesList)

        # get list of testsets(list of random indexes in Ratings dataset)
        testSetIndexes = []
        for e in range(0, self._noCross):
            testIndexList = []
            for i in range(e, noRatings, self._noCross):
                testIndexList.append(randomIndexesList[i])
            testSetIndexes.append(testIndexList)

        # run algorithm for each test
        for e in range(0, self._noCross):
            TrainSet = R.copy()
            TestSet = R.copy()
            TestSet[TestSet > 0] = 0

            # obtain testset
            RatingsCopy = Ratings.copy()
            TestRatings = RatingsCopy.iloc[testSetIndexes[e]]
            assert (TestRatings.shape[0] == len(testSetIndexes[e]))

            # obtain testset and trainset as arrays
            for _, row in TestRatings.iterrows():
                movieId = int(row["movieId"])
                userId = int(row["userId"])
                assert (R[movieId][userId] == 1)
                TestSet[movieId][userId] = 1
                TrainSet[movieId][userId] = 0
            assert (TrainSet.sum() + TestSet.sum() == R.sum())

            # obtain trainset
            TrainRatings = Ratings.copy()
            TrainRatings = TrainRatings.drop(TrainRatings.index[testSetIndexes[e]])
            assert (TrainRatings.shape[0] + TestRatings.shape[0] == Ratings.shape[0])

            # flush to disk test sets and train on trainset
            self._saveCrossSets(TestSet, e, ratingsArray)
            self._trainSet(TrainRatings, Tags, e)


    def test(self):
        crossMAEMean = 0
        crossRMSDMean = 0
        for idx in range(0, self._noCross):
            # read data into arrays
            testSet = pd.read_csv(self._trainSetPath + self._trainSetName + str(idx) +
                                  self._testSetRatingsName + self._trainSetExt, low_memory=False)
            trainSet = pd.read_csv(self._trainSetPath + self._trainSetName + str(idx) + self._trainSetExt,
                                   low_memory=False)

            testSet = testSet.drop(testSet.columns[[0]], 1)
            trainSet = trainSet.drop(trainSet.columns[[0]], 1)

            testRatings = testSet.values
            trainSetArray = trainSet.values
            RatingsArray = self._readCrossResults(idx)
            RatingsArray = RatingsArray[0:trainSetArray.shape[0], 0:trainSetArray.shape[1]]
            testResults = RatingsArray * trainSetArray

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

