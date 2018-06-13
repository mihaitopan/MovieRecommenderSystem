import pandas as pd
import numpy as np
import heapq


class ContentFiltering:
    def __init__(self, ratingsFilename, tagsFilename, trainSetPath,):
        # necessary constants
        self._ratingsData = ratingsFilename
        self._tagsData = tagsFilename
        self._trainSetPath = trainSetPath

        # dataset constants
        self._ratingsSetName = "ContentRatings"
        self._featuresSetName = "ContentFeatures"
        self._trainSetExt = ".csv"


    def _readData(self, noMaxUsers=1000, noMaxMovies=10000):
        Ratings = pd.read_csv(self._ratingsData, encoding='latin1', low_memory=False)
        Tags = pd.read_csv(self._tagsData, encoding='latin1', low_memory=False)
        Ratings = Ratings[["movieId", "userId", "rating"]]
        Tags = Tags[["movieId", "userId", "tag"]]

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

        return RatingsArray, Ratings, Tags


    def _saveTrainRatings(self, noMaxUsers=1000, noMaxMovies=10000):
        FoundRatings = pd.read_csv(self._trainSetPath + self._ratingsSetName + self._trainSetExt, low_memory=False)

        noUniqueMovies = FoundRatings[["movieId"]].drop_duplicates().size
        assert (noUniqueMovies < noMaxMovies)
        noUniqueUsers = FoundRatings[["userId"]].drop_duplicates().size
        assert (noUniqueUsers < noMaxUsers)

        values = FoundRatings[["userId", "movieId", "rating"]].values
        #RatingsArray = np.zeros(shape=(noUniqueMovies, noUniqueUsers))
        RatingsArray = np.zeros(shape=(noUniqueMovies + 5, noUniqueUsers)) # 5 movies have no tags in my dataset
        for i in range(0, values.shape[0]):
            user = np.int64(values[i][0])
            movie = np.int64(values[i][1])
            rating = values[i][2]
            assert (RatingsArray[movie][user] == 0)
            RatingsArray[movie][user] = rating

        recommendationsSet = pd.DataFrame.from_records(RatingsArray)
        recommendationsSet.to_csv(self._trainSetPath + self._ratingsSetName + self._trainSetExt)

        return RatingsArray


    def _readFeatures(self):
        FeatureArray = pd.read_csv(self._trainSetPath + self._featuresSetName + self._trainSetExt, low_memory=False)
        FeatureArray = FeatureArray.drop(FeatureArray.columns[[0]], 1)
        return FeatureArray


    def _readRatings(self):
        RatingsArray = pd.read_csv(self._trainSetPath + self._ratingsSetName + self._trainSetExt, low_memory=False)
        RatingsArray = RatingsArray.drop(RatingsArray.columns[[0]], 1)
        return RatingsArray


    def train(self):
        # read data
        RatingsArray, Ratings, Tags = self._readData()

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

        # flush movie-tags to disk (will be used as features later)
        Features = Tags[["movieId", "tag", "TagWeight"]]
        Features.to_csv(self._trainSetPath + self._featuresSetName + self._trainSetExt)

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

        # append to the results file each user's predictions
        FoundRatings = pd.DataFrame()
        userIdsArray = np.unique(Ratings["userId"])
        for userId in userIdsArray:
            userPreference = usersPreferences[usersPreferences["user"] == userId]

            # append ratings to file and reset dataframe (not to get too big)
            FoundRatings.to_csv(self._trainSetPath + self._ratingsSetName + self._trainSetExt, mode='a', header=False)
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
        self._saveTrainRatings()


    def getSimilarMovies(self, movieId, noMovies):
        FeatureArray = self._readFeatures()
        givenMovieFeatureArray = FeatureArray.loc[FeatureArray["movieId"] == movieId]

        similarMovies = []
        distinctMovies = np.unique(FeatureArray["movieId"])
        for movie in distinctMovies:
            if movie == movieId:
                continue

            movieData = FeatureArray[FeatureArray["movieId"] == movie]
            movieData = pd.merge(movieData, givenMovieFeatureArray, on="tag", how='right', sort=False)
            movieData["TagWeight_y"] = movieData["TagWeight_y"].fillna(0)

            movieData["TagWeight_x * TagWeight_y"] = movieData["TagWeight_x"] * movieData["TagWeight_y"]

            totalWeight = np.sqrt(np.sum(np.square(movieData["TagWeight_x"]), axis=0))
            totalPreference = np.sqrt(np.sum(np.square(movieData["TagWeight_y"]), axis=0))

            movieData = movieData.groupby(['movieId_x', 'movieId_y'])[["TagWeight_x * TagWeight_y"]] \
                .sum() \
                .rename(columns={"TagWeight_x * TagWeight_y": "cosineSimilarity"}).reset_index()
            movieData["cosineSimilarity"] = movieData["cosineSimilarity"] / (totalWeight * totalPreference)

            compareFeatureVectors = 0
            if not movieData.empty:
                compareFeatureVectors = movieData["cosineSimilarity"].values[0]

            assert (len(similarMovies) <= noMovies)
            if len(similarMovies) < noMovies:
                heapq.heappush(similarMovies, (compareFeatureVectors, movie))
            else:
                #heapq.heapify(similarMovies)
                heapq.heappushpop(similarMovies, (compareFeatureVectors, movie))

        similarMoviesIds = [elem[1] for elem in similarMovies]
        return similarMoviesIds


    def getImpersonatedUserMovies(self, userId, noMovies):
        RatingsArray = self._readRatings()
        RatingsArray = RatingsArray.values
        userRatings = list(RatingsArray[:,userId])
        for i in range(0, len(userRatings)):
            userRatings[i] = (i, userRatings[i])
        userRatings.sort(key=lambda x: x[1], reverse=True)
        assert (len(userRatings) > noMovies)
        return userRatings[:noMovies]
