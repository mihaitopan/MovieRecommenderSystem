from CollaborativeFiltering import CollaborativeFiltering
from ContentFiltering import ContentFiltering
from Repository import Repository
from math import sqrt


def isPrime(magicNumber):
    if magicNumber < 2:
        return False
    if magicNumber == 2:
        return True
    if magicNumber % 2 == 0:
        return False
    for number in range(3, int(sqrt(magicNumber)) + 1, 2):
        if magicNumber % number == 0:
            return False
    else:
        return True


class Controller:
    def __init__(self, trainSetPath, dataset, ratingsFilename, tagsFilename, moviesFilename):
        # trainSetPath = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\training\\sets\\"
        self._trainSetPath = trainSetPath

        # dataset = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\collaborative\\collaborativeRatings.csv"
        self._dataset = dataset

        # ratingsFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentRatings.csv"
        self._ratingsFilename = ratingsFilename

        # tagsFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentTags.csv"
        self._tagsFilename = tagsFilename

        # moviesFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentMovies.csv"
        self._repo = Repository(moviesFilename)


    def _hybridise(self, noMovies, collaborativeMovies, contentMovies):
        recommendedMovies = []
        count = 0
        while count < noMovies:
            if count == 0:
                recommendedMovies.append(contentMovies[0])
                contentMovies = contentMovies[1:]
                count += 1
            if isPrime(count):
                recommendedMovies.append(contentMovies[0])
                contentMovies = contentMovies[1:]
                count += 1
                if count < noMovies:
                    recommendedMovies.append(contentMovies[0])
                    contentMovies = contentMovies[1:]
                    count += 1
            else:
                recommendedMovies.append(collaborativeMovies[0])
                collaborativeMovies = collaborativeMovies[1:]
                count += 1
        return recommendedMovies


    def getSimilarMovies(self, method, movieId, noMovies):

        if method == "Collaborative":
            recommender = CollaborativeFiltering(self._ratingsFilename, self._trainSetPath)
            # recommender.train()
            return recommender.getSimilarMovies(movieId, noMovies)

        elif method == "Content":
            recommender = ContentFiltering(self._ratingsFilename, self._tagsFilename, self._trainSetPath)
            # recommender.train()
            return recommender.getSimilarMovies(movieId, noMovies)

        elif method == "Hybrid":
            collaborativeTrainer = CollaborativeFiltering(self._ratingsFilename, self._trainSetPath)
            # collaborativeTrainer.train()
            collaborativeMovies = collaborativeTrainer.getSimilarMovies(movieId, noMovies)
            contentTrainer = ContentFiltering(self._ratingsFilename, self._tagsFilename, self._trainSetPath)
            # contentTrainer.train()
            contentMovies = contentTrainer.getSimilarMovies(movieId, noMovies)
            return self._hybridise(noMovies, collaborativeMovies, contentMovies)

        else:
            pass


    def getUserMovies(self, method, userId, noMovies):

        if method == "Collaborative":
            recommender = CollaborativeFiltering(self._ratingsFilename, self._trainSetPath)
            # recommender.train()
            return recommender.getImpersonatedUserMovies(userId, noMovies)

        elif method == "Content":
            recommender = ContentFiltering(self._ratingsFilename, self._tagsFilename, self._trainSetPath)
            # recommender.train()
            return recommender.getImpersonatedUserMovies(userId, noMovies)

        elif method == "Hybrid":
            collaborativeTrainer = CollaborativeFiltering(self._ratingsFilename, self._trainSetPath)
            # collaborativeTrainer.train()
            collaborativeMovies = collaborativeTrainer.getImpersonatedUserMovies(userId, noMovies)
            contentTrainer = ContentFiltering(self._ratingsFilename, self._tagsFilename, self._trainSetPath)
            # contentTrainer.train()
            contentMovies = contentTrainer.getImpersonatedUserMovies(userId, noMovies)
            return self._hybridise(noMovies, collaborativeMovies, contentMovies)

        else:
            pass


    def getMovies(self):
        return self._repo.getMovies()

