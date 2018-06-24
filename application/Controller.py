from CollaborativeFiltering import CollaborativeFiltering
from ContentFiltering import ContentFiltering
from Repository import Repository
from math import sqrt


def isMagicNumber(number):
    # 0, 2, 3, 5 are magic
    # from 5 on all primes and next of primes are magic
    if number < 0:
        return False
    if number == 0:
        return True
    if number == 1:
        return False
    if number == 2:
        return True
    if number == 3:
        return True
    if number % 2 == 0:
        return False
    for number in range(5, int(sqrt(number)) + 1, 2):
        if number % number == 0:
            return False
    else:
        return True


class Controller:
    def __init__(self, trainSetPath, dataset, ratingsFilename, tagsFilename, moviesFilename):
        self._trainSetPath = trainSetPath
        self._dataset = dataset
        self._ratingsFilename = ratingsFilename
        self._tagsFilename = tagsFilename
        self._repo = Repository(moviesFilename)


    def _hybridise(self, noMovies, collaborativeMovies, contentMovies):
        recommendedMovies = []
        recommendedMoviesSet = set()
        count = 0
        while count < noMovies:
            if isMagicNumber(count):
                movieToAdd = contentMovies[0]
                if movieToAdd not in recommendedMoviesSet:
                    recommendedMoviesSet.add(movieToAdd)
                    recommendedMovies.append(movieToAdd)
                    count += 1
                contentMovies = contentMovies[1:]
            else:
                movieToAdd = collaborativeMovies[0]
                if movieToAdd not in recommendedMoviesSet:
                    recommendedMoviesSet.add(movieToAdd)
                    recommendedMovies.append(movieToAdd)
                    count += 1
                collaborativeMovies = collaborativeMovies[1:]
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
            collaborativeMovies = collaborativeTrainer.getSimilarMovies(movieId, noMovies*2)
            contentTrainer = ContentFiltering(self._ratingsFilename, self._tagsFilename, self._trainSetPath)
            # contentTrainer.train()
            contentMovies = contentTrainer.getSimilarMovies(movieId, noMovies*2)
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
            collaborativeMovies = collaborativeTrainer.getImpersonatedUserMovies(userId, noMovies*2)
            contentTrainer = ContentFiltering(self._ratingsFilename, self._tagsFilename, self._trainSetPath)
            # contentTrainer.train()
            contentMovies = contentTrainer.getImpersonatedUserMovies(userId, noMovies*2)
            return self._hybridise(noMovies, collaborativeMovies, contentMovies)

        else:
            pass


    def getMovies(self):
        return self._repo.getMovies()

