import pandas as pd
from Movie import Movie


class Repository:
    def __init__(self, filename):
        self._filename = filename
        self._movies = dict()
        self._readData()

    def _readData(self):
        Movies = pd.read_csv(self._filename, encoding='latin1', low_memory=False)
        for _, row in Movies.iterrows():
            self._movies[row["movieId"]] = Movie(row["movieId"], row["title"], row["genres"])

    def getMovieById(self, movieID):
        if movieID in self._movies.keys():
            return self._movies[movieID]
        return None

    def getMovies(self):
        return self._movies
