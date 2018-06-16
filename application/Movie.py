
class Movie:
    def __init__(self, movieId, title, genres):
        self._movieId = movieId
        self._title = title
        self._genres = genres

    def getMovieId(self):
        return self._movieId

    def getTitle(self):
        return self._title

    def getGenres(self):
        return self._genres

    def __str__(self):
        return self._title
