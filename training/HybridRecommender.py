from CollaborativeFiltering import CollaborativeFiltering
from ContentFiltering import ContentFiltering
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


if __name__ == "__main__":
    # constants
    noMovies = 10
    recommendedMovies = []

    dataset = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentRatings.csv"
    ratingsFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentRatings.csv"
    tagsFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentTags.csv"
    trainSetPath = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\training\\sets\\"
    collaborativeTrainer = CollaborativeFiltering(dataset, trainSetPath)
    #collaborativeTrainer.train()
    contentTrainer = ContentFiltering(ratingsFilename, tagsFilename, trainSetPath)
    #contentTrainer.train()

    collaborativeMovies = collaborativeTrainer.getImpersonatedUserMovies(noMovies, 10)
    contentMovies = contentTrainer.getImpersonatedUserMovies(noMovies, 10)

    i = 0
    while i < noMovies:
        if isPrime(i):
            recommendedMovies.append(contentMovies[0])
            contentMovies = contentMovies[1:]
            i += 1
            if i < noMovies:
                recommendedMovies.append(contentMovies[0])
                contentMovies = contentMovies[1:]
                i += 1
        else:
            recommendedMovies.append(collaborativeMovies[0])
            contentMovies = contentMovies[1:]
            i += 1

    for elem in recommendedMovies:
        print(elem)

