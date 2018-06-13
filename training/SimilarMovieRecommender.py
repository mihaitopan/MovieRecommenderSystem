from CollaborativeFiltering import CollaborativeFiltering
from ContentFiltering import ContentFiltering


if __name__ == "__main__":
    # constants
    command = "ColCon"

    # what to validate?
    if command == "ColCol":
        print("run: collaborative algorithm - collaborative dataset")
        dataset = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\collaborative\\collaborativeRatings.csv"
        trainSetPath = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\training\\sets\\"

        recommender = CollaborativeFiltering(dataset, trainSetPath)
        #recommender.train()
        movieIds = recommender.getSimilarMovies(1, 5)
        for movieId in movieIds:
            print(movieId)

    elif command == "ColCon":
        print("run: collaborative algorithm - content dataset")
        dataset = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentRatings.csv"
        trainSetPath = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\training\\sets\\"

        recommender = CollaborativeFiltering(dataset, trainSetPath)
        #recommender.train()
        movieIds = recommender.getSimilarMovies(722, 5)
        for movieId in movieIds:
            print(movieId)

    elif command == "ConCon":
        print("run: content algorithm - content dataset")
        ratingsFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentRatings.csv"
        tagsFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentTags.csv"
        trainSetPath = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\training\\sets\\"

        recommender = ContentFiltering(ratingsFilename, tagsFilename, trainSetPath)
        #recommender.train()
        movieIds = recommender.getSimilarMovies(722, 5)
        for movieId in movieIds:
            print(movieId)

    else:
        print("bad command")
