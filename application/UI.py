from Controller import Controller
from tkinter import *
from sys import exit


class UI:
    def __init__(self, trainSetPath, dataset, ratingsFilename, tagsFilename, moviesFilename):
        self._ctrl = Controller(trainSetPath, dataset, ratingsFilename, tagsFilename, moviesFilename)
        self._movies = self._ctrl.getMovies()

        # defaults
        self._defaultMethod = "Hybrid"
        self._defaultNoMovies = 16
        self._defaultMovieId = 157
        self._defaultUserId = 512
        self._defaultMaxNoUsers = 861

        # ui objects
        self._movieIdEntry = None
        self._allMoviesListBox = None
        self._similarMoviesListBox = None
        self._rightFrame = None
        self._buttonsFrame = None
        self._methodVar = None
        self._noMoviesEntry = None
        self._userIdEntry = None


    def _onSelect(self, event):
        w = event.widget
        try:
            index = int(w.curselection()[0])
            value = w.get(index)
            print(value, " has been selected")
            self._movieIdEntry.delete(0, END)
            self._movieIdEntry.insert(END, str(index))
        except IndexError:
            # if user inputs bad index by hand, it cannot be auto-selected
            pass


    def _endExecution(self, _):
        print("End execution")
        exit()


    def _getSimilarMovies(self, _):
        method = self._methodVar.get()

        # read movieId
        keyMovieId = self._defaultMovieId
        try:
            keyMovieId = self._movieIdEntry.get()
            keyMovieId = int(keyMovieId)
        except ValueError:
            print("movieId  ", keyMovieId, " is invalid. run default movieId")
            keyMovieId = self._defaultMovieId
            self._movieIdEntry.delete(0, END)
            self._movieIdEntry.insert(END, str(self._defaultMovieId))

        if keyMovieId < 0 or keyMovieId > len(self._movies):
            print("movieId  ", keyMovieId, " is invalid. run default movieId")
            keyMovieId = self._defaultMovieId
            self._movieIdEntry.delete(0, END)
            self._movieIdEntry.insert(END, str(self._defaultMovieId))

        # read noMovies
        noMovies = self._defaultNoMovies
        try:
            noMovies = self._noMoviesEntry.get()
            noMovies = int(noMovies)
        except ValueError:
            print("noMovies  ", noMovies, " is invalid. run default noMovies")
            noMovies = self._defaultNoMovies
            self._noMoviesEntry.delete(0, END)
            self._noMoviesEntry.insert(END, str(self._defaultNoMovies))

        if noMovies < 0 or noMovies > 50:
            print("noMovies  ", noMovies, " is invalid. run default noMovies")
            noMovies = self._defaultNoMovies
            self._noMoviesEntry.delete(0, END)
            self._noMoviesEntry.insert(END, str(self._defaultNoMovies))

        self._allMoviesListBox.see(keyMovieId)
        self._similarMoviesListBox.delete(0, END)
        idx = 0
        self._similarMoviesListBox.insert(idx, "SELECTED MOVIE: " + str(self._movies[keyMovieId]))
        idx += 1
        self._similarMoviesListBox.insert(idx, " ")
        recommendedMovieIds = self._ctrl.getSimilarMovies(method, keyMovieId, noMovies)
        for movieId in recommendedMovieIds:
            idx += 1
            self._similarMoviesListBox.insert(idx, str(self._movies[movieId]))

        self._similarMoviesListBox.grid(row=1, column=1)
        print("getSimilarMovies  ", method, "  ", self._movies[keyMovieId])


    def _getUserMovies(self, _):
        method = self._methodVar.get()

        # read userId
        userId = self._defaultUserId
        try:
            userId = self._userIdEntry.get()
            userId = int(userId)
        except ValueError:
            print("userId  ", userId, " is invalid. run default userIds")
            userId = self._defaultUserId
            self._userIdEntry.delete(0, END)
            self._userIdEntry.insert(END, str(self._defaultUserId))

        if userId < 0 or userId > self._defaultMaxNoUsers:
            print("userId  ", userId, " is invalid. run default userIds")
            userId = self._defaultUserId
            self._userIdEntry.delete(0, END)
            self._userIdEntry.insert(END, str(self._defaultUserId))

        # read noMovies
        noMovies = self._defaultNoMovies
        try:
            noMovies = self._noMoviesEntry.get()
            noMovies = int(noMovies)
        except ValueError:
            print("noMovies  ", noMovies, " is invalid. run default noMovies")
            noMovies = self._defaultNoMovies
            self._noMoviesEntry.delete(0, END)
            self._noMoviesEntry.insert(END, str(self._defaultNoMovies))

        if noMovies < 0 or noMovies > 50:
            print("noMovies  ", noMovies, " is invalid. run default noMovies")
            noMovies = self._defaultNoMovies
            self._noMoviesEntry.delete(0, END)
            self._noMoviesEntry.insert(END, str(self._defaultNoMovies))

        self._similarMoviesListBox.delete(0, END)
        idx = 0
        self._similarMoviesListBox.insert(idx, "SELECTED USER ID: " + str(userId))
        idx += 1
        self._similarMoviesListBox.insert(idx, " ")
        recommendedMovieIds = self._ctrl.getUserMovies(method, userId, noMovies)
        for movieId, rating in recommendedMovieIds:
            idx += 1
            self._similarMoviesListBox.insert(idx, str(self._movies[movieId]) + "  " + str(round(rating, 2)))

        self._similarMoviesListBox.insert(1, self._userIdEntry.get())
        self._similarMoviesListBox.grid(row=1, column=1)
        print("getUserMovies  ", method, "  userID:", userId)


    def run(self):
        # User Interface
        master = Tk()
        master.title("Movie Recommender System")
        leftFrame = Frame(master)
        leftFrame.pack(side=LEFT)
        middleFrame = Frame(master)
        middleFrame.pack(side=LEFT)
        methodFrame = Frame(middleFrame)
        methodFrame.pack(side=TOP)
        self._buttonsFrame = Frame(middleFrame)
        self._buttonsFrame.pack(side=TOP)
        self._rightFrame = Frame(master)
        self._rightFrame.pack(side=LEFT)

        movies = self._ctrl.getMovies()
        self._allMoviesListBox = Listbox(leftFrame, width=60, height=40)
        for key in movies.keys():
            # for now, key in list has to be movieId
            self._allMoviesListBox.insert(key, movies[key])

        # method
        self._methodVar = StringVar(None)
        self._methodVar.set(str(self._defaultMethod))
        colRadioButton = Radiobutton(methodFrame, text="Collaborative", variable=self._methodVar, value="Collaborative")
        colRadioButton.grid(row=1, column=1)
        conRadioButton = Radiobutton(methodFrame, text="Content", variable=self._methodVar, value="Content")
        conRadioButton.grid(row=2, column=1)
        hybRadioButton = Radiobutton(methodFrame, text="Hybrid", variable=self._methodVar, value="Hybrid")
        hybRadioButton.grid(row=3, column=1)

        # number of movies entry
        noMoviesLabelText = StringVar()
        noMoviesLabelText.set("number of movies: ")
        noMoviesLabel = Label(self._buttonsFrame, textvariable=noMoviesLabelText, height=6)
        noMoviesLabel.grid(row=1, column=1)
        noMoviesEntryText = StringVar(None)
        self._noMoviesEntry = Entry(self._buttonsFrame, textvariable=noMoviesEntryText, width=6)
        self._noMoviesEntry.insert(END, str(self._defaultNoMovies))
        self._noMoviesEntry.grid(row=1, column=2)

        # movie id entry
        movieIdLabelText = StringVar()
        movieIdLabelText.set("movie id: ")
        movieIdLabel = Label(self._buttonsFrame, textvariable=movieIdLabelText, height=6)
        movieIdLabel.grid(row=2, column=1)
        movieIdEntryText = StringVar(None)
        self._movieIdEntry = Entry(self._buttonsFrame, textvariable=movieIdEntryText, width=6)
        self._movieIdEntry.insert(END, str(self._defaultMovieId))
        self._movieIdEntry.grid(row=2, column=2)

        # user id entry
        userIdLabelText = StringVar()
        userIdLabelText.set("user id: ")
        userIdLabel = Label(self._buttonsFrame, textvariable=userIdLabelText, height=6)
        userIdLabel.grid(row=3, column=1)
        userIdEntryText = StringVar(None)
        self._userIdEntry = Entry(self._buttonsFrame, textvariable=userIdEntryText, width=6)
        self._userIdEntry.insert(END, str(self._defaultUserId))
        self._userIdEntry.grid(row=3, column=2)

        self._allMoviesListBox.grid(row=1, column=1)
        self._allMoviesListBox.bind('<<ListboxSelect>>', self._onSelect)

        self._similarMoviesListBox = Listbox(self._rightFrame, width=60, height=30)
        endExecutionButton = Button(self._buttonsFrame, text='Exit')
        getSimilarMoviesButton = Button(self._buttonsFrame, text='Get similar movies')
        getUserMoviesButton = Button(self._buttonsFrame, text='Get user movies')

        endExecutionButton.grid(row=1, column=3)
        endExecutionButton.bind('<Button-1>', self._endExecution)
        getSimilarMoviesButton.grid(row=2, column=3)
        getSimilarMoviesButton.bind('<Button-1>', self._getSimilarMovies)
        getUserMoviesButton.grid(row=3, column=3)
        getUserMoviesButton.bind('<Button-1>', self._getUserMovies)

        master.mainloop()
