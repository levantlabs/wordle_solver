#Create a small wordle simulator to test the analysis in
#Manar El-Chammas

#import matplotlib.pyplot as plt
import numpy as np

class Wordle_Simulator:

    def __init__(self, wordSize, tryCount):
        self.wordLen = wordSize #Length of word
        self.tryMax = tryCount #Number of tries
        self.choiceStatus = np.zeros([tryCount, wordSize]) #Contains details on status of choices
        self.letterList = list(range(65,+65+26))
        for i in range(len(self.letterList)):
            self.letterList[i] = chr(self.letterList[i])
        
        #print(self.letterList)

    def initializeFile(self, filename):
        self.wordfilename = filename
        print('Filename has been initialized to {}'.format(self.wordfilename))
        

    def loadWords(self):
        #Load words into a list
        my_file = open(self.wordfilename, 'r')
        wordlist_temp  = my_file.read().splitlines()[2:] #Remove first two lines
        my_file.close()
        self.wordlist = []
        for i in wordlist_temp:
            self.wordlist.append(i.upper())
        #print(self.wordlist)
        print('Word list has been created.  Example word = {}.  Total number of words = {}'.format(self.wordlist[0], len(self.wordlist)))

    def filterWords(self, N):
        #Number of letters = N
        self.wordlist_N = []
        if N < 1:
            print('Error: letters need to be larger than 1')
            exit()

        #Now, reduce list to letters equal to N
        for idx, w in enumerate(self.wordlist):
            if len(w) == N:
                self.wordlist_N.append(w)

        print('Word list has been filtered.  Example word = {}. Total number of words = {}'.format(self.wordlist_N[0], len(self.wordlist_N)))

    def chooseWord(self):
        indW = np.random.randint(len(self.wordlist_N))
        targetWord = self.wordlist_N[indW]
        print('Target word = {}'.format(targetWord))
        self.targetWord = targetWord

    def compareWord(self, choice, tryNo):
        #Go through letter of choice, and compare to targetWord
        #tryNo is the iteration
        ch = choice.upper() #Convert to upper case to standardize
        

        #First compare each letter to targetWord
        for i in range(len(ch)):
            #Remove letter from selection
            letterInd = ord(ch[i]) - 65
            self.letterList[letterInd] = ''
            for j in range(len(self.targetWord)):
                if j == i: #Skip this, doing the analysis later
                    continue
                if ch[i] == self.targetWord[j]:
                    self.choiceStatus[tryNo, i] = 1#1 means it exists, just different position
            if ch[i] == self.targetWord[i]:
                self.choiceStatus[tryNo, i] = 2 #2 means the spot matches the letter

        if ch == self.targetWord: # you are done
            return True
        return False

                
        

        
        
        
            





if __name__ == "__main__":
    print('Begin wordle simulator...')

    w = Wordle_Simulator(5, 6)
    w.initializeFile('words/Collins_Scrabble_Words_2019.txt')
    w.loadWords()
    w.filterWords(N=5)

    #Choose target word
    w.chooseWord()
    w.targetWord = 'PANIC'

    #plt.figure()
    #plt.ion()
    #plt.show()

    numIter = 6
    for i in range(numIter):
        txt = input('Enter word: ')
        print('You entered {}.  Length = {}'.format(txt, len(txt)))
        status = w.compareWord(txt, i)
        print('Result = {}'.format(status))
        print('Choice Status = \n{}'.format(w.choiceStatus))
        print('Remaining letters = \n{}'.format(w.letterList))

        if status == True:
            print('YOU SOLVED IT!')
            exit()
        

    
    
