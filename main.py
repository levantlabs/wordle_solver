#Analyze wordle file
#Manar El-Chammas

import numpy as np
import collections
#import matplotlib.pyplot as plt

#Notes: want to choose a single word that has the most letters
#and out of these, the word that is most likely

class Wordle:

    #Class for wordle analysis
    def __init__(self):
        self.wordfilename = ''
        self.wordlist = []
        self.wordlist_N = []
        self.Nletters = 26 #26 letters

    def initializeFile(self, filename):
        self.wordfilename = filename
        print('Filename has been initialized to {}'.format(self.wordfilename))

    def loadWords(self):
        #Load words into a list
        my_file = open(self.wordfilename, 'r')
        wordlist_temp  = my_file.read().splitlines()[2:] #Remove first two lines
        #convert to upper
        self.wordlist = []
        for i in wordlist_temp:
            self.wordlist.append(i.upper())
        my_file.close()
        #print(self.wordlist)
        print('Word list has been created.  Example word = {}.  Total number of words = {}.'.format(self.wordlist[0], len(self.wordlist)))

    def filterWords(self, N):
        self.wordlist_N = []
        #Number of letters = N
        if N < 1:
            print('Error: letters need to be larger than 1')
            exit()

        #Now, reduce list to letters equal to N
        for idx, w in enumerate(self.wordlist):
            if len(w) == N:
                self.wordlist_N.append(w)

        print('Word list has been filtered.  Example word = {}. Total number of words = {}'.format(self.wordlist_N[0], len(self.wordlist_N)))

    def getDistribution(self, words):
        #Get distribution of letters
        #Create a matrix, in order to get distribution of each letter in each position
        self.count = np.zeros([self.Nletters, len(words[0])])
        self.countglobal = np.zeros(self.Nletters) #This is probability over all words, independent of position

        #Original approach
        #Not based on position in word
                
        #combined = ''.join(words)
        #self.letter_dict = collections.Counter(combined)
        #self.letters = sorted(self.letter_dict.keys())
        #print(self.letters)
        #for i in range(26): #26 letters
        #    self.count[i] = self.letter_dict[chr(i+65)]

        #New approach: based on position in word
        for i in range(len(words)): #Iterate over all owords
            for j in range(len(words[i])): #Iterate over letters
                ind = ord(words[i][j]) - 65 #Index of letter, normalized to 0-->25
                self.count[ind][j] += 1 #How often this letter occurs in this position
                self.countglobal[ind] += 1

        #totalCount = np.sum(self.count)
        #End of new approach
                              
        self.letter_distribution = self.count / np.sum(self.count)
        self.letter_distribution_global = self.countglobal / np.sum(self.countglobal)
        
        print('Letter distribution calculated')
        print('Letter Distribution Is:')
        for i in range(26):
            ch = chr(i+65)
            di = self.letter_distribution_global[i]
            print('Letter {}: {}'.format(ch, di))

        #exit()

    def reduceWordList(self, words):
        #Choose words that don't have repeated letters
        self.reducedWordList = []
        for idx, w in enumerate(words):
            if len(set(w)) == len(w):
                #All unique letters
                self.reducedWordList.append(w)

        print('Reduced word list has length of {}'.format(len(self.reducedWordList)))
        #print(self.reducedWordList)

    def removeWords(self, lettersExcluded, lettersIncluded, lettersIncludedPosition, lettersIncludedPositionX, words):
        #Remove words that contain certain letters (lettersExcluded)
        #and only keep words that contain certain letters (lettersIncluded)
        #if lettersIncludedPosition is not -1, then we know exactly where it goes, filter words that contain those
        #lettersIncludedPositionX contains the positions that it is not in (if lettersIncludedPosition is -1
        #letters is in a list form
        latestWords = []
        #keepWord = True
        keepWordI = True
        for i in range(len(words)):
            keepWordE = True
            for j in range(len(lettersExcluded)):
                if lettersExcluded[j] in words[i]: #all is good
                    #print(words[i])
                    keepWordE = False
                    #continue
                #else:
                #    keepWord = True

            
            if len(lettersIncluded) > 0: #If no list, assume all are included
                keepWordI = True
                for j in range(len(lettersIncluded)):
                    if lettersIncluded[j] not in words[i]: #This letter isn't in the word, remove it
                        keepWordI = False
                    if lettersIncludedPosition[j] > -1: #it isn't -1, that means we know the position
                        if words[i][lettersIncludedPosition[j]] != lettersIncluded[j]: #the letter isn't in the right place
                            #print('***')
                            #print(words[i][lettersIncludedPosition[j]])
                            #print(lettersIncluded[j])
                            keepWordI = False #Also remove it
                     #If -1, then look at the relevant entry of lettersIncludedPositionX, and remove words with letters in that position
                     #This is a set
                     #Not just if we don't know, you also know where it isn't
                    for k in range(len(lettersIncludedPositionX[j])):
                        if words[i][lettersIncludedPositionX[j][k]] == lettersIncluded[j]: #This woudl have been green
                            keepWordI = False #Remove it
                               

            if keepWordE and keepWordI:
                latestWords.append(words[i])
                #print(words[i])

        print('Removed words with letters = {}'.format(lettersExcluded))
        print('Word list contained {} words'.format(len(words)))
        self.wordlist_N = latestWords
        #print(self.wordlist_N)
        print('Word list now contains {} words'.format(len(self.wordlist_N)))

    def getEntropy(self, words):
        #Calculate entropy for each word
        #Go through each word list and calculate distribution
        #Do this for unique letters though
        tempEntropy = np.zeros(len(words))+1
        for idx, w in enumerate(words):
            #Convert each letter to an integer, and subtract 65 (for capital letters, 97 for small letters
            #uw = list(set(w)) #Uniquifiy it
            uw = w
            #print(list(uw))
            for j in range(len(uw)):
                len_Int = ord(uw[j]) - 65
                #Change distributino to P(x|when x is in position j)
                dist = self.letter_distribution[len_Int][j]
                dist = self.letter_distribution_global[len_Int]
                tempEntropy[idx] -= dist*np.log(dist)/np.log(2)
                #tempEntropy[idx] *= dist
                #Need to look at two distributions: distribution of position i, and distribution of being in the word
                #both are valuable?  

            

        self.entropy_N = tempEntropy #Entropy for N letter words

        #plt.figure()
        #plt.plot(np.sort(self.entropy_N))
        #print(np.max(np.sort(self.entropy_N)[-3:-2]))
        #print('10 max entropy values are: {}'.format(np.sort(self.entropy_N)[-2:]))
        #plt.show()

    def getSeparation(self, words):
        #Look at each letter
        #How many words in the set that contain this letter
        #How many words in the set that doesn't contain any letter
        #Look at count that contains only zero letters one letter, 2 letters, ... 5 letters
        self.contains = np.zeros([len(words), len(words[0])+1])
        self.containsnone = np.zeros(len(words))

        #Either no letters match
        #or one letter matches
        #or 2 letters match
        #... or 4 letters match.  This is all there is, 
        
        for i in range(len(words)):
            tempcount = 0
            for j in range(len(words)):
                if i == j:
                    continue

                #Get how many words won't be matched with anything
                noMatch = True
                matchCount = 0
                matchScore = 0
                for k in range(len(words[i])):
                    #Don't double count the same letter
                    if words[i][k] in words[i][:k]:
                        continue #Don't count, go to the next letter
                    if words[i][k] in words[j]: #It exists somewhere
                        #noMatch = False
                        matchCount += 1#Increment if there is a match, don't count doubles for now
                        matchScore += 1#self.letter_distribution_global[ord(words[i][k]) - 65]
                    #Otherwise, it is True, and there is no overlap
                #if noMatch:
                #    tempcount += 1
                self.contains[i][matchCount] += matchScore #1#tempcount
            
                    
                
            #for k in range(len(words[i])):
                #Get how many words would be matched with one letter
            #    self.contains[i][k] = self.letter_distribution_global[ord(words[i][k]) - 65]

        print(self.contains)
        print(np.max(np.asarray(self.contains), axis=1))
        ind = np.argmin(np.max(np.asarray(self.contains), axis=1))
        print('Minimum of the maximum is {}, with word {}'.format(self.contains[ind], words[ind]))
        #print('Minimum overlap is {}, for word {}'.format(np.min(self.containsnone), words[np.argmin(self.containsnone)]))
                

    def sortEntropy(self, words, entropy):
        ent_sort = np.sort(entropy)
        ent_sort_ind = np.argsort(entropy)
        maxval = np.max(entropy)
        numPrint = min(len(words), 5)
        print('Max Entropy = {}'.format(maxval))
        print('{} top entropy enntries = {}'.format(numPrint, entropy[ent_sort_ind[-numPrint:]]))
        for i in range(numPrint):
            print('----**---- {}, Entropy = {}'.format(words[ent_sort_ind[-1-i]], entropy[ent_sort_ind[-1-i]]))
              
                                                   
        
        ind = np.argmax(entropy)
        #print(words[entropy == maxval])
        print('Highest entropy word are: ')
        for i in range(len(words)):
            if entropy[i] >= maxval - 0.00000001:
                print('--- {}'.format(words[i]))
        minval = np.min(entropy)
        print('Minimum entropy word is: ')
        #for i in range(len(words)):
        #    if entropy[i] <= minval + 0.1:
        #        print('--- {}'.format(words[i]))
        #print('Lowest entropy word is: {}'.format(words[entropy == min(entropy)]))

    def strategy_A(self):
        #This is the entropy strategy
        lettersI = []
        lettersIPos = []
        lettersIPosE = [[]]
        lettersE = []


        self.removeWords(lettersE, lettersI, lettersIPos, lettersIPosE, self.wordlist_N)
        #Get distribution
        self.getDistribution(self.wordlist_N)
        self.getEntropy(self.wordlist_N)
        self.sortEntropy(self.wordlist_N)
        
            
        
        


w = Wordle();
#w.initializeFile('words/Collins_Scrabble_Words_2019.txt')
w.initializeFile('words/sgb-words.txt')
w.loadWords()
w.filterWords(N=5)
#w.getDistribution(w.wordlist_N) #Do I get distribution before or after
#w.reduceWordList(w.wordlist_N)
#w.getDistribution(w.reducedWordList)
#w.getDistribution(w.wordlist_N)
#w.getEntropy(w.reducedWordList)
#w.sortEntropy(w.reducedWordList, w.entropy_N)
#w.getEntropy(w.wordlist_N)
#w.sortEntropy(w.wordlist_N, w.entropy_N)

#w.getSeparation(w.wordlist_N)

#exit()

#Remove some words
#Also know that if it is yellow, that it isn't in that position
lettersI =   []
lettersIPos = []
lettersIPosE = [] #The included letters are not in these positions
#lettersIPosE = [[], [], [], []]
lettersE = []
w.removeWords(lettersE, lettersI, lettersIPos, lettersIPosE,  w.wordlist_N)
#Get next best word
w.reduceWordList(w.wordlist_N) #Remove repeated letters
#Recalculate word list
#w.getDistribution(w.reducedWordList)
#What if I don't recalculate the distributions
w.getDistribution(w.wordlist_N)
#Get entropy with actual list
#w.getEntropy(w.reducedWordList)
#print('Recommendation with unique letters!')
#w.sortEntropy(w.reducedWordList, w.entropy_N)
w.getEntropy(w.wordlist_N)
#print('Recommendation with multiple letters!')
w.sortEntropy(w.wordlist_N, w.entropy_N)
#w.getSeparation(w.wordlist_N)



