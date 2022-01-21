#Analyze wordle file
#Manar El-Chammas

import numpy as np
import collections
import wordle_simulator as wrd
#import matplotlib.pyplot as plt

#Notes: want to choose a single word that has the most letters
#and out of these, the word that is most likely

class WordleSolver:

    #Class for wordle analysis
    def __init__(self, debug=False):
        self.wordfilename = ''
        self.wordlist = []
        self.wordlist_N = []
        self.Nletters = 26 #26 letters
        self.lettersE = [] #Excluded letters
        self.lettersI = [] #Included letters
        self.lettersIPos = [] #Included letters, final position
        self.lettersIPosE = [] #Included letters, not in this position
        self.debug = debug

    def initializeFile(self, filename):
        self.wordfilename = filename
        if self.debug:
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
        i
        #print(self.wordlist)
        if self.debug:
            print('Word list has been created.  Example word = {}.  Total number of words = {}.'.format(self.wordlist[0], len(self.wordlist)))

    def filterWords(self, N, M = -1):
        self.wordlist_N = []
        #Number of letters = N
        if N < 1:
            print('Error: letters need to be larger than 1')
            exit()

        #Now, reduce list to letters equal to N
        for idx, w in enumerate(self.wordlist):
            if len(w) == N:
                self.wordlist_N.append(w)

        if M > 0: #Limit words
            self.wordlist_N = self.wordlist_N[0:M]

        if self.debug:
            print('Word list has been filtered.  Example word = {}. Total number of words = {}'.format(self.wordlist_N[0], len(self.wordlist_N)))

    def getDistribution(self, words):
        #Get distribution of letters
        #Create a matrix, in order to get distribution of each letter in each position
        #if len(words) == 0: #No words in the list
        self.count = np.zeros([self.Nletters, 5])#len(words[0])])
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

        if self.debug:
            print('Letter distribution calculated')
        #print('Letter Distribution Is:')
        #for i in range(26):
        #    ch = chr(i+65)
        #    di = self.letter_distribution_global[i]
        #    print('Letter {}: {}'.format(ch, di))

        #exit()

    def reduceWordList(self, words):
        #Choose words that don't have repeated letters
        self.reducedWordList = []
        for idx, w in enumerate(words):
            if len(set(w)) == len(w):
                #All unique letters
                self.reducedWordList.append(w)

        if self.debug:
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
                    keepWordE = False
                
            if len(lettersIncluded) > 0: #If no list, assume all are included
                keepWordI = True
                for j in range(len(lettersIncluded)):
                    if lettersIncluded[j] not in words[i]: #This letter isn't in the word, remove it
                        keepWordI = False
                    #Iterate over positions
                    for k in range(1):#len(lettersIncludedPosition[j])):
                        if lettersIncludedPosition[j][k] > -1: #it isn't -1, that means we know the position
                            if words[i][lettersIncludedPosition[j][k]] != lettersIncluded[j]: #the letter isn't in the right place
                                keepWordI = False #Also remove it
                    
                     #Not just if we don't know, you also know where it isn't
                    for k in range(len(lettersIncludedPositionX[j])):
                        #print('*** {}, {}'.format(lettersIncluded[j], lettersIncludedPositionX[j][k]))
                        if words[i][lettersIncludedPositionX[j][k]] == lettersIncluded[j]: #This woudl have been green
                            #print(words[i])
                            keepWordI = False #Remove it
                               

            if keepWordE and keepWordI:
                latestWords.append(words[i])
                #print(words[i])

        if self.debug:
            print('Removed words with letters = {}'.format(lettersExcluded))
            print('Word list contained {} words'.format(len(words)))
        self.wordlist_N = latestWords
        #print(self.wordlist_N)
        if self.debug:
            print('Word list now contains {} words'.format(len(self.wordlist_N)))

    def getEntropyWord(self, word, simplified = False):
        tempEntropy = 0
        for j in range(len(word)):
            len_Int = ord(word[j]) - 65
            dist = self.letter_distribution[len_Int][j]
            if simplified == True:
                dist = self.letter_distribution_global[len_Int]
            tempEntropy -= dist*np.log(dist)/np.log(2)

        return tempEntropy
    
    def getEntropy(self, words, simplified = False):
        #Calculate entropy for each word
        #Go through each word list and calculate distribution
        #Do this for unique letters though
        tempEntropy = np.zeros(len(words))+1
        for idx, w in enumerate(words):
            #Convert each letter to an integer, and subtract 65 (for capital letters, 97 for small letters
            #uw = list(set(w)) #Uniquifiy it
            uw = w
            #print(list(uw))
            #for j in range(len(uw)):
            #    len_Int = ord(uw[j]) - 65
            #    #Change distributino to P(x|when x is in position j)
            #    dist = self.letter_distribution[len_Int][j]
            #    if simplified == True:
            #        dist = self.letter_distribution_global[len_Int]
            #    tempEntropy[idx] -= dist*np.log(dist)/np.log(2)
            tempEntropy[idx] = self.getEntropyWord(w, simplified=simplified)
            #tempEntropy[idx] *= dist
                #Need to look at two distributions: distribution of position i, and distribution of being in the word
                #both are valuable?  

        self.entropy_N = tempEntropy #Entropy for N letter words

    def getEntropyLetters(self, simplified = False, prob=1):
        #Just get then entropy of all the letters, ignoring the words
        #Prob is the probability that this event can happen
        tempEntropy = 0
        for i in range(self.Nletters):
            if simplified == True: #Just look at letter distribution, ignoring position
                dist = prob*self.letter_distribution_global[i]
                tempEntropy -= dist*np.log(dist)/np.log(2)
            else:
                for j in range(5): #5 letters
                    #print(i)
                    dist = prob*self.letter_distribution[i][j]
                    if dist > 0:
                        tempEntropy -= dist*np.log(dist)/np.log(2)
        return tempEntropy
            

    def getMutualInformation_v2(self, words):
        #Second attempt
        #Go through each word.
        #Then, calculate the new distribution if each word had N letters match
        #Then, calculate the resulting entropy
        #The word that minimizes the entropy, results in the most information

        #For now, best to calculate the entropy of the letters, not of the individual words

        baselineEntropy = self.getEntropyLetters(simplified = False)
        print('Baseline Entropy is: {}'.format(baselineEntropy))

        originalWordList = self.wordlist_N
        originalDistribution_global = self.letter_distribution_global
        originalDistribution = self.letter_distribution
        #self.wordlist_N = 0
        #print(originalWordList)
        c = 3
        wordEntropy = np.zeros(len(originalWordList))
        for idx, w in enumerate(originalWordList):
            print('{}: Analyzing word  {}'.format(idx, w))
            newEntropy = 0
            #if 0, then not in the word
            #if 1, then is somewhere in the word, but not in this position
            #if 2, then it is in the word, and is in this position
            #
            for i1 in range(c):
                for i2 in range(c):
                    for i3 in range(c):
                        for i4 in range(c):
                            for i5 in range(c):
                                lettersInc = []
                                lettersIncPos = []
                                lettersIncPosExc = [] #Included, but not in this position
                                lettersExc = []
                                prob = 1
                                #Choice: is this letter in or not (binary), for 5 letters
                                lc = ord(w[0])-65 #letter 0
                                cPos = 0
                                if i1 == 1: #Included somewhere
                                    lettersInc.append(w[cPos])
                                    lettersIncPos.append([-1])
                                    lettersIncPosExc.append([cPos])
                                    prob *= (originalDistribution_global[lc] - originalDistribution[lc][cPos]) #We know it isn't in the first position
                                elif i1 == 0: #Not included
                                    lettersExc.append(w[cPos])
                                    prob *= (1-originalDistribution_global[lc])
                                elif i1 == 2: #Included in this position
                                    lettersInc.append(w[cPos])
                                    lettersIncPos.append([cPos]) #0 position, since we are looking at i1
                                    lettersIncPosExc.append([]) #Don't exclude, we know it is in a position
                                    prob *= originalDistribution[lc][cPos]

                                cPos = 1
                                lc = ord(w[cPos])-65 #letter 0
                                if i2 == 1: #Included somewhere
                                    lettersInc.append(w[cPos])
                                    lettersIncPos.append([-1])
                                    lettersIncPosExc.append([cPos])
                                    prob *= (originalDistribution_global[lc] - originalDistribution[lc][cPos]) #We know it isn't in the first position
                                elif i2 == 0: #Not included
                                    lettersExc.append(w[cPos])
                                    prob *= (1-originalDistribution_global[lc])
                                elif i2 == 2: #Included in this position
                                    lettersInc.append(w[cPos])
                                    lettersIncPos.append([cPos]) #0 position, since we are looking at i1
                                    lettersIncPosExc.append([]) #Don't exclude, we know it is in a position
                                    prob *= originalDistribution[lc][cPos]

                                cPos = 2
                                lc = ord(w[cPos])-65 #letter 0
                                if i3 == 1: #Included somewhere
                                    lettersInc.append(w[cPos])
                                    lettersIncPos.append([-1])
                                    lettersIncPosExc.append([cPos])
                                    prob *= (originalDistribution_global[lc] - originalDistribution[lc][cPos]) #We know it isn't in the first position
                                elif i3 == 0: #Not included
                                    lettersExc.append(w[cPos])
                                    prob *= (1-originalDistribution_global[lc])
                                elif i3 == 2: #Included in this position
                                    lettersInc.append(w[cPos])
                                    lettersIncPos.append([cPos]) #0 position, since we are looking at i1
                                    lettersIncPosExc.append([]) #Don't exclude, we know it is in a position
                                    prob *= originalDistribution[lc][cPos]

                                cPos = 3
                                lc = ord(w[cPos])-65 #letter 0
                                if i4 == 1: #Included somewhere
                                    lettersInc.append(w[cPos])
                                    lettersIncPos.append([-1])
                                    lettersIncPosExc.append([cPos])
                                    prob *= (originalDistribution_global[lc] - originalDistribution[lc][cPos]) #We know it isn't in the first position
                                elif i4 == 0: #Not included
                                    lettersExc.append(w[cPos])
                                    prob *= (1-originalDistribution_global[lc])
                                elif i4 == 2: #Included in this position
                                    lettersInc.append(w[cPos])
                                    lettersIncPos.append([cPos]) #0 position, since we are looking at i1
                                    lettersIncPosExc.append([]) #Don't exclude, we know it is in a position
                                    prob *= originalDistribution[lc][cPos]


                                cPos = 4
                                lc = ord(w[cPos])-65 #letter 0
                                if i5 == 1: #Included somewhere
                                    lettersInc.append(w[cPos])
                                    lettersIncPos.append([-1])
                                    lettersIncPosExc.append([cPos])
                                    prob *= (originalDistribution_global[lc] - originalDistribution[lc][cPos]) #We know it isn't in the first position
                                elif i5 == 0: #Not included
                                    lettersExc.append(w[cPos])
                                    prob *= (1-originalDistribution_global[lc])
                                elif i5 == 2: #Included in this position
                                    lettersInc.append(w[cPos])
                                    lettersIncPos.append([cPos]) #0 position, since we are looking at i1
                                    lettersIncPosExc.append([]) #Don't exclude, we know it is in a position
                                    prob *= originalDistribution[lc][cPos]

                                    
                                #if i2 == 1:
                                #    lettersInc.append(w[1])
                                #    prob *= originalDistribution_global[ord(w[1])-65]
                                #elif i2 == 0:
                                #    lettersExc.append(w[1])
                                #    prob *= (1-originalDistribution_global[ord(w[1])-65])
                                #if i3 == 1:
                                #    lettersInc.append(w[2])
                                #    prob *= originalDistribution_global[ord(w[2])-65]
                                #elif i3 == 0:
                                #    lettersExc.append(w[2])
                                #    prob *= (1-originalDistribution_global[ord(w[2])-65])
                                #if i4 == 1:
                                #    lettersInc.append(w[3])
                                #    prob *= originalDistribution_global[ord(w[3])-65]
                                #elif i4 == 0:
                                #    lettersExc.append(w[3])
                                #    prob *= (1-originalDistribution_global[ord(w[3])-65])
                                #if i5 == 1:
                                #    lettersInc.append(w[4])
                                #    prob *= originalDistribution_global[ord(w[4])-65]
                                #elif i5 == 0:
                                #    lettersExc.append(w[4])
                                #    prob *= (1-originalDistribution_global[ord(w[4])-65])

                                #Reduce words bsaed on this
                                #Assume don't know where it is
                                #Just set -1 for all, and don't ignore any positions
                                #print(lettersExc)
                                #print(lettersInc)
                                #lettersIncPos = [[-1]]*len(lettersInc)
                                #print(lettersIncPos)
                                self.removeWords(lettersExc, lettersInc, lettersIncPos, lettersIncPosExc, originalWordList )
                                #print(len(originalWordList))
                                #print(self.wordlist_N)
                                self.getDistribution(self.wordlist_N)
                                #Now, I can get entropy of new list
                                newEntropy += self.getEntropyLetters(prob=prob)
            #print(newEntropy)
            wordEntropy[idx] = newEntropy
                                #self.removeWords(lettersInc
                                #Now, get new distribution
            
            #Go over word, look at all combinations, calculate distribution, get entropy
            #For now, assume
            #Assume I pick this word.  What is the probability that none of the letters are correct?
        #Now, choose minimum entropy
        print('Minimum entropy is {}, with word {}'.format(np.min(wordEntropy), originalWordList[np.argmin(wordEntropy)]))
        print('Maximum entropy is {}, with word {}'.format(np.max(wordEntropy), originalWordList[np.argmax(wordEntropy)]))
        self.wordlist_N = originalWordList #Replace
        return originalWordList[np.argmin(wordEntropy)]
        exit()

    def getMutualInformation(self, words): #Get mutual information for words
        #How to approach this?
        #Each word has its entropy
        #How does selecting another word reduce its entropy
        #This is mutual information: H(X) - H(X|Y)
        #We want to maximize the information, to reduce the overall entropy

        #Start with simplified entropy (not positional
        self.getEntropy(words, simplified = True)

        self.mutualInformation = np.zeros(len(words))

        for i in range(len(words)):
            print('{}: {}'.format(i, words[i]))
            
            for j in range(len(words)):
                w1 = words[i]
                w2 = words[j]

                hx_y_temp = 0

                #I = H(X) + H(Y) - H(Y, X)
                #Basically, one way is to see what letters are left
                #And calculate the entropy of that
                #In this case, assume that all the letters are shared.  Then H(X|Y) is 0
                #The information gain can be split into several parts
                #1. If one letter matches
                #2. If two letters match
                #3... etc., if all letters match
                #Can also start looking at wether they match in the same spot, but that is too complex for now
                #Alternatively, we can look at: if no letters match, and then at least one letter matches is 1-p
                match = 0 #Number of letters that match
                for k in w1:
                    #if k not in w2:
                    if k in w2:
                        match += 1 #Increment, this is number of matches letters
                    else:
                        len_Int = ord(k) - 65
                        dist = self.letter_distribution_global[len_Int]
                        hx_y_temp -= dist*np.log(dist)/np.log(2)

                #If match == 0, then it shares no letters.  Then we know for a fact that we have gained full information
                #hx_y_temp is the conditionalentropy
                #Get mutual information for word j given word i is selected
                if match == 0:
                    mi_x_y_temp = self.entropy_N[j]
                else:
                    mi_x_y_temp = self.entropy_N[j] - hx_y_temp
                self.mutualInformation[i] += mi_x_y_temp #This is how much information this choice results in

        print(self.mutualInformation)
        ind = np.argmax(self.mutualInformation)
        print('Information = {}, word = {}'.format(self.mutualInformation[ind], words[ind]))

        exit()

        return words[ind]
                    

        

        x = 1

        
    def getSeparation2(self, words):
        #Second attempt, but want to make it faster
        #Look at how many words contain 0 letters, 1 letter, 2, ... 5 letters
        self.contains = np.zeros([len(words), len(words[0])+1])

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
                        matchScore += self.letter_distribution_global[ord(words[i][k]) - 65]
                    #Otherwise, it is True, and there is no overlap
                #if noMatch:
                #    tempcount += 1
                #With this one, the best word is TRIED
                self.contains[i][matchCount] += 1#matchScore #1#tempcount
                #With this, the best word is 
                #self.contains[i][matchCount] += matchScore
            
                    
                
            #for k in range(len(words[i])):
                #Get how many words would be matched with one letter
            #    self.contains[i][k] = self.letter_distribution_global[ord(words[i][k]) - 65]

        if self.debug:
            print(self.contains)
            print(np.max(np.asarray(self.contains), axis=1))
        ind = np.argmin(np.max(np.asarray(self.contains), axis=1))
        if True: ###self.debug:
            print('Minimum of the maximum is {}, with word {}'.format(self.contains[ind], words[ind]))

        return words[ind]
        #print('Minimum overlap is {}, for word {}'.format(np.min(self.containsnone), words[np.argmin(self.containsnone)]))
                

    def sortEntropy(self, words, entropy):
        ent_sort = np.sort(entropy)
        ent_sort_ind = np.argsort(entropy)
        maxval = np.max(entropy)
        numPrint = min(len(words), 5)
        if self.debug:
            print('Max Entropy = {}'.format(maxval))
            print('{} top entropy enntries = {}'.format(numPrint, entropy[ent_sort_ind[-numPrint:]]))
            for i in range(numPrint):
                print('----**---- {}, Entropy = {}'.format(words[ent_sort_ind[-1-i]], entropy[ent_sort_ind[-1-i]]))
              
                                                   
        
        ind = np.argmax(entropy)
        #print(words[entropy == maxval])
        if self.debug:
            print('Highest entropy word is: ')
        for i in range(len(words)):
            if entropy[i] == maxval: #>= maxval - 0.00000001:
                if self.debug:
                    print('--- {}'.format(words[i]))
                return words[i]
        #minval = np.min(entropy)
        #print('Minimum entropy word is: ')
        #for i in range(len(words)):
        #    if entropy[i] <= minval + 0.1:
        #        print('--- {}'.format(words[i]))
        #print('Lowest entropy word is: {}'.format(words[entropy == min(entropy)]))

    def strategy_A(self):
        #Look at probability for a letter in pos i, and then calculate entropy
        self.removeWords(self.lettersE, self.lettersI, self.lettersIPos, self.lettersIPosE, self.wordlist_N)
        #Get distribution
        self.getDistribution(self.wordlist_N)
        self.getEntropy(self.wordlist_N)
        bestguess = self.sortEntropy(self.wordlist_N, self.entropy_N)
        print('The best guess in this iteration is: {}'.format(bestguess))
        return bestguess

    def strategy_B(self):
        #Look at probability of letter (regardless of position)
        self.removeWords(self.lettersE, self.lettersI, self.lettersIPos, self.lettersIPosE, self.wordlist_N)
        #Get distribution
        self.getDistribution(self.wordlist_N)
        self.getEntropy(self.wordlist_N, simplified=True)
        bestguess = self.sortEntropy(self.wordlist_N, self.entropy_N)
        print('The best guess in this iteration is: {}'.format(bestguess))
        return bestguess

    def strategy_C(self):
        #Look at probability of letter (regardless of position)
        self.removeWords(self.lettersE, self.lettersI, self.lettersIPos, self.lettersIPosE, self.wordlist_N)
        #Get distribution
        self.getDistribution(self.wordlist_N)
        #self.getEntropy(self.wordlist_N, simplified=True)
        
        #bestguess = self.sortEntropy(self.wordlist_N, self.entropy_N)
        bestguess = self.getSeparation(self.wordlist_N)
        print('The best guess in this iteration is: {}'.format(bestguess))
        return bestguess


    def strategy_D(self):
        #Look at probability of letter (regardless of position)
        self.removeWords(self.lettersE, self.lettersI, self.lettersIPos, self.lettersIPosE, self.wordlist_N)
        #Get distribution
        self.getDistribution(self.wordlist_N)
        #self.getEntropy(self.wordlist_N, simplified=True)
        bestguess = self.getMutualInformation_v2(self.wordlist_N)
        #bestguess = self.sortEntropy(self.wordlist_N, self.entropy_N)
        print('The best guess in this iteration is: {}'.format(bestguess))
        return bestguess
    
    def initializeWordleGame(self, num = -1):
        self.wordSize = 5
        self.tryCount = 6
        self.game = wrd.Wordle_Simulator(self.wordSize, self.tryCount)
        self.game.initializeFile(self.wordfilename)
        self.game.loadWords()
        self.game.filterWords(self.wordSize)
        if num == -1: #Choose random word
            #self.game.targetWord = 'PANDA' #'PANIC'
            self.game.chooseWord()
            #self.game.targetWord = 'RUFFS'
            #self.game.targetWord = 'PANIC'
        else:
            self.game.chooseWord(num)
            #self.game.targetWord = 'PANDA'
            #self.game.targetWord = 'RUFFS'
            self.game.targetWord = 'PANIC'
            #self.game.targetWord = 'PROXY'

    def playWordleGame(self, strategy):
        #Play the game for self.tryCount
        status = False
        for i in range(self.tryCount):
            if strategy == 1:
                bestguess = self.strategy_A()
            elif strategy == 2:
                bestguess = self.strategy_B()
            elif strategy == 3:
                if i == 0:
                    bestguess = 'TRIED' #Can run this to find out
                    #bestguess = 'DREAM'
                else:
                    bestguess = self.strategy_C()
            elif strategy == 4:
                if i == -1:
                    bestguess = 'SEXES'
                    bestguess = 'AEROS'
                else:
                    bestguess = self.strategy_D()
            #bestguess = 'PANIC'
            status = w.game.compareWord(bestguess, i)
            if self.debug:
                print('Result = {}'.format(status))
                print('Choice Status = \n{}'.format(self.game.choiceStatus))
            #print('Remaining letters = \n{}'.format(self.game.letterList))

            if status == True:
                print('You solved it in {} steps'.format(i+1))
                break

            self.interpretPlay(bestguess, self.game.choiceStatus, i)

            
        if status == False: #Did not succeed
            print('You did not succeed.')

        return status, i+1 #Number of steps needed 

        #print(i)

    def interpretPlay(self, currentGuess, gameStatus, step):
        #Figure out which letters were used so far, and if they are valid
        #currentGuess is the guess for this round
        #gamestatus is the status of the game so far
        #step is which step this is on
        currentStatus = gameStatus[step]
        #print(currentStatus)
        #Go through each letter in the word and see what the status is
        for i in range(len(currentGuess)):
            val = currentStatus[i]
            if val == 0: #Does not exist
                self.lettersE.append(currentGuess[i])
            elif val == 1 or val == 2: #Exists, but not in this position
                #Check to see if it already exists in lettersI
                if currentGuess[i] not in self.lettersI:
                    self.lettersI.append(currentGuess[i])
                    if val == 1:
                        self.lettersIPos.append([-1]) #Don't know where it is, so it is -1
                        self.lettersIPosE.append([i]) #We know it isn't in this position
                    elif val == 2:
                        self.lettersIPos.append([i]) #Store position
                        self.lettersIPosE.append([]) #Store empty list
                else: #It already is in the list
                    ind = self.lettersI.index(currentGuess[i])
                    if self.lettersIPos[ind][0] > -1: #We know where it is.  But need to store multiple spots
                        #Only add this position if it isn't already there
                        if i not in self.lettersIPos[ind]:
                            print('debug...')
                            print(self.lettersIPos[ind])
                            if val == 2: #Only append if val == 2
                                self.lettersIPos[ind].append(i)
                            print(self.lettersIPos[ind])
                    elif self.lettersIPos[ind][0] == -1: #Update to position
                        if val == 2:#If the right position, update position
                            self.lettersIPos[ind] = []
                            self.lettersIPos[ind].append(i)
                        elif val == 1: #Still not found there, just update the exclude pos
                            self.lettersIPosE[ind].append(i)
                        #print(self.lettersIPos[ind])
            #elif val == 2: #It exists here, save the position if it doesn't exist

        if self.debug:
            print('**ANALYSIS**')
            print(self.lettersI)
            print(self.lettersIPos)
            print(self.lettersIPosE)
            print(self.lettersE)
        #self.lettersE = [] #Excluded letters
        #self.lettersI = [] #Included letters
        #self.lettersIPos = [] #Included letters, final position
        #self.lettersIPosE = [] #Included letters, not in this position
                      
        
        

soloplay = False

if soloplay: #Playing online  
    w = WordleSolver()
    w.initializeFile('words/sgb-words.txt')
    w.loadWords()
    w.filterWords(N=5)

    currentGuess = 'TRIED'
    currentGuess = 'BURLY'
    currentGuess = 'GNARL'
    currentGuess = 'LATER'
    currentGuess = 'MORAL'
    currentStatus = [[0,1,0,0,0],
                     [0,0,1,1,0],
                     [0,0,1,1,1],
                     [1,1,0,0,1],
                     [0,2,1,2,1],
                     [0,0,0,0,0]]
    step = 4
    w.interpretPlay(currentGuess, currentStatus, step)
    bestguess = w.strategy_C()
    print('Next guess is: {}'.format(bestguess))

    

    exit()

numWords = 1
totalStatus = np.zeros(numWords)
totalSteps = np.zeros(numWords)
for i in range(numWords):
    print('*****************')
    print('ITERATION = {}'.format(i))
    w = WordleSolver();
    #w.initializeFile('words/Collins_Scrabble_Words_2019.txt')
    w.initializeFile('words/sgb-words.txt')
    w.loadWords()
    w.filterWords(N=5, M=100)#, M = 2000)
    #w.strategy_A()
    w.initializeWordleGame(num = i)
    status, stepcount = w.playWordleGame(strategy = 4)
    totalStatus[i] = status
    totalSteps[i] = stepcount

print('***************')
print('*** Final Results ****')
print('TOTAL WORDS = {}'.format(numWords))
print('TOTAL SUCCESS = {}'.format(np.sum(totalStatus)))
print('AVERAGE STEPS = {}'.format(np.mean(totalSteps)))
print('MAX STEPS = {}, MIN STEPS = {}'.format(np.max(totalSteps), np.min(totalSteps)))



