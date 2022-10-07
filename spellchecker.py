#CS 172
#Drexel University 2018
#Professor Mark Boady and Professor Matthew Burlick

#Example Class for checking spelling.
class spellchecker:
    #Give the name of a file containing
    #a list of words spelled correctly.
    #Creates a spelling dictionary from the file.
    #Assumes file exits and throws exception otherwise
    def __init__(self, word_file):
        self.__words = set()
        try:
            f = open(word_file, "r")
            for word in f:
                #use only lower case
                word = word.lower()
                word = word.strip()
                self.__words.add(word)
        except Exception as e:
            print(e)
    
    #Print the Spelling Dictionary variable
    #Prints the total number of words in the dictionary.
    def __str__(self):
        res = "<Spelling Dictionary with "
        res += "{:,.0f}".format(len(self.__words))
        res += " words.>"
        return res
    
    #Check if a word is spelled correctly
    #Returns True/False
    #True = Word is in spelling dictionary
    #False = Word is not in dictionary
    def check(self, word):
        #Clean up word.
        #Remove white-space on either side
        word = word.strip()
        #switch to lower case
        word = word.lower()
        #remove common punctuation
        P = ".,/;:!\"'"
        for p in P:
            word = word.replace(p, "")
        #Check if word in dictionary
        if word in self.__words:
            return True
        else:
            return False


if __name__=="__main__":
    #Run a simple Test of the class
    SP = spellchecker("english_words.txt")
    print(SP)
    L = ["orange","ORanGe","ORANGE","ORANGE.","   ORAnge     ","oranges","gloob"]
    for a in L:
        print("Checking Dictionary for \""+a+"\"")
        print("Result:", SP.check(a))