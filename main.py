# CS 172-060 Tram Le14284628 (ttl43)
# import spellchecker package
from spellchecker import spellchecker as SP

# make a function
def get_file():
    #Welcome user to the program
    print("Welcome to Text File Spellchecker.")
    
    #import the dictionary file and apply the function in SP
    sp = SP("english_words.txt")
    #while True:
    #    dictionary = input("Enter the name of the file to populate dictionary:")
    #    sp = SP(dictionary)
    #    if '0' in str(sp).split(' '):
    #        print("Could not open file.")
    #    else:
    #        break
    
    #read the word file and open it. If the file user entered is not there say could not open and prompt them to enter another name    
    while True:
        name = input("Enter the name of the file to read:\n") #ask the user for the file name input
        try:
            file_content = open(name, "r") #try to open the file by the given name
            return file_content, sp
        except Exception as e:
            print("Could not open file.") #if encounter an error while open the file, go to exception and show user that the program can't open what they input

file_content, sp = get_file()
#count total words and failed words
total_words = 0
failed_words = 0
#go through the lines in the file and label the lines with number
for count,line in enumerate(file_content):
#go through every words in the line to see if it'll pass according to the dictionary
    for word in line.split(): #split the line to get the word and eliminate white spaces
        total_words += 1 #count the total amount of words in the document
        if sp.check(word.strip("\n")) == False: # check to see if the word is in the dictionary. If it's not then it's False
            failed_words += 1 #count the total amount of failed (false) words in the document
            print(f"Possible Spelling Error on line {count+1}: {word}")#print any word that might be an error and the line that it's on. Strip excess blank lines    
   
print("{:,}".format(total_words-failed_words), "words passed spell checker.")#passed words
print(failed_words, "words failed spell checker.")#failed words
print(f"{round(((total_words-failed_words)/total_words)*100, 2)}% of the words passed.") #calculate and show the % of passed words