'''
	Check.py is for evaluating your model. 
	Function eval() will print out the accuracy of training and testing data. 
	To call:
        	import Check
        	Check.eval(o_train, p_train, o_test, p_test)
        
	At the end of this file, it also contains how to read data from a file.
'''

#eval:
#   Input: original training labels list, predicted training labels list,
#	       original testing labels list, predicted testing labels list.
#   Output: print out training and testing accuracy
def eval(o_train, p_train, o_test, p_test):
    print('\nTraining Result!')
    accuracy(o_train, p_train)
    print('\nTesting Result!')
    test_accuracy = accuracy(o_test, p_test)
    return test_accuracy

#accuracy:
#   Input: original labels list, predicted labels list
#   Output: print out accuracy
def accuracy(orig, pred):
    
    num = len(orig)
    print("rows", num)
    if(num != len(pred)):
        print('Error!! Num of labels are not equal.')
        return
    match = 0
    for i in range(len(orig)):
        o_label = orig[i]
        p_label = pred[i]
        if(o_label == p_label):
            match += 1
    zzz
    print('***************\nAccuracy: ' + str(float(match)/num) + '\n***************')
    return str(float(match)/num)


#readfile:
#   Input: filename
#   Output: return a list of rows.
def readfile(filename):    
    f = open(filename).read()
    rows = []
    char_split = '\r'
    import platform
    if(platform.system()=='Windows'):
        char_split = '\n'

    for line in f.split(char_split):
        if line.strip():
            rows.append(line.split('\t'));
    return rows 


if __name__ == '__main__':

   import Solution as sl
   labels = sl.DecisionTreeBounded(7)
   if labels == None or len(labels) != 4:
       print('\nError: DecisionTree Return Value.\n')
   else:
       eval(labels[0],labels[1],labels[2],labels[3])

   # for i in range(15):
   #     labels = sl.DecisionTreeBounded(i)
   #     print("Max Depth = ",i+1)
   #     print(eval(labels[0], labels[1], labels[2], labels[3]))


