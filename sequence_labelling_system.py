from collections import defaultdict

def read_file(filename,type="no_label"):
    f = open(filename,'r')
    if type=="no_label":
        result=[]
        sequence=[]
        for it in f.readlines():
            it=it.strip('\n')
            it=it.split(' ')
            if it[0] != '':
                sequence.append(it[0])
            else:
                result.append(sequence)
                del sequence
                sequence=[]
    elif type=="with_label":
        result=[]
        sequence=[]
        for it in f.readlines():
            it=it.strip('\n')
            it=it.split(' ')

            if it[0] != '':
                sequence.append(it)
            else:
                result.append(sequence)
                del sequence
                sequence=[]
    f.close()
    return result

def write_file(filename,content):
    try:
        import os
        f = open(filename,'w')
        for sequence in content:
            for i in sequence:
                f.write(i[0]+" "+i[1]+os.linesep)
            f.write(os.linesep)
        f.close()
        return "File write OK!!!"
    except:
        return "File write failure!!!Check file path or content format?"

def emission_parameter_calcul_on_train_set(labelled_data,state_set,observation_set):
    #emission(x,y) = count(y->x) / count(y)
    #note that count(y) can be used multiple times, so it is good to have it recorded
    #TODO : implement a better one!

    numerator=defaultdict(dict)
    for y in state_set:
        state_to_observation_count=defaultdict(int)
        numerator[y]=state_to_observation_count

    denominator=defaultdict(int)
    for sentence in labelled_data :
        for item_and_state in sentence :
            x,y=item_and_state
            denominator[y]+=1
            numerator[y][x]+=1


    emission_parameter=defaultdict(dict)
    for y in state_set:
        state_to_observation_count=defaultdict(float)
        emission_parameter[y]=state_to_observation_count

    for state in numerator.keys():
        for observation,value in numerator[state].items():
            emission_parameter[state][observation]=1.0*value/denominator[state]

    return  emission_parameter,numerator,denominator

def emission_parameter_calcul(state,observation,count_set=None,labelled_data=None):
    #emission(x,y) = count(y->x) / count(y)
    #note that count(y) can be used multiple times, so it is good to have it record
    numerator,denominator = count_set
    if count_set!=None:
        if observation in numerator[state].keys():

            e=1.0*(numerator[state][observation]+1)/(denominator[state]+1)
        else:

            numerator[state][observation]=1
            e=1.0*1/(denominator[state]+1)
        return e


    numerator=0
    denominator=0
    for sentence in labelled_data :
        for x,y in sentence :
            if y==state :
                denominator+=1
                if x==observation :
                    numerator+=1
    if numerator==0 :
        numerator=1

    denominator+=1

    e=1.0 *numerator / denominator

    return  e

def transition_parameter_calcul_on_train_set(labelled_data,state_set):

    #initialize the count
    numerator=defaultdict(dict)
    for y in state_set:
        state_to_next_state_count=defaultdict(int)
        numerator[y]=state_to_next_state_count

    denominator=defaultdict(int)

    for sentence in labelled_data :
        current_state="START"
        for item_and_state in sentence :
            x,y=item_and_state
            denominator[current_state]+=1
            numerator[current_state][y]+=1
            current_state=y

        numerator[current_state]["STOP"]+=1
        denominator[current_state]+=1

    transition_parameter=defaultdict(dict)

    for y in state_set:
        state_to_next_state_count=defaultdict(float)
        transition_parameter[y]=state_to_next_state_count

    for state in numerator.keys():
        for next_state,value in numerator[state].items():
            transition_parameter[state][next_state]=1.0*value/denominator[state]

    return  transition_parameter,numerator,denominator



def predict_simple_on_sequences(sequences,emission_parameter,count_set,state_set,labelled_data):
    result=[]

    for sequence in sequences:

        result_seq=[]
        for x in sequence :
            state=''
            em=0
            for y in state_set :
                if y in ["START","STOP"]:
                    continue
                if emission_parameter[y][x]==0:
                    emission_parameter[y][x]= emission_parameter_calcul(y,x,count_set=count_set,labelled_data=labelled_data)
                if emission_parameter[y][x] >= em :
                    state=y
                    em=emission_parameter[y][x]

            temp=[]
            temp.append(x)
            temp.append(state)

            result_seq.append(temp)
            del temp
        result.append(result_seq)
        del result_seq
    return result

def predict_using_Viterbi(sequences,state_set,transition_parameter,emission_parameter,count_set_emission):
    from math import log
    from pprint import pprint as pr
    result = []
    for sequence in sequences:
        result_seq=[]
        pi = defaultdict(dict)
        pi[0]=defaultdict(lambda:-10000000.0)

        track = defaultdict(dict)
        track[0] = defaultdict(lambda:"START")
        for state in state_set:
            pi[0][state] = -10000000 if state!="START" else 0


        for i,word in enumerate(sequence):
            #i is 0~len(sequence-1) add one to it to form an index of 1~len(sequence)
            index=i+1
            pi[index]=defaultdict(lambda:-10000000.0)
            track[index]=defaultdict(lambda:"START")
            for state in state_set:

                if state in ["START","STOP"]:continue

                pi[index][state]=-10000000.0
                for previous_state in state_set:

                    emis=log(emission_parameter[state][word]) if emission_parameter[state][word]!=0 else log(emission_parameter_calcul(state,word,count_set=count_set_emission,labelled_data=None))

                    try:
                        tran=log(transition_parameter[previous_state][state])
                    except:
                        continue

                    if pi[index-1][previous_state]+emis+tran > pi[index][state]:
                        #print("for pi",index,state,"change from",track[index][state],pi[index][state],"to",previous_state,pi[index-1][previous_state]+emis+tran)
                        pi[index][state]=pi[index-1][previous_state]+emis+tran
                        track[index][state]=previous_state

                    #print(pi[index][state],pi[index-1][previous_state]*emis*tran,state,track[index][state])


        pi[len(sequence)+1]["STOP"]=-10000000
        for state in state_set:
            try:
                tran = transition_parameter[state]["STOP"]
            except:
                tran=0
            if pi[len(sequence)][state]+tran > pi[len(sequence)+1]["STOP"]:
                pi[len(sequence)+1]["STOP"]= pi[len(sequence)][state]+tran
                track[len(sequence)+1]["STOP"]=state

        #pr(pi)
        def back_tracking(length,state):
            try:

                if state != "START" :

                    back_tracking(length-1,track[length][state])

                    if state!="STOP" :
                        result_seq.append([sequence[length-1],state])

                    return
                else :
                    return
            except:
                print("index error!on ",length," of state ",state)

        #pr(pi)

        back_tracking(len(sequence)+1,"STOP")
        result.append(result_seq)
        del result_seq

    return result




def evaluate(data_set,standard_set):
    predict=0
    correct_pedict=0
    gold_predicts=0
    flag_for_data=False
    flag_for_standard=False
    correct=False

    for data_seq,standard_seq in zip(data_set,standard_set):

        for data,standard in zip(data_seq,standard_seq):

            if data[0]!=standard[0]:
                print("!error!the format of data are not consistent!")

            data_label=data[1].split("-")
            standard_label=standard[1].split("-")

            #data_label_order : data[1][0]
            #data_label_sentiment : data[1][1]
            #standard_label_order : standard[1][0]
            #standard_label_sentiment :standard[1][1]

            if (data_label[0]=="B" or data_label[0]=="I") and flag_for_data!=True :
                flag_for_data=True
                correct=True
                predict+=1

            if standard_label[0]=="B" and flag_for_standard!=True :
                flag_for_standard=True
                gold_predicts+=1

            if standard_label[0]=="O":
                flag_for_standard=False
            if data_label[0]=="O":
                flag_for_data=False

            if standard_label[0]=="O" and data_label[0]=="O":
                if correct==True:
                    correct_pedict+=1
                    correct=False

            if flag_for_data==True and flag_for_standard==True:
                try:
                    if data_label[1]!=standard_label[1] or data_label[0]!=standard_label[0]:
                        correct=False
                except:
                    if standard_label[0]!=data_label[0]:
                        correct=False

    print("correct predicts ", correct_pedict,"total number of predict we made ",predict,"gold predicts",gold_predicts)
    precision = 1.0*correct_pedict/predict
    recall = 1.0*correct_pedict/gold_predicts
    try:
        F = 2.0 / (1.0/precision+1.0/recall)
    except:
        F = 0

    return (precision,recall,F)

def project_part_2_prepare():
    from pprint import pprint
    labelled_data=read_file('CN/train',type="with_label")
    pprint(labelled_data[:3])

    word_set = set(it[0] for sequence in labelled_data for it in sequence )
    label_set = set(it[1] for sequence in labelled_data for it in sequence )
    #pprint(word_set)
    #pprint(label_set)
    emission_parameter,c1,c2 = emission_parameter_calcul_on_train_set(labelled_data,label_set,word_set)
    pprint(emission_parameter["O"]["高兴"])


    from collections import Counter
    words=[it[0] for sequence in labelled_data for it in sequence]
    word_count=Counter(words)


    print("count of 高兴 is : "+str(word_count["高兴"]))
    print("e(O->高兴) is ")
    print(emission_parameter["O"]["高兴"])

    emission_parameter["O"]["撒达到"] = emission_parameter_calcul("O","撒达到",count_set=(c1,c2),labelled_data=labelled_data)

    print("撒达到 is an new word,which hasn't been in the train set，count of its occurance is ："+str(word_count["撒达到"]))
    print("e(O->撒达到) is : ")
    print(emission_parameter["O"]["撒达到"])

    print("according to the special case handling they all appears to be 1 time, and the latter one is slightly smaller(which is also consistent with the algotithm)")

    return

def learn_and_predict_evaluate_part_2(identifier_name="CN"):
    labelled_data=read_file(identifier_name+'/train',type="with_label")
    word_set = set(it[0] for sequence in labelled_data for it in sequence )
    label_set = set(it[1] for sequence in labelled_data for it in sequence )
    label_set.add("STOP")
    label_set.add("START")
    test_data=read_file(identifier_name+"/dev.in",type="no_label")
    standard_data=read_file(identifier_name+"/dev.out",type="with_label")

    emission_parameter,count_1,count_2 = emission_parameter_calcul_on_train_set(labelled_data,label_set,word_set)
    test_result=predict_simple_on_sequences(test_data,emission_parameter,(count_1,count_2),label_set,labelled_data)
    print(write_file(identifier_name+"/dev.p2.out",test_result))
    p,r,f=evaluate(test_result,standard_data)
    print(p,r,f)
    return

def project_part_2():
    language=["CN","EN","ES","SG"]
    for lan in language:
        learn_and_predict_evaluate_part_2(lan)
    return

def project_part_3():
    language=["CN","EN","ES","SG"]
    for lan in language:
        learn_and_predict_evaluate_part_3(lan)
    return

def learn_and_predict_evaluate_part_3(identifier_name="CN"):

    from pprint import pprint as pr
    labelled_data=read_file(identifier_name+'/train',type="with_label")
    word_set = set(it[0] for sequence in labelled_data for it in sequence )
    label_set = set(it[1] for sequence in labelled_data for it in sequence )
    label_set.add("STOP")
    label_set.add("START")
    test_data=read_file(identifier_name+"/dev.in",type="no_label")
    standard_data=read_file(identifier_name+"/dev.out",type="with_label")


    transition_parameters,_,_ = transition_parameter_calcul_on_train_set(labelled_data,label_set)
    emission_parameters,count_1,count_2 = emission_parameter_calcul_on_train_set(labelled_data,label_set,word_set)

    #pr(transition_parameters)
    #pr(emission_parameters)

    test_result=predict_using_Viterbi(test_data,label_set,transition_parameter=transition_parameters,emission_parameter=emission_parameters,count_set_emission=(count_1,count_2))

    print(write_file(identifier_name+"/dev.p3.out",test_result))
    p,r,f=evaluate(test_result,standard_data)
    print(p,r,f)


#project_part_2_prepare()
#project_part_2()
project_part_3()
#learn_and_predict_evaluate_part_3("CN")
