# this indicates an error of sequences of impossible
# -*- coding: utf-8 -*-

from collections import defaultdict
import pdb
from pprint import pprint as pr


# MACRO constant definition

VERY_NEGATIVE_NUMBER=-1e30
from math import log


def special_log(i):
    return log(i) if i!=0 else VERY_NEGATIVE_NUMBER

def emission_parameter_calcul_on_train_set(labelled_data,state_set,observation_set):
    #emission(x,y) = count(y->x) / count(y)
    #note that count(y) can be used multiple times, so it is good to have it recorded
    #TODO : implement a better one!

    numerator=defaultdict(dict)
    for y in state_set:
        numerator[y]=defaultdict(int)

    denominator=defaultdict(int)
    for sentence in labelled_data :
        for item_and_state in sentence :
            x,y=item_and_state
            denominator[y]+=1
            numerator[y][x]+=1


    emission_parameter=defaultdict(dict)
    for y in state_set:
        emission_parameter[y]=defaultdict(float)

    for state in numerator.keys():
        for observation,value in numerator[state].items():
            emission_parameter[state][observation]=1.0*value/denominator[state]

    return  emission_parameter,numerator,denominator

def emission_parameter_calcul(state,observation,count_set=None,labelled_data=None):
    #emission(x,y) = count(y->x) / count(y)
    #note that count(y) can be used multiple times, so it is good to have it record
    if len(count_set)==3:
        numerator,denominator,word_set= count_set
    if len(count_set)==2:
        (numerator,denominator,word_set),_= count_set

    if count_set!=None:

        if observation in word_set:
            e=1.0*(numerator[state][observation])/(denominator[state]+1)

        else:
            # that is this word is a new word(never occurs in training set)


            #calcul prior possibility
            # prior_prob={}
            # sum_of_state = sum(denominator.values())
            # for i in state_set:
            #     if i not in ["STOP","START"]:
            #         prior_prob[i]= 1.0 * denominator[i] / sum_of_state
            #         prior_prob[i] = log(prior_prob[i])

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

def modified_emission_parameter_calcul(state,observation,count_set=None,labelled_data=None):
    #emission(x,y) = count(y->x) / count(y)
    #note that count(y) can be used multiple times, so it is good to have it record
    (numerator,denominator,word_set),prior_prob = count_set

    if count_set!=None:

        if observation in word_set:
            e=1.0*(numerator[state][observation])/(denominator[state]+1)

        else:
            # that is this word is a new word(never occurs in training set)


            #calcul prior possibility
            # prior_prob={}
            # sum_of_state = sum(denominator.values())
            # for i in state_set:
            #     if i not in ["STOP","START"]:
            #         prior_prob[i]= 1.0 * denominator[i] / sum_of_state
            #         prior_prob[i] = log(prior_prob[i])

            # numerator[state][observation]=1
            # e=1.0*1/(denominator[state]+1)
            e=prior_prob[state]


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
    from math import log
    result=[]

    numerator,denominator,word_set=count_set
    #calcul prior possibility
    prior_prob={}
    sum_of_state = sum(denominator.values())
    for i in state_set:
        if i not in ["STOP","START"]:
            prior_prob[i]= 1.0 * denominator[i] / sum_of_state


    #calcul emssion
    for sequence in sequences:

        result_seq=[]
        for x in sequence :
            state=''
            em=VERY_NEGATIVE_NUMBER
            for y in state_set :
                if y in ["START","STOP"]:
                    continue
                # if emission_parameter[y][x]==0:
                #     emission_parameter[y][x]= emission_parameter_calcul(y,x,count_set=count_set,labelled_data=labelled_data)
                # temp_em = log(emission_parameter[y][x])+prior_prob[y]
                temp_em = special_log(emission_parameter_calcul(y,x,count_set=count_set,labelled_data=labelled_data))
                if temp_em >= em :
                    state=y
                    em=temp_em

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

    result = []

    numerator,denominator,_=count_set_emission
    #calcul prior possibility
    prior_prob={}
    sum_of_state = sum(denominator.values())
    for i in state_set:
        if i not in ["STOP","START"]:
            prior_prob[i]= 1.0 * denominator[i] / sum_of_state
            prior_prob[i] = special_log(prior_prob[i])

    for sequence in sequences:
        result_seq=[]
        pi = defaultdict(dict)
        pi[0]=defaultdict(lambda:VERY_NEGATIVE_NUMBER)

        track = defaultdict(dict)
        track[0] = defaultdict(lambda:"START")
        for state in state_set:
            pi[0][state] = VERY_NEGATIVE_NUMBER if state!="START" else 0


        for i,word in enumerate(sequence):
            #i is 0~len(sequence-1) add one to it to form an index of 1~len(sequence)
            index=i+1
            pi[index]=defaultdict(lambda:VERY_NEGATIVE_NUMBER)
            track[index]=defaultdict(lambda:"START")
            for state in state_set:

                if state in ["START","STOP"]:continue


                for previous_state in state_set:
                    emis = special_log(emission_parameter_calcul(state,word,count_set=count_set_emission,labelled_data=None))
                    # emis=log(emission_parameter[state][word]) if emission_parameter[state][word]!=0 else prior_prob[state]+log(emission_parameter_calcul(state,word,count_set=count_set_emission,labelled_data=None))

                    #emis=log(emission_parameter[state][word]) if emission_parameter[state][word]!=0 else log(emission_parameter_calcul(state,word,count_set=count_set_emission,labelled_data=None))

                    try:
                        tran=log(transition_parameter[previous_state][state])
                    except:
                        continue

                    if pi[index-1][previous_state]+emis+tran > pi[index][state]:
                        #print("for pi",index,state,"change from",track[index][state],pi[index][state],"to",previous_state,pi[index-1][previous_state]+emis+tran)
                        pi[index][state]=pi[index-1][previous_state]+emis+tran
                        track[index][state]=previous_state






                    #print(pi[index][state],pi[index-1][previous_state]*emis*tran,state,track[index][state])


        pi[len(sequence)+1]["STOP"]=VERY_NEGATIVE_NUMBER
        for state in state_set:
            try:
                tran = log(transition_parameter[state]["STOP"])
            except:
                tran = VERY_NEGATIVE_NUMBER
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
                # this indicates an error of sequences of impossible
                # TODO
                pass


        #pr(pi)

        back_tracking(len(sequence)+1,"STOP")
        result.append(result_seq)
        del result_seq

    return result

def predict_using_Viterbi_top_k(sequences,state_set,transition_parameter,emission_parameter,count_set_emission,number_k=5,emission_parameter_calcul=emission_parameter_calcul):
    from math import log

    result = []
    for k in range(number_k):
        result.append([])
    numerator,denominator,_=count_set_emission
    #calcul prior possibility
    prior_prob={}
    sum_of_state = sum(denominator.values())
    for i in state_set:
        if i not in ["STOP","START"]:
            prior_prob[i]= 1.0 * denominator[i] / sum_of_state

    for sequence in sequences:



        pi_set=list()
        track_set=list()
        for k in range(number_k):
            pi = defaultdict(dict)
            pi[0]=defaultdict(lambda:VERY_NEGATIVE_NUMBER)

            for state in state_set:
                pi[0][state] = VERY_NEGATIVE_NUMBER if state!="START" or k!=0 else 0
            pi_set.append(pi)

            track = defaultdict(dict)
            track[0] = defaultdict(lambda:["START",0])
            track_set.append(track)


        for i,word in enumerate(sequence):
            #i is 0~len(sequence-1) add one to it to form an index of 1~len(sequence)
            index=i+1
            for k in range(number_k):

                pi_set[k][index]=defaultdict(lambda:VERY_NEGATIVE_NUMBER)
                track_set[k][index]=defaultdict(lambda:["START",0])



            for state in state_set:

                if state in ["START","STOP"]:continue
                compare_set=list()
                for k in range(number_k):
                    pi_set[k][index][state]=VERY_NEGATIVE_NUMBER
                    for previous_state in state_set:
                        emis=special_log(emission_parameter_calcul(state,word,count_set=(count_set_emission,prior_prob),labelled_data=None))
                        # emis=log(emission_parameter[state][word]) if emission_parameter[state][word]!=0 else prior_prob[state]+log(emission_parameter_calcul(state,word,count_set=count_set_emission,labelled_data=None))

                        #emis=log(emission_parameter[state][word]) if emission_parameter[state][word]!=0 else log(emission_parameter_calcul(state,word,count_set=count_set_emission,labelled_data=None))

                        try:
                            tran=log(transition_parameter[previous_state][state])
                        except:
                            continue

                        temp_value=pi_set[k][index-1][previous_state]+emis+tran
                        compare_set.append([previous_state,k,temp_value])



                compare_set.sort(key=lambda temp:temp[2],reverse=True)

                for k in range(number_k):
                    pi_set[k][index][state]=compare_set[k][2]
                    track_set[k][index][state]=compare_set[k][0:2]

                del compare_set

        # now compute the final score
        for k in range(number_k):
            pi_set[k][len(sequence)+1]["STOP"]=VERY_NEGATIVE_NUMBER

        compare_set=[]
        for state in state_set:
            try:
                tran = transition_parameter[state]["STOP"]
            except:
                tran = 0
            for k in range(number_k):
                temp_value=pi_set[k][len(sequence)][state]+tran
                compare_set.append([state,k,temp_value])

        compare_set.sort(key=lambda temp:temp[2],reverse=True)
        for k in range(number_k):
            pi_set[k][len(sequence)+1]["STOP"]=compare_set[k][2]
            track_set[k][len(sequence)+1]["STOP"]=compare_set[k][0:2]

        #pr(track_set)

        def back_tracking_top_k(length,state,k):
            try:

                if state!= "START" :

                    back_tracking_top_k(length-1,track_set[k][length][state][0],track_set[k][length][state][1])

                    if state!="STOP" :
                        result_seq.append([sequence[length-1],state])

                    return
                else :
                    return
            except(KeyError):
                # this indicates an error of sequences of impossible
                # TODO
                pass


        #pr(pi)
        for k in range(number_k):

            result_seq=[]
            back_tracking_top_k(len(sequence)+1,"STOP",k)

            result[k].append(result_seq)
            del result_seq

    return result




def find_best_among_k(candidates,number=5):

    def vote_weight(i):
        return int((number-i) * (number-i)/2)

    from collections import Counter
    result=[]
    for i in range(len(candidates[0])):
        try:
            can=[]
            for k in range(number):
                try:
                    vote=vote_weight(k)
                    for _ in range(vote):
                        can.append(candidates[k][i])
                except:
                    continue
            c=Counter(can)
            result.append(c.most_common()[0][0])
        except:
            pdb.set_trace()
    return result

def result_rerank(source_array,number=5):

    result=[]
    for i in range(len(source_array[0])):
        sequences=[item[0] for item in source_array[0][i]]
        tag_sequences=[ [item[1]for item in source_array[index][i]] for index in range(number) ]
        result_seq=find_best_among_k(tag_sequences,number=number)
        result.append(zip(sequences,result_seq))
    return result
