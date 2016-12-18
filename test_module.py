from read_write import *
from sequence_labelling_system import *
from pprint import pprint as pr

# this is the preparation phase used to test that this labelling system has basic funcitons
def project_part_2_prepare():
    labelled_data=read_file('CN/train',type="with_label")
    pr(labelled_data[:3])

    word_set = set(it[0] for sequence in labelled_data for it in sequence )
    label_set = set(it[1] for sequence in labelled_data for it in sequence )

    emission_parameter,c1,c2 = emission_parameter_calcul_on_train_set(labelled_data,label_set,word_set)
    pr(emission_parameter["O"]["高兴"])


    from collections import Counter
    words=[it[0] for sequence in labelled_data for it in sequence]
    word_count=Counter(words)


    print("count of 高兴 is : "+str(word_count["高兴"]))
    print("e(O->高兴) is ")
    print(emission_parameter["O"]["高兴"])

    emission_parameter["O"]["撒达到"] = emission_parameter_calcul("O","撒达到",count_set=(c1,c2,word_set),labelled_data=labelled_data)

    print("撒达到 is an new word,which hasn't been in the train set，count of its occurance is ："+str(word_count["撒达到"]))
    print("e(O->撒达到) is : ")
    print(emission_parameter["O"]["撒达到"])

    print("according to the special case handling they all appears to be 1 time, and the latter one is slightly smaller(which is also consistent with the algotithm)")

    return

# run the labelling system for part2 in the dataset and use it to predict
def learn_and_predict_evaluate_part_2(identifier_name="CN"):
    labelled_data=read_file(identifier_name+'/train',type="with_label")
    word_set = set(it[0] for sequence in labelled_data for it in sequence )
    label_set = set(it[1] for sequence in labelled_data for it in sequence )
    label_set.add("STOP")
    label_set.add("START")
    test_data=read_file(identifier_name+"/dev.in",type="no_label")
    standard_data=read_file(identifier_name+"/dev.out",type="with_label")

    emission_parameter,count_1,count_2 = emission_parameter_calcul_on_train_set(labelled_data,label_set,word_set)
    test_result=predict_simple_on_sequences(test_data,emission_parameter,(count_1,count_2,word_set),label_set,labelled_data)
    print(write_file(identifier_name+"/dev.p2.out",test_result))
    return
# loop on 4 datasets
def project_part_2():
    language=["CN","EN","ES","SG"]
    for lan in language:
        learn_and_predict_evaluate_part_2(lan)
    return

# run the labelling system for part3 in the dataset and use it to predict
def learn_and_predict_evaluate_part_3(identifier_name="CN"):


    # train data set
    labelled_data=read_file(identifier_name+'/train',type="with_label")
    word_set = set(it[0] for sequence in labelled_data for it in sequence )
    label_set = set(it[1] for sequence in labelled_data for it in sequence )
    label_set.add("STOP")
    label_set.add("START")

    # test data set
    test_data=read_file(identifier_name+"/dev.in",type="no_label")
    # standard data set
    standard_data=read_file(identifier_name+"/dev.out",type="with_label")


    transition_parameters,_,_ = transition_parameter_calcul_on_train_set(labelled_data,label_set)
    emission_parameters,count_1,count_2 = emission_parameter_calcul_on_train_set(labelled_data,label_set,word_set)

    #pr(transition_parameters)
    #pr(emission_parameters)

    test_result=predict_using_Viterbi(test_data,label_set,transition_parameter=transition_parameters,emission_parameter=emission_parameters,count_set_emission=(count_1,count_2,word_set))

    print(write_file(identifier_name+"/dev.p3.out",test_result))

# loop on 4 datasets
def project_part_3():
    language=["CN","EN","ES","SG"]
    for lan in language:
        learn_and_predict_evaluate_part_3(lan)
    return




# run the labelling system for part4 in the dataset and use it to predict
def learn_and_predict_evaluate_part_4(identifier_name="CN"):


    # train data set
    labelled_data=read_file(identifier_name+'/train',type="with_label")
    word_set = set(it[0] for sequence in labelled_data for it in sequence )
    label_set = set(it[1] for sequence in labelled_data for it in sequence )
    label_set.add("STOP")
    label_set.add("START")

    # test data set
    test_data=read_file(identifier_name+"/dev.in",type="no_label")
    # standard data set
    standard_data=read_file(identifier_name+"/dev.out",type="with_label")


    transition_parameters,_,_ = transition_parameter_calcul_on_train_set(labelled_data,label_set)
    emission_parameters,count_1,count_2 = emission_parameter_calcul_on_train_set(labelled_data,label_set,word_set)

    #pr(transition_parameters)
    #pr(emission_parameters)

    test_result=predict_using_Viterbi_top_k(test_data,label_set,transition_parameter=transition_parameters,emission_parameter=emission_parameters,count_set_emission=(count_1,count_2,word_set),number_k=5)

    print(write_file(identifier_name+"/dev.p4.out",test_result[4]))
# loop on 4 datasets
def project_part_4():
    language=["CN","EN","ES","SG"]
    for lan in language:
        learn_and_predict_evaluate_part_4(lan)
    return


# run the labelling system for part5 in the dataset and use it to predict
def learn_and_predict_evaluate_part_5(identifier_name="CN",number=5):


    # train data set
    labelled_data=read_file(identifier_name+'/train',type="with_label")
    word_set = set(it[0] for sequence in labelled_data for it in sequence )
    label_set = set(it[1] for sequence in labelled_data for it in sequence )
    label_set.add("STOP")
    label_set.add("START")

    # test data set
    test_data=read_file(identifier_name+"/dev.in",type="no_label")
    # standard data set
    standard_data=read_file(identifier_name+"/dev.out",type="with_label")


    transition_parameters,_,_ = transition_parameter_calcul_on_train_set(labelled_data,label_set)
    emission_parameters,count_1,count_2 = emission_parameter_calcul_on_train_set(labelled_data,label_set,word_set)

    #pr(transition_parameters)
    #pr(emission_parameters)

    test_result=predict_using_Viterbi_top_k(test_data,label_set,transition_parameter=transition_parameters,emission_parameter=emission_parameters,count_set_emission=(count_1,count_2,word_set),number_k=number,emission_parameter_calcul=modified_emission_parameter_calcul)

    # print(write_file(identifier_name+"/dev.p5.out",test_result[0]))

    print(write_file(identifier_name+"/dev.p5.out",result_rerank(test_result,number=number)))

    if identifier_name in ["EN","ES"]:
        real_test_data=read_file(identifier_name+"/test.in",type="no_label")

        real_test_result=predict_using_Viterbi_top_k(real_test_data,label_set,transition_parameter=transition_parameters,emission_parameter=emission_parameters,count_set_emission=(count_1,count_2,word_set),number_k=number,emission_parameter_calcul=modified_emission_parameter_calcul)

        print(write_file(identifier_name+"/test.out",result_rerank(real_test_result,number=number)))

# loop on 4 datasets
def project_part_5(number=5):
    language=["CN","EN","ES","SG"]
    for lan in language:
        learn_and_predict_evaluate_part_5(lan,number)
    return
