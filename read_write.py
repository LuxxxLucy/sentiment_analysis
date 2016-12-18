
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
