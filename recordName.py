import os

def get_file(root_path,all_files=[],index=0):
    '''
    递归函数，遍历该文档目录和子目录下的所有文件，获取其path
    '''
    files = os.listdir(root_path)
    for file in files:
        if not os.path.isdir(root_path + '/' + file):   # not a dir
            all_files.append(root_path + '/' + file + " " + str(index))
        else:  # is a dir
            get_file((root_path+'/'+file),all_files,index)
            index += 1
    return all_files

def writePathTotxt(base_path,all_files,path,outfile):
    file = open(outfile, "w")
    for files in all_files:
        name = files.split(path + '/')[1]
        file.write(base_path + name + "\n")


if __name__ == '__main__':
    path = './dataset'
    outfile = "train_list.txt"
    writePathTotxt("/home/xmy/PycharmProjects/test/paddle/proj3_recognizeMyself/dataset/",get_file(path), path, outfile)


