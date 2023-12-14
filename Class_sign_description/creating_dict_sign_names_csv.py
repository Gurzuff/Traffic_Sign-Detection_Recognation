import os
import pandas as pd

def main():
    # Directory of full sign names for each classes
    class_list = []
    data_url = 'E:\DataSets\Traffic Sign - Detection&Recognation\Traffic_Sign - 200 classes\Classes'
    for root, dirs, full_name_group in os.walk(data_url):
        for subdir in dirs:
            if full_name_group:
                group_sign = root[len(data_url)+1:]
                class_sign = subdir[:subdir.find(' ')]
                name_sign = subdir[subdir.find(' ') + 1:]
                full_name = full_name_group[0]
                sign_info = [group_sign, full_name_group[0][:-4], class_sign, name_sign]
                class_list.append(sign_info)

    df_sign_info = pd.DataFrame(class_list, columns=['sign_group_priority', 'sign_group_name', 'sign_class', 'sign_name'])
    df_sign_info = df_sign_info[df_sign_info.sign_class != '-']
    df_sign_info.to_csv('full_name_signs.csv', sep=',', header=True, index=None)
    print(df_sign_info.head())

if __name__ == '__main__':
    main()
