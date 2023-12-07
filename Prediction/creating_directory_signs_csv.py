if __name__ == '__main__':
    import os
    import pandas as pd

    data_url = r'E:\DataSets\Traffic Sign\Traffic_Sign - 205 classes (GTSRB+162 custom classes)\Classes'

    # Directory describing signs and their classes
    class_list = []
    for root, dirs, full_name_group in os.walk(data_url):
        for subdir in dirs:
            if full_name_group:
                group_sign = root[87:]    # 86 = root.rfind('\\')
                class_sign = subdir[:subdir.find(' ')]
                name_sign = subdir[subdir.find(' ') + 1:]
                full_name = full_name_group[0]
                sign_info = [group_sign, full_name_group[0][:-4], class_sign, name_sign]
                class_list.append(sign_info)

    df_sign_info = pd.DataFrame(class_list, columns=['sign_group_priority', 'sign_group_name', 'sign_class', 'sign_name'])
    df_sign_info.to_csv('Directory_signs.csv', sep=',', header=True, index=None)

    print(df_sign_info.head())

