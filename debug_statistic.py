import pandas as pd
import os

env = os.path.dirname(os.path.abspath(__file__))

def get_statistics():
    # Read the case study data
    df = pd.read_csv(f'D:\DATA\CASBUSI/backups\database3/CaseStudyData.csv')

    total_rows = len(df)

    # Count the number of rows where BI-RADS is 1 and there is a Malignant biopsy
    condition1 = (df['BI-RADS'] == '1') & (df['Biopsy'].str.contains("Malignant"))
    count1 = len(df[condition1])
    percent1 = (count1 / total_rows) * 100

    # Count the number of rows where BI-RADS is 2 and there is a Malignant biopsy
    condition2 = (df['BI-RADS'] == '2') & (df['Biopsy'].str.contains("Malignant"))
    count2 = len(df[condition2])
    percent2 = (count2 / total_rows) * 100

    # Count the number of duplicate rows based on 'Patient_ID' and 'Study_Laterality'
    df['is_duplicate'] = df.duplicated(subset=['Patient_ID', 'Study_Laterality'], keep=False)
    count_duplicates = df['is_duplicate'].sum()
    percent_duplicates = (count_duplicates / total_rows) * 100

    # Count the number of duplicate rows based on 'Accession_Number'
    count_duplicates_acc = df.duplicated(subset=['Accession_Number'], keep=False).sum()
    percent_duplicates_acc = (count_duplicates_acc / total_rows) * 100

    # Count the number of rows where 'Trustworthiness_Score' is 1, 2, or 3
    trust_scores = [1, 2, 3]
    for score in trust_scores:
        count_score = len(df[df['trustworthiness'] == score])
        percent_score = (count_score / total_rows) * 100
        print(f"Number of rows where 'trustworthiness' is {score}: {count_score} ({percent_score:.2f}%)")

    # Count the number of rows where 'Biopsy_Laterality' has different lateralities ('left' and 'right') while 'Study_Laterality' is not 'BILATERAL'
    condition3 = (df['Biopsy_Laterality'].apply(lambda x: 'left' in x and 'right' in x)) & (df['Study_Laterality'] != 'BILATERAL')
    count3 = len(df[condition3])
    percent3 = (count3 / total_rows) * 100

    print(f"Number of rows where BI-RADS is 1 and there is a Malignant biopsy: {count1} ({percent1:.2f}%)")
    print(f"Number of rows where BI-RADS is 2 and there is a Malignant biopsy: {count2} ({percent2:.2f}%)")
    print(f"Number of duplicate rows based on 'Patient_ID' and 'Study_Laterality': {count_duplicates} ({percent_duplicates:.2f}%)")
    print(f"Number of duplicate rows based on 'Accession_Number': {count_duplicates_acc} ({percent_duplicates_acc:.2f}%)")
    print(f"Number of rows where 'Biopsy_Laterality' has different lateralities and 'Study_Laterality' is not 'BILATERAL': {count3} ({percent3:.2f}%)")

if __name__ == '__main__':
    get_statistics()
