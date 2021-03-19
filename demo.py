import os
import math
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import stdiomask
from scipy import stats as ss
from mysql import connector as mysql
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as gs
from scipy.signal import savgol_filter
plt.style.use('bmh')  # matplotlib graph style
sns.set_style('dark')  # seaborn graph style
warnings.filterwarnings('ignore')  # to ignore messages from seaborn graphs


""" MAKE SURE THE DATABASE DETAILS ARE CORRECT FOR YOU """
course_len = 4  # how many years is the course
path = "C:/ProgramData/MySQL/MySQL Server 8.0/Data/btechcse"  # path to database and to store prediction models
db = mysql.connect(host='localhost', user='Ashwin', password='3431', database="btechcse")  # initializing connection to database
cursor = db.cursor(buffered=True)  # cursor


def grades(test_type, test_amount, max_mark, weightage, pass_percent, final_test_name, n=1000, graphs=False):  # grades generator
    """Func that generates train/test data for the classifier/regressor and returns the trained classifier/regressor"""
    df = pd.DataFrame(index=range(1, n+1))  # making the dataframe and generating dummy marks
    df.index.name = 'Student'
    print("\nGenerating mock data\n")
    for x in range(len(test_type)):
        m = max_mark[x]  # storing max marks for each type of test
        if test_amount[x] > 1:  # generating random marks in marking range with a gaussian weight distribution to each mark
            # mew = 65% of max mark and sigma = a third of max mark
            for y in range(1, test_amount[x] + 1):
                df[f"{test_type[x]} {y}"] = [round(x) for x in (ss.truncnorm.rvs(((0 - int(m * 0.65)) / (m//3)),
                                                                                 ((m - int(m * 0.65)) / (m//3)),
                                                                                 loc=round(m * 0.65, 0), scale=(m//3), size=n))]
        else:
            for y in range(1, test_amount[x] + 1):
                df[f"{test_type[x]}"] = [round(x) for x in (ss.truncnorm.rvs(((0 - int(m * 0.65)) / (m//3)),
                                                                             ((m - int(m * 0.65)) / (m//3)),
                                                                             loc=round(m * 0.65, 0), scale=(m//3), size=n))]

    # calculating total grade weight weightage
    df['Total %'] = [0] * len(df)
    for x in range(len(test_type)):
        df['Total %'] += round((df.filter(regex=test_type[x]).sum(axis=1) / (test_amount[x] * max_mark[x])) * weightage[x], 2)

    # determining pass/fail
    df['Pass/Fail'] = ["Pass" if x >= pass_percent else "Fail" for x in df['Total %']]
    print("Generated mock data!\n")

    print(f"\nStudents passed -> {len(df[df['Pass/Fail'] == 'Pass'])}\
    \nStudents Failed -> {len(df[df['Pass/Fail'] == 'Fail'])}\n")

    X = df.drop(['Pass/Fail', 'Total %', final_test_name], axis=1)
    y = df[['Pass/Fail', 'Total %']].copy()
    y['Pass/Fail'] = LabelEncoder().fit_transform(y['Pass/Fail'])

    print("Creating and fitting models\n")
    # making train and test data for models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y['Pass/Fail'])

    # passing probability predictor
    passfail = make_pipeline(StandardScaler(),
                             LogisticRegressionCV(Cs=np.arange(0.1, 1.1, 0.1),
                                                  cv=RepeatedStratifiedKFold(n_splits=10, random_state=7),
                                                  max_iter=1000, n_jobs=-1, refit=True,
                                                  random_state=7,
                                                  class_weight='balanced')).fit(X_test, y_test['Pass/Fail'])

    if graphs == True:
        for x in range(len(test_type)):
            if test_amount[x] > 1:  # plotting grade distribution for tests with more than one evaluation of that type
                if test_amount[x] % 2 != 0:
                    y = test_amount[x]+1
                else:
                    y = test_amount[x]

                fig = plt.figure(figsize=(10, 7), constrained_layout=True)
                grid = gs(nrows=int(y/2), ncols=2, figure=fig)
                for y in range(test_amount[x]):
                    ax = fig.add_subplot(grid[y])
                    sns.distplot(df.filter(regex=test_type[x]).iloc[:, y], fit=ss.norm, ax=ax, norm_hist=True, color='red',
                                 hist_kws=dict(edgecolor='black', align='right', color='red'),
                                 bins=max_mark[x]//2)
                    plt.xticks(size=10)
                    plt.yticks(size=12)
                    plt.xlabel('Marks', fontdict={'size': 13})
                    plt.ylabel('Density', fontdict={'size': 13})
                    plt.title(f"{test_type[x]} {y+1}", fontsize=14)
                    plt.tight_layout()
                fig.suptitle(f"Grade for {test_type[x]}", fontsize=15)
                grid.tight_layout(fig)
                if graphs == True:
                    plt.show()
            else:  # plotting grade distribution for singular evaluation test
                plt.figure(figsize=(8, 5))
                sns.distplot(df[test_type[x]], fit=ss.norm, norm_hist=True, color='red',
                             hist_kws=dict(edgecolor='black', align='right', color='red'),
                             bins=max_mark[x]//2)
                plt.title(f"Grade for {test_type[x]}", fontsize=14)
                plt.xlabel('Marks', fontdict={'size': 13})
                plt.xticks(size=12)
                plt.yticks(size=12)
                plt.ylabel('Density', fontdict={'size': 13})
                plt.tight_layout()
                plt.show()

        fig, ax = plt.subplots()
        plot_confusion_matrix(passfail, X_test, y_test['Pass/Fail'], labels=[0, 1],
                              display_labels=['Fail', 'Pass'], cmap='afmhot', ax=ax)
        plt.rcParams.update({'font.size': 18})
        ax.set_title('Confusion Matrix')
        plt.show()

    # final overall grade predictor
    overallgrade = make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1)).fit(X_test, y_test['Total %'])
    print("Models created")

    print("LogisticRegressionCV classifer:-")
    print(f"Accuracy -> {round(passfail.score(X_test, y_test['Pass/Fail'])*100, 2)}%")
    print(f"f1_score -> {round(f1_score(y_test['Pass/Fail'], passfail.predict(X_test))*100, 2)}%\n")

    print("LinearRegression regressor:-")
    print(f"Accuracy -> {round(overallgrade.score(X_test, y_test['Total %'])*100, 2)}%")
    print("Models created")

    return passfail, overallgrade


def rolling_predict(marks, subject, record):  # to present rolling predictions based on a student's marks
    all_marks = marks  # saved to calculate rolling actual grade (calculated with subject structure details)
    marks = np.array(marks[:-1]).reshape(1, -1)  # prepping data to be used for predictions
    # loading subject prediction models
    passfail = pickle.load(open(f"{path}/{subject}_passfail", 'rb'))
    overallgrade = pickle.load(open(f"{path}/{subject}_overallgrade", 'rb'))

    dummy = [0] * len(marks[0])  # making dummy list to cummulatively add each test score
    pass_probabs = []  # to store each probability as each test score gets entered
    for x in range(len(marks[0])):
        dummy[x] = marks[0][x]
        pass_probabs.append(passfail.predict_proba(np.array([dummy]))[0][1] * 100)

    # interpolating results to give a smoother graph
    pass_probabs_l = len(pass_probabs)
    if pass_probabs_l % 2 == 0:
        pass_probabs_l -= 1
    pass_probabs = savgol_filter(pass_probabs, pass_probabs_l, 4)

    limit1 = math.ceil(max([abs(x - 50) for x in pass_probabs]))  # limits determiend to scale the pass/fail graph better

    pf = None  # predicting if in the end the model predicts student failing or passing
    if passfail.predict(marks) == 0:
        pf = 'Fail'
    else:
        pf = 'Pass'

    dummy = [0] * len(marks[0])  # calculating rolling overall mark percentage
    total_percent = []
    for x in range(len(marks[0])):
        dummy[x] = marks[0][x]
        total_percent.append(round(overallgrade.predict(np.array(dummy).reshape(1, -1))[0]*100, 3))

    # interpolating results to give a smoother graph
    total_percent_l = len(total_percent)
    if total_percent_l % 2 == 0:
        total_percent_l -= 1

    total_percent = savgol_filter(total_percent, total_percent_l, 4)

    limit2 = math.ceil(max([abs(x-60) for x in total_percent]))  # limits determined to scale the overall grade graph better

    # calculating grade
    grade_p = overallgrade.predict(marks)[0]*100
    grade = None
    if grade_p >= 90:
        grade = "A+"  # milou
    elif grade_p >= 80:
        grade = "A"
    elif grade_p >= 70:
        grade = "B"
    elif grade_p >= 60:
        grade = "C"
    elif grade_p >= 50:
        grade = "D"
    else:
        grade = "F"

    # calculating the rolling actual grade depending on subject structure
    dummy = [0] * len(all_marks)
    actual_grades = []
    for x in range(len(all_marks)):
        dummy[x] = all_marks[x]
        cursor.execute(f"SELECT Amount FROM {subject}_details")
        amounts = [int(x[0]) for x in cursor.fetchall()]

        cursor.execute(f"SELECT Weightage FROM {subject}_details")
        weightages = [float(x[0]) for x in cursor.fetchall()]

        cursor.execute(f"SELECT Max_mark FROM {subject}_details")
        max_marks = [int(x[0]) for x in cursor.fetchall()]

        # calculating rolling actual grade of student
        mults = []
        c = 0
        for x in range(len(amounts)):
            for y in range(amounts[x]):
                mults.append((dummy[c] / max_marks[x]) * (weightages[x] / amounts[x]))
                c += 1
        mults = [np.round(x*100, 2) for x in mults]
        p_grade = np.sum(mults)
        actual_grades.append(p_grade)

    # interpolating results to give a smoother graph
    actual_grades_l = len(actual_grades)
    if actual_grades_l % 2 == 0:
        actual_grades_l -= 1

    actual_grades = savgol_filter(actual_grades, actual_grades_l, 4)

    limit3 = math.ceil(max([abs(x-60) for x in actual_grades]))  # limits determined to scale the overall grade graph better

    actual_grade = None
    if actual_grades[-1] >= 90:
        actual_grade = "A+"  # milou
    elif actual_grades[-1] >= 80:
        actual_grade = "A"
    elif actual_grades[-1] >= 70:
        actual_grade = "B"
    elif actual_grades[-1] >= 60:
        actual_grade = "C"
    elif actual_grades[-1] >= 50:
        actual_grade = "D"
    else:
        actual_grade = "F"

    # getting the name of tests used for predictions
    cursor.execute(f"DESCRIBE {record}_{subject}")
    tests = [x[0] for x in cursor.fetchall()][1:-3]

    fig = plt.figure(f"Grade prediction and calculation for {subject}", figsize=(10, 5))
    grid = gs(nrows=2, ncols=2, figure=fig)
    plt.suptitle(
        f"Chance of Passing and Predicted Total Grade for {subject}\nTake the predictions with a grain of salt", fontsize=12)
    ax1 = fig.add_subplot(grid[0, 0])
    plt.title(f"Probability of passing the subject after each test taken\nPredicted Pass or Fail? -> {pf}\
    \nChance of passing subject -> {passfail.predict_proba(marks)[0][1] * 100:.2f}%", fontsize=11)
    plt.axhline(50, color='r', label="Threshold", linestyle='--')
    plt.plot(tests[:-1], pass_probabs, c='blue', lw=1, label='Predicted passing chance')
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8, rotation=45)
    plt.ylabel('Probability (%)', fontsize=9)
    plt.ylim(ymin=50-limit1, ymax=50+limit1)
    plt.margins(0.02, 0.02)
    plt.legend(loc='best', fontsize=7)
    plt.tight_layout()

    ax2 = fig.add_subplot(grid[0, 1])
    plt.title(f"Predicting Overall grade after each test\nPredicted Overall Subject Grade -> {grade_p:.2f}%\nPredicted Grade -> {grade}", fontsize=11)
    plt.axhline(60, color='r', label='Passing Threshold', linestyle='--')
    plt.plot(tests[:-1], total_percent, c='blue', lw=1, label='Predicted Overall Grade')
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8, rotation=45)
    plt.ylabel('Final Grade', fontsize=9)
    plt.margins(x=0.01, y=0.01)
    plt.ylim(ymin=60-limit2, ymax=60+limit2)
    plt.legend(loc='best', fontsize=7)
    plt.tight_layout()

    ax3 = fig.add_subplot(grid[1, :])
    plt.title(f"Actual Rolling Total Mark (out of 100) calculated for {subject} -> {actual_grades[-1]} ({actual_grade})", fontsize=11)
    plt.axhline(60, color='r', label='Passing Threshold', linestyle='--')
    plt.plot(tests, actual_grades, c='blue', lw=1, label='Caculated Grade (After each test)')
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8, rotation=45)
    plt.ylabel('Grade (out of 100)', fontsize=9)
    plt.margins(x=0.01, y=0.01)
    plt.legend(loc='best', fontsize=7)
    plt.ylim(ymin=60-limit3, ymax=60+limit3)
    plt.tight_layout()
    plt.show()
    print("\nDetailed predictions for marks shown\n")


def student_session(record, user):
    while True:
        print("Student menu")
        print("\n1. View grades")
        print("2. View account details")
        print("3. Change account details")
        print("4. Logout")
        choice = input("Choice : ")
        if choice == '1':  # view grades of a subject
            print("\nView grades\b")
            cursor.execute("SELECT id, name FROM subjects")
            subjects = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
            [print(f"{x[0]}. {x[1]}") for x in subjects]  # choosing subject
            print(f"{len(subjects)+1}. Go back")
            choice = input("Choice : ")
            if int(choice) in [x[0] for x in subjects]:
                subject = subjects[int(choice)-1][1]
                cursor.execute(f"SELECT id FROM {record} WHERE username = '{user}'")
                id = cursor.fetchall()[0][0]
                cursor.execute(f"DESCRIBE {record}_{subject}")
                tests = [x[0] for x in cursor.fetchall() if x[0] != 'student_id']
                print("")
                for test in tests:  # printing marks on a subject for each test
                    cursor.execute(f"SELECT {test} FROM {record}_{subject} WHERE student_id = {id}")
                    print(f"{test} -> {cursor.fetchall()[0][0]}")

                marks = []  # storing marks of student
                print("\n1. View more info")  # to show prediction result graphs
                print("2. Go back")
                choice = input("Choice : ")
                if choice == '1':
                    cursor.execute(f"DESCRIBE {record}_{subject}")
                    tests = [x[0] for x in cursor.fetchall()][1:-3]
                    for test in tests:  # iterating and getting marks for tests except for the final test
                        cursor.execute(f"SELECT {test} FROM {record}_{subject} WHERE student_id = {id}")
                        marks.append(cursor.fetchall()[0][0])
                    rolling_predict(marks, subject, record)
                    print(f"\nMarks displayed for {user} in {subject}\n")
                else:
                    print("\nGoing back\n")
            else:
                print("\nGoing back\n")
                continue

        elif choice == '2':  # showing student account details
            print("\nShowing account details\n")
            cursor.execute(f"DESCRIBE {record}")
            cols = [x[0] for x in cursor.fetchall()]
            for col in cols:
                cursor.execute(f"SELECT {col} FROM {record} WHERE username = '{user}'")
                print(f"{col} -> {cursor.fetchall()[0][0]}")
            print(f"\nAccount details for username {user} shown\n")

        elif choice == '3':  # changing detail in student account
            print(f"\nChanging account details for {user}\n")
            cursor.execute(f"DESCRIBE {record}")
            cols = list(enumerate([x[0] for x in cursor.fetchall() if x[0] not in ["id", "email", "username"]], start=1))
            [print(f"{x[0]}. {x[1]}") for x in cols]
            print(f"{len(cols)+1} Go back")
            choice = input("Choice : ")
            if int(choice) in [x[0] for x in cols]:
                col = cols[int(choice)-1][1]
                cursor.execute(f"SELECT {col} FROM {record} WHERE username = '{user}'")
                old_detail = cursor.fetchall()[0][0]
                new_detail = input(f"\nOld {col} -> {old_detail}\nEnter new {col} : ")
                cursor.execute(f"UPDATE {record} SET {col} = '{new_detail}' WHERE username = '{user}'")
                db.commit()
                print(f"{user} {col} changed from {old_detail} to {new_detail}\n")
            else:
                print("Going back\n")
                continue

        elif choice == '4':
            print("Logging out\n")
            break

        else:
            print("Enter valid choice\n")


def student_auth():
    cursor.execute("SELECT start_year FROM students_batch")
    records = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
    print("\nWhich record do you belong to?")
    [print(f"{x[0]}. {x[1]}") for x in records]  # choosing which student record they belong to
    choice = input("Choice : ")
    if int(choice) in [x[0] for x in records]:
        record = f"students_{records[int(choice)-1][1]}"
        print("\nStudent Login\n")
        user = input("Username : ")  # checking login details
        passw = stdiomask.getpass(prompt='Password : ')
        cursor.execute(f"SELECT username FROM {record}")
        if user in [x[0] for x in cursor.fetchall()]:
            cursor.execute(f"SELECT password FROM {record} WHERE username = '{user}'")
            if passw == cursor.fetchall()[0][0]:
                student_session(record, user)
            else:
                print("\nIncorrect details, going back...\n")
        else:
            print("\nIncorrect details, going back...\n")
    else:
        print("Incorrect choice, going back...\n")


def teacher_session(teacher_id):
    cursor.execute(f"SELECT id FROM subjects WHERE teacher_id = {teacher_id}")
    subjects = list(enumerate([x[0] for x in cursor.fetchall()], start=1))  # keeping a track of this particular teacher's subjects
    while True:
        print("\nWelcome to the teacher menu\n")
        print("1. Review overall marks for a subject")
        print("2. Update a test score")
        print("3. Update a student's score")
        print("4. Edit account info")
        print("5. View subject details")
        print("6. Logout")
        choice = input("Choice : ")
        if choice == '1':  # review marks for a subjects
            print("For which subject?\n")
            [print(f"{x[0]}. {x[1]}") for x in subjects]
            print(f"{len(subjects)+1}. Go back")
            choice = input("Choice : ")
            # choosing for which subject to edit marks
            if int(choice) in [x[0] for x in subjects]:
                print("")
                subject = subjects[int(choice)-1][1]
                cursor.execute("SHOW TABLES WHERE tables_in_btechcse LIKE 'students_%' AND tables_in_btechcse NOT LIKE 'students_batch'")
                # choosing for which batch to edit marks for
                records = list(enumerate([x[0] for x in cursor.fetchall() if len(x[0]) < 15], start=1))
                print("Choose student batch\n")
                [print(f"{x[0]}. {x[1]}") for x in records]
                print(f"{len(records)+1}. Go back")
                choice = input("Choice : ")
                if int(choice) in [x[0] for x in records]:
                    record = records[int(choice)-1][1]
                    table = f"{record}_{subject}".lower()
                    print(table)
                    print(pd.read_sql(f"SELECT * FROM {table}", db, index_col='student_id'), '\n')
                    print(f"\n{record}_{subject} values shown\n")
                else:
                    print("Incorrect choice, going back...\n")
                    continue

            else:
                print("Incorrect choice, going back...\n")
                continue

        elif choice == '2':  # updating scores of students for a test
            print("For which subject?\n")
            [print(f"{x[0]}. {x[1]}") for x in subjects]
            print(f"{len(subjects)+1}. Go back")
            choice = input("Choice : ")
            # choosing for which subject to edit marks
            if int(choice) in [x[0] for x in subjects]:
                print("")
                subject = subjects[int(choice)-1][1]
                cursor.execute("SHOW TABLES WHERE tables_in_btechcse LIKE 'students_%' AND tables_in_btechcse NOT LIKE 'students_batch'")
                # choosing for which batch to edit marks for
                records = list(enumerate([x[0] for x in cursor.fetchall() if len(x[0]) < 15], start=1))
                print("Choose student batch\n")
                [print(f"{x[0]}. {x[1]}") for x in records]
                print(f"{len(records)+1}. Go back")
                choice = input("Choice : ")
                if int(choice) in [x[0] for x in records]:
                    record = records[int(choice)-1][1]
                    # choosing which test to edit marks for
                    cursor.execute(f"DESCRIBE {record}_{subject}")
                    tests = list(enumerate([x[0] for x in cursor.fetchall() if x[0] not in ["student_id", "PASS_CHANCE", "PREDICTED_GRADE"]], start=1))
                    print("\nWhich test?\n")
                    [print(f"{x[0]}. {x[1]}") for x in tests]
                    print(f"{len(tests)}. Go back")
                    choice = input("Choice : ")

                    # loading subject prediction models
                    passfail = pickle.load(open(f"{path}/{subject}_passfail", 'rb'))
                    overallgrade = pickle.load(open(f"{path}/{subject}_overallgrade", 'rb'))

                    if int(choice) in [x[0] for x in tests]:
                        test = tests[int(choice)-1][1]
                        print(f"\nEditing marks of {test} for {record} for subject {subject}\n")
                        # getting the list of student_ids in this mark sheet
                        cursor.execute(f"SELECT student_id FROM {record}_{subject}")
                        ids = [x[0] for x in cursor.fetchall()]
                        print("\nEnter marks\n")
                        for id in ids:
                            mark = int(input(f"Student {id} -> "))
                            # update mark of each student
                            cursor.execute(f"UPDATE {record}_{subject} SET {test} = {mark} WHERE student_id = {id}")
                            db.commit()

                        for id in ids:
                            cursor.execute(f"DESCRIBE {record}_{subject}")
                            # get the name of tests for prediction
                            tests = [x[0] for x in cursor.fetchall()][1:-4]
                            mark = []
                            for test in tests:
                                cursor.execute(f"SELECT {test} FROM {record}_{subject} WHERE student_id = {id}")
                                mark.append(cursor.fetchall()[0][0])

                            # converting to a 2d array so scikit-learn models can use them for predictions
                            mark = np.array(mark).reshape(1, -1)
                            predict_passing = round(passfail.predict_proba(mark)[0][1]*100, 2)  # proabability of passing the subject
                            predict_grade = round(overallgrade.predict(mark)[0]*100, 2)  # predicted grade of overall subject after finals
                            predict_lettergrade = None  # giving letter based grade based on predicted overall percentage after finals
                            if predict_grade >= 90:
                                predict_lettergrade = 'A+'
                            elif predict_grade >= 80:
                                predict_lettergrade = 'A'
                            elif predict_grade >= 70:
                                predict_lettergrade = 'B'
                            elif predict_grade >= 60:
                                predict_lettergrade = 'C'
                            elif predict_grade >= 50:
                                predict_lettergrade = 'D'
                            else:
                                predict_lettergrade = 'F'

                            # new predict grade with mark and letter grade
                            predict_grade = f"{predict_grade} ({predict_lettergrade})"

                            cursor.execute(f"UPDATE {record}_{subject} SET PASS_CHANCE = {predict_passing} WHERE student_id = {id}")
                            db.commit()

                            cursor.execute(f"UPDATE {record}_{subject} SET PREDICTED_GRADE = '{predict_grade}' WHERE student_id = {id}")
                            db.commit()

                            # gathering tests to calculate actual grade using subject details and actual current marks
                            cursor.execute(f"SELECT * FROM {record}_{subject} WHERE student_id = {id}")
                            marks = [int(x) for x in [x for x in cursor.fetchall()[0]][1:-3]]

                            cursor.execute(f"SELECT Amount FROM {subject}_details")
                            amounts = [int(x[0]) for x in cursor.fetchall()]

                            cursor.execute(f"SELECT Weightage FROM {subject}_details")
                            weightages = [float(x[0]) for x in cursor.fetchall()]

                            cursor.execute(f"SELECT Max_mark FROM {subject}_details")
                            max_marks = [int(x[0]) for x in cursor.fetchall()]

                            # calculating rolling actual grade of student
                            mults = []
                            c = 0
                            for x in range(len(amounts)):
                                for y in range(amounts[x]):
                                    mults.append((marks[c] / max_marks[x]) * (weightages[x] / amounts[x]))
                                    c += 1
                            mults = [np.round(x*100, 2) for x in mults]
                            p_grade = np.sum(mults)
                            l_grade = None
                            if p_grade >= 90:
                                l_grade = 'A+'
                            elif p_grade >= 80:
                                l_grade = 'A'
                            elif p_grade >= 70:
                                l_grade = 'B'
                            elif p_grade >= 60:
                                p_grade = 'C'
                            elif p_grade >= 50:
                                l_grade = 'D'
                            else:
                                l_grade = 'F'

                            grade = f"{p_grade} ({l_grade})"
                            cursor.execute(f"UPDATE {record}_{subject} SET GRADE = '{grade}' WHERE student_id = {id}")
                            db.commit()

                        print("Marks updated\n")

                    else:
                        print("Incorreect choice, going back...\n")
                        continue

                else:
                    print("Incorrect choice, going back...\n")
                    continue

            elif int(choice) == len(subjects)+1:
                print("Incorrect choice, going back...\n")
                continue

        elif choice == '3':  # updating grade of one student in a particular subject
            print("For which subject?\n")
            [print(f"{x[0]}. {x[1]}") for x in subjects]
            print(f"{len(subjects)+1}. Go back")
            choice = input("Choice : ")
            # choosing for which subject to edit marks
            if int(choice) in [x[0] for x in subjects]:
                print("")
                subject = subjects[int(choice)-1][1]
                cursor.execute("SHOW TABLES WHERE tables_in_btechcse LIKE 'students_%' AND tables_in_btechcse NOT LIKE 'students_batch'")
                # choosing for which batch to edit marks for
                records = list(enumerate([x[0] for x in cursor.fetchall() if len(x[0]) < 15], start=1))
                print("Choose student batch\n")
                [print(f"{x[0]}. {x[1]}") for x in records]
                print(f"{len(records)+1}. Go back")
                choice = input("Choice : ")
                if int(choice) in [x[0] for x in records]:
                    record = records[int(choice)-1][1]
                    print("Choose student id to edit marks for : \n")
                    cursor.execute(f"SELECT student_id FROM {record}_{subject}")
                    ids = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
                    [print(f"{x[0]}. {x[1]}") for x in ids]
                    print(f"{len(ids)+1}. Go back")
                    choice = input("Choice : ")
                    if int(choice) in [x[0] for x in ids]:
                        student_id = ids[int(choice)-1][1]
                        cursor.execute(f"DESCRIBE {record}_{subject}")
                        tests = list(enumerate([x[0] for x in cursor.fetchall() if x[0] not in ["student_id", "PASS_CHANCE", "PREDICTED_GRADE"]], start=1))
                        print("\nWhich test?\n")
                        [print(f"{x[0]}. {x[1]}") for x in tests]
                        choice = input("Choice : ")
                        if int(choice) in [x[0] for x in tests]:
                            test = tests[int(choice)-1][1]
                            mark = input(f"\nEnter new mark for student_id {student_id} in {test} for {subject}\n")
                            cursor.execute(f"UPDATE {record}_{subject} SET {test} = {int(mark)} WHERE student_id = {student_id}")
                            db.commit()

                            # loading subject prediction models
                            passfail = pickle.load(open(f"{path}/{subject}_passfail", 'rb'))
                            overallgrade = pickle.load(open(f"{path}/{subject}_overallgrade", 'rb'))

                            cursor.execute(f"DESCRIBE {record}_{subject}")
                            # get the name of tests for prediction
                            exams = [x[0] for x in cursor.fetchall()][1: -4]
                            mark = []
                            for exam in exams:
                                cursor.execute(f"SELECT {exam} FROM {record}_{subject} WHERE student_id = {student_id}")
                                mark.append(cursor.fetchall()[0][0])

                            # converting to a 2d array so scikit-learn models can use them for predictions
                            mark = np.array(mark).reshape(1, -1)
                            predict_passing = round(passfail.predict_proba(mark)[0][1]*100, 2)  # proabability of passing the subject
                            predict_grade = round(overallgrade.predict(mark)[0]*100, 2)  # predicted grade of overall subject after finals
                            predict_lettergrade = None  # giving letter based grade based on predicted overall percentage after finals
                            if predict_grade >= 90:
                                predict_lettergrade = 'A+'
                            elif predict_grade >= 80:
                                predict_lettergrade = 'A'
                            elif predict_grade >= 70:
                                predict_lettergrade = 'B'
                            elif predict_grade >= 60:
                                predict_lettergrade = 'C'
                            elif predict_grade >= 50:
                                predict_lettergrade = 'D'
                            else:
                                predict_lettergrade = 'F'

                            # new predict grade with mark and letter grade
                            predict_grade = f"{predict_grade} ({predict_lettergrade})"

                            cursor.execute(f"UPDATE {record}_{subject} SET PASS_CHANCE = {predict_passing} WHERE student_id = {student_id}")
                            db.commit()

                            cursor.execute(f"UPDATE {record}_{subject} SET PREDICTED_GRADE = '{predict_grade}' WHERE student_id = {student_id}")
                            db.commit()
                            # gathering tests to calculate actual grade using subject details and actual current marks
                            cursor.execute(f"SELECT * FROM {record}_{subject} WHERE student_id = {student_id}")
                            marks = [int(x) for x in [x for x in cursor.fetchall()[0]][1:-3]]

                            cursor.execute(f"SELECT Amount FROM {subject}_details")
                            amounts = [int(x[0]) for x in cursor.fetchall()]

                            cursor.execute(f"SELECT Weightage FROM {subject}_details")
                            weightages = [float(x[0]) for x in cursor.fetchall()]

                            cursor.execute(f"SELECT Max_mark FROM {subject}_details")
                            max_marks = [int(x[0]) for x in cursor.fetchall()]

                            # calculating rolling actual grade ef student
                            mults = []
                            c = 0
                            for x in range(len(amounts)):
                                for y in range(amounts[x]):
                                    mults.append((marks[c] / max_marks[x]) * (weightages[x] / amounts[x]))
                                    c += 1
                            mults = [np.round(x*100, 2) for x in mults]
                            p_grade = np.sum(mults)
                            l_grade = None
                            if p_grade >= 90:
                                l_grade = 'A+'
                            elif p_grade >= 80:
                                l_grade = 'A'
                            elif p_grade >= 70:
                                l_grade = 'B'
                            elif p_grade >= 60:
                                p_grade = 'C'
                            elif p_grade >= 50:
                                l_grade = 'D'
                            else:
                                l_grade = 'F'

                            grade = f"{p_grade} ({l_grade})"
                            cursor.execute(f"UPDATE {record}_{subject} SET GRADE = '{grade}' WHERE student_id = {student_id}")
                            db.commit()

                            print(f"\nMark updated for student_id {student_id} in {test} for {subject}\n")
                        else:
                            print("Incorrect choice, going back...\n")
                    else:
                        print("Incorrect choice, going back...\n")
                else:
                    print("Incorrect choice, going back...\n")
            else:
                print("Incorrect choice, going back...\n")

        elif choice == '4':  # changing first_name, last_name or password of teacher account
            print("\nChanging account detail\n")
            cursor.execute("DESCRIBE teachers")
            details = list(enumerate([x[0] for x in cursor.fetchall() if x[0] not in ['teacher_id', 'username']], start=1))
            [print(f"{x[0]}. {x[1]}") for x in details]
            print(f"{len(details)+1} Go back")
            choice = input("Choice : ")
            if int(choice) in [x[0] for x in details]:
                detail = details[int(choice)-1][1]
                print(f"\nChanging {detail}\n")
                new_detail = input(f"Enter new {detail} : ")
                cursor.execute(f"UPDATE teachers SET {detail} = '{new_detail}' WHERE teacher_id = {teacher_id}")
                db.commit()
                print(f"{detail} updated\n")

            else:
                print("Incorrect choice, going back...\n")
                continue

        elif choice == '5':
            print("Viewing subject details\n")
            cursor.execute(f"SELECT id FROM subjects WHERE teacher_id = {teacher_id}")
            subjects = cursor.fetchall()
            if len(subjects) < 1:
                print("There are no subjects assigned to you\n")
            else:
                subjects = list(enumerate(subjects[0], start=1))
                [print(f"{x[0]}. {x[1]}") for x in subjects]
                choice = input("Choice : ")
                if int(choice) in [x[0] for x in subjects]:
                    subject = f"{subjects[int(choice)-1][1].lower()}_details"
                    print(pd.read_sql(f"SELECT * FROM {subject}", db, index_col="Type"))
                    print(f"Details for {subjects[int(choice)-1][1]} shown\n")
                else:
                    print("Invalid option, going back\n")

        elif choice == '6':
            print("Logging out...\n")
            break

        else:
            print("No valid option entered...\n")


def teacher_auth():
    print("\nTeacher Login\n")
    user = input("Username : ")
    passw = stdiomask.getpass(prompt='Password : ')
    cursor.execute("SELECT username FROM teachers")
    if user in [x[0] for x in cursor.fetchall()]:
        cursor.execute(f"SELECT password FROM teachers WHERE username = '{user}'")
        if passw == cursor.fetchall()[0][0]:
            cursor.execute(f"SELECT teacher_id FROM teachers WHERE username = '{user}'")
            teacher_session(cursor.fetchall()[0][0])
        else:
            print("\nWrong password...\n")
    else:
        print("\nWrong username...\n")


def admin_session():
    while True:
        print("\nWelcome to the admin menu\n")
        print("1. Set up main tables (students_{year} , teachers and subjects)")
        print("2. Show tables")
        print("3. Manage subjects")
        print("4. Manage teacher accounts")
        print("5. Manage student account")
        print("6. Logout")
        choice = input("Choice : ")

        if choice == '1':  # automatically setting up the main 3 prerequisite tables required for further operations in course
            cursor.execute("SHOW TABLES LIKE 'subjects'")
            if len(cursor.fetchall()) < 1:
                print("\nCreating teacher table")
                cursor.execute(
                    "CREATE TABLE IF NOT EXISTS teachers (teacher_id INT(5) NOT NULL AUTO_INCREMENT, first_name VARCHAR(30) NOT NULL, last_name VARCHAR(30) NOT NULL, username VARCHAR(20) NOT NULL,email TEXT(50) NULL, password VARCHAR(20) NOT NULL,  PRIMARY KEY (teacher_id), UNIQUE (username))")
                db.commit()

                print("\nCreating subjects table")
                cursor.execute(
                    "CREATE TABLE subjects (id VARCHAR(10) NOT NULL , name TEXT NOT NULL ,semester INT(2) NOT NULL , teacher_id INT(5) NULL , PRIMARY KEY (id))")
                db.commit()

                # creating foreign key for subjects and teachers for teacher_id
                cursor.execute(
                    "ALTER TABLE subjects ADD FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL ON UPDATE CASCADE")
                db.commit()

                # creating table to store students semester count, year start and expected year of graduation
                print("\nCreating students table")
                cursor.execute("CREATE TABLE students_batch (start_year INT NOT NULL AUTO_INCREMENT, grad_year INT, students INT NOT NULL DEFAULT 0, lat_students INT DEFAULT 0, tot_students INT DEFAULT 0, cur_semester INT DEFAULT 1, PRIMARY KEY (start_year))")
                db.commit()
                print("\nstudents_batch, teachers and subjects table successfully created\n")

            else:
                print("Main tables seem to already have been created\n")

        elif choice == '2':  # showing all table names and details about them
            cursor.execute("SHOW TABLES")
            tables = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
            print('')
            while True:
                print("Which table to show?\n")
                [print(f"{x[0]}. {x[1]}") for x in tables]
                print(f"{len(tables)+1}. Go back")
                choice = input("Option : ")
                if int(choice) in range(1, len(tables)+1):
                    table = tables[int(choice)-1][1]
                    print('\n')
                    print("Describing the table\n")
                    print(pd.read_sql(f"DESCRIBE {table}", db, index_col="Field"), '\n')

                    print("Details within the table\n")
                    df = pd.read_sql(f"SELECT * FROM {table}", db)
                    print(df.set_index(df.columns[0]), '\n')
                    print(f"\n{table} details shown\n")

                elif int(choice) == len(tables)+1:
                    print("\nGoing back...\n")
                    break

                else:
                    print("Enter valid choice...\n")

        elif choice == '3':  # adding or deleting a subject
            while True:
                print("Managing subjects")
                print("\n1. Add a subject")
                print("2. Delete a subject")
                print("3. Go back")
                choice = input("Choice : ")
                if choice == '1':
                    print("\nAdding a subject\n")
                    teachers = pd.read_sql("SELECT * FROM teachers", db, index_col='teacher_id')
                    if len(teachers) < 1:
                        print("There are no teachers registered in the records yet\n")
                        continue
                    else:
                        print("IS THE TEACHER THAT TEACHES THIS SUBJECT IN THE TABLE?\n")
                        print(teachers, '\n')
                    print("1. YES")
                    print("2. NO")
                    choice = input("Choice : ")
                    if choice == '1':
                        cursor.execute("SELECT id FROM subjects")
                        subject_names = [x[0] for x in cursor.fetchall()]
                        subj_name = input("Enter abbreviation of subject : ").strip()

                        if subj_name not in subject_names:
                            try:
                                full_name = input("Enter full name of subject : ").strip()
                                semester = int(input("Which semester is this subject in : "))
                                teach = int(input("Enter teacher ID for this subject : "))
                                cursor.execute(f"INSERT INTO subjects VALUES ('{subj_name}', '{full_name}', {semester}, {teach})")
                                db.commit()

                                table_name = f"{subj_name}_details"
                                # type of evaluations
                                test_type = tuple(input("\nEnter the types of tests (seperated by a space): ").split())
                                print(" ")
                                test_amount = tuple(int(input(f"How many tests for {x}?: "))
                                                    for x in test_type)  # amount of tests per evaluation
                                print(" ")
                                max_mark = tuple(int(input(f"{x} out of how many marks?: "))
                                                 for x in test_type)  # maximum marks for each type of tests
                                print(" ")

                                while True:
                                    weightage = tuple(float(input(f"What is the weightage for {x}?: "))/100 for x in test_type)
                                    if np.sum(weightage) == 1.0:
                                        print(" ")
                                        break
                                    else:
                                        print("Make sure the weightage for all tests add up to 1.0!\n")
                                pass_percent = float(input("What is the passing percentage threshold?: "))/100

                                final_test_name = test_type[-1]
                                cursor.execute(
                                    f"CREATE TABLE {table_name} (Type VARCHAR(30), Amount INT(2), Weightage FLOAT, Max_mark INT(3))")
                                db.commit()

                                passfail, overallgrade = grades(test_type, test_amount, max_mark,
                                                                weightage, pass_percent, final_test_name)

                                with open(f"{path}/{subj_name}_passfail", 'wb') as f:
                                    pickle.dump(passfail, f)

                                with open(f"{path}/{subj_name}_overallgrade", 'wb') as f:
                                    pickle.dump(overallgrade, f)

                                # inserting details about new subject
                                for x in [tuple((test_type[x], test_amount[x], weightage[x], max_mark[x])) for x in range(len(test_type))]:
                                    cursor.execute(f"INSERT INTO {table_name} VALUES ('{x[0]}', {x[1]}, {x[2]}, {x[3]})")
                                    db.commit()

                                print(f"Details for {full_name} added\n")

                                # getting all student record tables for all years
                                cursor.execute("SHOW TABLES WHERE tables_in_btechcse LIKE 'students_%' AND tables_in_btechcse NOT LIKE 'students_batch'")
                                tables = [x[0] for x in cursor.fetchall()]
                                if len(tables) > 0:
                                    # making marking sheets for subjects for all students who have a student record
                                    for table in tables:
                                        new_table = f"{table}_{subj_name}"
                                        tests = []
                                        cursor.execute(f"SELECT type, amount FROM {table_name}")
                                        # getting the column names of tests for subject
                                        for x in cursor.fetchall():
                                            if x[1] > 1:
                                                for y in range(1, x[1]+1):
                                                    tests.append(f"{x[0]}_{y}")
                                            else:
                                                tests.append(x[0])

                                        # creating new mark sheet for students doing that particular subject
                                        cursor.execute(f"CREATE TABLE {new_table} (student_id INT(5) PRIMARY KEY)")
                                        db.commit()

                                        # adding foreign key to link student ids together
                                        cursor.execute(
                                            f"ALTER TABLE {new_table} ADD FOREIGN KEY (student_id) REFERENCES {table}(id) ON DELETE CASCADE ON UPDATE CASCADE")
                                        db.commit()

                                        # adding columns of tests and making the default marks 0
                                        for test in tests:
                                            cursor.execute(f"ALTER TABLE {new_table} ADD {test} INT(3) NOT NULL DEFAULT 0")
                                            db.commit()

                                        # custom columns to store predictor variables
                                        cursor.execute(f"ALTER TABLE {new_table} ADD PASS_CHANCE FLOAT NOT NULL DEFAULT 0")
                                        db.commit()
                                        cursor.execute(f"ALTER TABLE {new_table} ADD PREDICTED_GRADE VARCHAR(10) NOT NULL DEFAULT 0")
                                        db.commit()
                                        cursor.execute(f"ALTER TABLE {new_table} ADD GRADE VARCHAR(10) NOT NULL DEFAULT 0")
                                        db.commit()

                                        # adding each student id for each student record table and this subject
                                        cursor.execute(f"SELECT id FROM {table}")
                                        for x in [x[0] for x in cursor.fetchall()]:
                                            cursor.execute(f"INSERT INTO {new_table} (student_id) VALUES ({x})")
                                            db.commit()

                                        cursor.execute(
                                            f"CREATE TRIGGER {new_table} AFTER INSERT ON {table} FOR EACH ROW INSERT INTO {new_table} (student_id) values (new.id)")
                                        db.commit()

                                    print(f"Grades sheets for {subj_name} created\n")
                                else:
                                    print("No student record tables found...\n")
                            except:
                                cursor.execute("DELETE FROM subjects WHERE id = %s", (subj_name,))
                                db.commit()
                                print("Enter valid subject details...\n")
                        else:
                            print(f"{subj_name} already exists in the subjects table...\n")
                    else:
                        print("Make sure to enter the teacher details in the teacher table first...\n")

                elif choice == '2':  # deleting a subject
                    print("\nDeleting a subject\n")
                    cursor.execute("SELECT id FROM subjects")
                    names = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
                    [print(f"{x[0]}. {x[1]}") for x in names]
                    print(f"{len(names)+1}. Go back")
                    choice = input("Choice : ")
                    if int(choice) in range(1, len(names)+1):
                        subject = names[int(choice)-1][1]
                        print(f"Deleting {subject}\n")
                        print("Are you sure?")
                        print("1. YES")
                        print("2. NO")
                        choice = input("Choice : ")
                        if choice == '1':
                            cursor.execute("DELETE FROM subjects WHERE id = %s", (subject,))
                            db.commit()

                            cursor.execute(f"SHOW TABLES LIKE '%{subject}%'")
                            tables = [x[0] for x in cursor.fetchall()]
                            if len(tables) > 0:
                                for table in tables:
                                    cursor.execute(f"DROP TABLE {table}")
                                    db.commit()

                            cursor.execute(f"SHOW TABLES WHERE tables_in_btechcse LIKE 'students_%' AND tables_in_btechcse NOT LIKE 'students_batch''")
                            records = [x[0] for x in cursor.fetchall() if len(x[0]) < 15]
                            for record in records:
                                cursor.execute(f"DROP TRIGGER IF EXISTS {record}_{subject}")
                                db.commit()

                            if os.path.exists(f"{path}/{subject}_passfail"):
                                os.remove(f"{path}/{subject}_passfail")

                            if os.path.exists(f"{path}/{subject}_overallgrade"):
                                os.remove(f"{path}/{subject}_overallgrade")

                            print(f"{subject} has been deleted\n")

                        else:
                            print("Going back...\n")
                            break

                elif choice == '3':
                    print("Going back...\n")
                    break

                else:
                    print("Invalid choice, going back...\n")

        elif choice == '4':  # managing teacher accounts
            while True:
                print("Managing teacher accounts")
                print("\n1. Add a teacher account")
                print("2. Delete a teacher account")
                print("3. Go back")
                choice = input("Choice : ")
                if choice == '1':  # deleting a teacher account
                    print("Adding a teacher\n")
                    first = input("First name : ")
                    last = input("Last name : ")
                    user = input("Enter username : ")
                    passw = input("Enter password : ")
                    cursor.execute("INSERT INTO teachers (first_name, last_name, username, password) VALUES (%s, %s, %s, %s)",
                                   (first, last, user, passw))
                    db.commit()
                    print(f"{user} has been added as a teacher\n")

                elif choice == '2':
                    print("Deleting a teacher\n")
                    cursor.execute("SELECT teacher_id FROM teachers")
                    ids = [x[0] for x in cursor.fetchall()]
                    print(pd.read_sql("SELECT * FROM teachers", db, index_col='teacher_id'))
                    choice = input("Input id of teacher to delete : ")
                    if int(choice) in ids:
                        print("Are you sure?")
                        print("1. Yes")
                        print("2. No")
                        choice = input("Choice : ")
                        if choice == '1':
                            cursor.execute(f"SELECT username FROM teachers WHERE teacher_id = {choice}")
                            user = cursor.fetchall()[0][0]
                            cursor.execute(f"DELETE FROM teachers WHERE teacher_id = {choice}")
                            db.commit()
                            print(f"{user} has been deleted from teacher records\n")
                    else:
                        print("Invalid choice, going back...\n")

                elif choice == '3':
                    print("Going back...\n")
                    break

                else:
                    print("Incorrect choice, enter a valid choice...\n")

        elif choice == '5':  # managing student account/records
            while True:
                print("Managing student accounts/records")
                print("\n1. Create new student records for new batch")
                print("2. Add a student account")
                print("3. Delete a student account")
                print("4. Go back")
                choice = input("Choice : ")
                if choice == '1':  # creating a new student records table for new batch and grade sheets
                    print("\nCreating new student record\n")
                    year = input("Enter year to create student record for : ")
                    cursor.execute("SELECT start_year FROM students_batch")

                    # checking if batch year already exists
                    if int(year) not in [x[0] for x in cursor.fetchall()]:
                        # creating entry for students batch table
                        cursor.execute(f"INSERT INTO students_batch (start_year, grad_year) VALUES ({int(year)}, {int(year) + course_len})")
                        db.commit()

                        table = f"students_{year}"  # name of new student records
                        # creating student record for that batch
                        cursor.execute(
                            f"CREATE TABLE {table}(id INT(3) NOT NULL AUTO_INCREMENT, first_name TEXT NOT NULL, last_name TEXT NOT NULL, mobile_no VARCHAR(15) DEFAULT NULL, email VARCHAR(40) DEFAULT NULL, username VARCHAR(20) NOT NULL, password VARCHAR(20) NOT NULL, start_year INT NOT NULL DEFAULT {int(year)},start_sem INT NOT NULL DEFAULT 1, cur_semester INT NOT NULL DEFAULT 1, entry VARCHAR(20) DEFAULT 'Normal', grad_year INT DEFAULT {int(year)+4}, PRIMARY KEY(id), UNIQUE(username))")
                        db.commit()

                        # foreign key referencing students_batch table to student record
                        cursor.execute(f"ALTER TABLE {table} ADD FOREIGN KEY (start_year) REFERENCES students_batch(start_year)")
                        db.commit()

                        # triggers for counting normal entry students after either deleting/updating/inserting new data for each record
                        cursor.execute(
                            f"CREATE TRIGGER normal_student_count_{year}_insert AFTER INSERT ON {table} FOR EACH ROW UPDATE students_batch SET students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Normal') WHERE start_year={int(year)}")
                        db.commit()

                        cursor.execute(
                            f"CREATE TRIGGER normal_student_count_{year}_delete AFTER DELETE ON {table} FOR EACH ROW UPDATE students_batch SET students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Normal') WHERE start_year={int(year)}")
                        db.commit()

                        # triggers for counting lateral entry students after either deleting/updating/inserting new data for each record
                        cursor.execute(
                            f"CREATE TRIGGER lateral_student_count_{year}_insert AFTER INSERT ON {table} FOR EACH ROW UPDATE students_batch SET lat_students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Lateral') WHERE start_year={int(year)}")
                        db.commit()

                        cursor.execute(
                            f"CREATE TRIGGER lateral_student_count_{year}_delete AFTER DELETE ON {table} FOR EACH ROW UPDATE students_batch SET lat_students = (SELECT COUNT(*) FROM {table} WHERE entry = 'Lateral') WHERE start_year={int(year)}")
                        db.commit()

                        # trigger to update semester count between student records table and students_batch table
                        cursor.execute(
                            f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {int(year)})")

                        # create 2 events to auto increment the semester count for each student batch
                        cursor.execute(
                            f"CREATE EVENT IF NOT EXISTS sem_count_{year}_jan ON SCHEDULE EVERY 1 YEAR STARTS '{int(year)+1}-01-01' ENDS '{int(year)+course_len}-01-01' DO UPDATE students_batch SET cur_semester = cur_semester + 1 WHERE start_year = {year}")

                        cursor.execute(
                            f"CREATE EVENT IF NOT EXISTS sem_count_{year}_aug ON SCHEDULE EVERY 1 YEAR STARTS '{int(year)+1}-08-01' ENDS '{int(year)+(course_len-1)}-08=01' DO UPDATE students_batch SET cur_semester = cur_semester + 1 WHERE start_year = {year}")

                        cursor.execute("SELECT id FROM subjects")
                        subjects = [x[0] for x in cursor.fetchall()]
                        for subject in subjects:
                            table_name = f"{subject}_details"  # name of subjects to iterate for making grade sheets
                            new_table = f"{table}_{subject}"  # name of table to store grades on subject for new students

                            tests = []
                            cursor.execute(f"SELECT type, amount FROM {table_name}")
                            # getting the column names of tests for subject
                            for x in cursor.fetchall():
                                if x[1] > 1:
                                    for y in range(1, x[1]+1):
                                        tests.append(f"{x[0]}_{y}")
                                else:
                                    tests.append(x[0])

                            # creating new mark sheet for students doing that particular subject
                            cursor.execute(f"CREATE TABLE {new_table} (student_id INT(5) PRIMARY KEY)")
                            db.commit()

                            # adding foreign key to link student ids together
                            cursor.execute(
                                f"ALTER TABLE {new_table} ADD FOREIGN KEY (student_id) REFERENCES {table}(id) ON DELETE CASCADE ON UPDATE CASCADE")
                            db.commit()

                            # adding columns of tests and making the default marks 0
                            for test in tests:
                                cursor.execute(f"ALTER TABLE {new_table} ADD {test} INT(3) NOT NULL DEFAULT 0")
                                db.commit()

                            # custom columns to store predictor variables
                            cursor.execute(f"ALTER TABLE {new_table} ADD PASS_CHANCE FLOAT NOT NULL DEFAULT 0")
                            db.commit()
                            cursor.execute(f"ALTER TABLE {new_table} ADD PREDICTED_GRADE VARCHAR(10) NOT NULL DEFAULT 0")
                            db.commit()
                            cursor.execute(f"ALTER TABLE {new_table} ADD GRADE VARCHAR(10) NOT NULL DEFAULT 0")
                            db.commit()

                            # adding each student id for each student record table and this subject
                            cursor.execute(f"SELECT id FROM {table}")
                            for x in [x[0] for x in cursor.fetchall()]:
                                cursor.execute(f"INSERT INTO {new_table} (student_id) VALUES ({x})")
                                db.commit()

                            cursor.execute(
                                f"CREATE TRIGGER {new_table} AFTER INSERT ON {table} FOR EACH ROW INSERT INTO {new_table} (student_id) values (new.id)")
                            db.commit()
                        print(f"{table} records created\n")
                    else:
                        print(f"Student record for {year} already exists\n")

                elif choice == '2':  # adding students to a record
                    print("\nAdding a student to a record\n")
                    cursor.execute("SELECT start_year FROM students_batch")
                    student_records = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
                    [print(f"{x[0]}. {x[1]}") for x in student_records]
                    print(f"{len(student_records)+1}. Go back")
                    choice = input("Year : ")
                    if int(choice) in range(1, len(student_records)+1):
                        student_record = f"students_{student_records[int(choice)-1][1]}"
                        cursor.execute(f"SELECT username FROM {student_record}")
                        existing_users = [x[0] for x in cursor.fetchall()]
                        year = student_record[len(student_record)-4:]
                        while True:
                            user = input("\nEnter username for new student : ")
                            if user not in existing_users:
                                email = f"{user}_{year}@dypiu.ac.in"
                                while True:
                                    try:
                                        sem = int(input("Which semester did the student join from? : "))
                                        if sem == 1:
                                            cursor.execute(f"DROP TRIGGER IF EXISTS semester_count_{year}")

                                            cursor.execute(
                                                f"INSERT INTO {student_record} (first_name, last_name, email, username, password) VALUES ('{input('Enter first name: ')}', '{input('Enter last name: ')}', '{email}', '{user}', '{input('Enter password: ')}')")
                                            db.commit()

                                            # trigger to update semester count between student records table and students_batch table
                                            cursor.execute(
                                                f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {int(year)})")
                                            break
                                        else:
                                            cursor.execute(f"DROP TRIGGER IF EXISTS semester_count_{year}")

                                            cursor.execute(
                                                f"INSERT INTO {student_record} (first_name, last_name, email, username, password, start_sem, entry) VALUES ('{input('Enter first name: ')}', '{input('Enter last name: ')}', '{email}', '{user}', '{input('Enter password: ')}', {sem}, 'Lateral')")
                                            db.commit()

                                            cursor.execute(
                                                f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {year})")
                                            break
                                    except:
                                        print("Enter valid value for semester\n")

                                cursor.execute(f"SELECT students + lat_students FROM students_batch WHERE start_year = {int(year)}")
                                tot_students = cursor.fetchall()[0][0]
                                cursor.execute(
                                    f"UPDATE students_batch SET tot_students = {tot_students} WHERE start_year = {int(year)}")
                                db.commit()
                                print(f"\n{user} has been added as a student\n")
                                break
                            else:
                                print(f"\n{username} already exists, try again\n")
                    else:
                        print('\nInvalid choice, going back...\n')
                        continue

                elif choice == '3':  # deleting a student from a record
                    print("\nDeleting a student from a record\n")
                    cursor.execute("SELECT start_year FROM students_batch")
                    student_records = list(enumerate([x[0] for x in cursor.fetchall()], start=1))
                    [print(f"{x[0]}. {x[1]}") for x in student_records]
                    print(f"{len(student_records)+1}. Go back")
                    choice = input("Which batch? : ")
                    if int(choice) in range(1, len(student_records)+1):
                        year = student_records[int(choice)-1][1]
                        student_record = f"students_{year}"
                        user = input("Input username of student to delete : ")

                        cursor.execute(f"SELECT username FROM {student_record}")
                        users = [x[0] for x in cursor.fetchall()]
                        if user not in users:
                            print(f"Invalid user: {user}, going back...\n")
                            continue

                        print(f"\nAre you sure to delete {user}?")
                        print("1. YES")
                        print("2. NO")
                        choice = input("Choice : ")
                        if choice == '1':
                            cursor.execute(f"DROP TRIGGER IF EXISTS semester_count_{year}")

                            cursor.execute(f"DELETE FROM {student_record} WHERE username = '{user}'")
                            db.commit()

                            cursor.execute(
                                f"CREATE TRIGGER semester_count_{year} AFTER UPDATE ON students_batch FOR EACH ROW UPDATE students_{int(year)} SET cur_semester = (SELECT cur_semester FROM students_batch WHERE start_year = {year})")

                            print(f"\n{user} has been removed from {student_record}\n")

                            cursor.execute(f"SELECT students + lat_students FROM students_batch WHERE start_year = {int(student_records[int(choice)-1][1])}")
                            tot_students = cursor.fetchall()[0][0]
                            cursor.execute(
                                f"UPDATE students_batch SET tot_students = {tot_students} WHERE start_year = {int(student_records[int(choice)-1][1])}")
                            db.commit()
                        else:
                            print("\nGoing back...\n")
                            continue
                    else:
                        print("\nInvalid choice, going back...\n")
                        continue

                elif choice == '4':
                    print("Going back...\n")
                    break

                else:
                    print("Enter a valid option...\n")

        elif choice == '6':
            print("Logging out...\n")
            break

        else:
            print("Incorrect choice, enter valid choice...\n")


def admin_auth():
    print("\nAdmin Login\n")
    if input("Username : ") == 'admin' and stdiomask.getpass(prompt='Password : ') == '1234':
        admin_session()
    else:
        print("Invalid admin login details\n")


def main():
    while True:
        print("B.Tech CSE Main Menu\n")
        print("1. Log in as admin")
        print("2. Log in as student")
        print("3. Log in as teacher")
        print("4. Exit")
        choice = input("Choice : ")

        if choice == '1':
            admin_auth()

        elif choice == '2':
            student_auth()

        elif choice == '3':
            teacher_auth()

        elif choice == '4':
            print("Bye!")
            break


# war begins, ionia calls. hasagi
main()
